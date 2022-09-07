import glob
import pickle
import os, re
import time
from random import randint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice

from omegaconf import OmegaConf
from einops import rearrange
import torch
from torch import autocast
from torchvision.utils import make_grid
from torchvision.io import read_image
from pytorch_lightning import seed_everything
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger
from transformers import logging
logging.set_verbosity_error()


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

config = "../optimizedSD/v1-inference.yaml"
ckpt = "../models/ldm/stable-diffusion-v1/model.ckpt"
sd = load_model_from_config(f"{ckpt}")
li, lo = [], []
for key, v_ in sd.items():
    sp = key.split(".")
    if (sp[0]) == "model":
        if "input_blocks" in sp:
            li.append(key)
        elif "middle_block" in sp:
            li.append(key)
        elif "time_embed" in sp:
            li.append(key)
        else:
            lo.append(key)
for key in li:
    sd["model1." + key[6:]] = sd.pop(key)
for key in lo:
    sd["model2." + key[6:]] = sd.pop(key)

config = OmegaConf.load(f"{config}")

model = instantiate_from_config(config.modelUNet)
_, _ = model.load_state_dict(sd, strict=False)
model.eval()

modelCS = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()

modelFS = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)
modelFS.eval()
del sd


def generate(
    prompt = "an astronaut riding a dragon",
    ddim_steps = 50,
    n_iter = 1,
    batch_size = 1,
    Height = 512,
    Width = 512,
    scale = 7.5,
    ddim_eta = 0.1,
    unet_bs = 1,
    device = "cuda",
    seed = 42,
    seed_step = 1000,
    sequence_step = None,
    outdir = "../output/txt2image-samples",
    sample_path = None,
    img_format = "png",
    turbo = True,
    full_precision = False,
    sampler = "ddim",
):

    C = 4
    f = 8
    start_code = None
    model.unet_bs = unet_bs
    model.turbo = turbo
    model.cdevice = device
    modelCS.cond_stage_model.device = device

    if seed == "":
        seed = randint(0, 1000000)
    seed = int(seed)
    seed_everything(seed)
    # Logging
    logger(locals(), "logs/txt2img_gradio_logs.csv")

    if device != "cpu" and full_precision == False:
        model.half()
        modelFS.half()
        modelCS.half()

    tic = time.time()
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    # Only generate a sample path if one is not provided
    if sample_path is None:
        sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompt)))[:150]

    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    
    # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    assert prompt is not None
    data = [batch_size * [prompt]]

    if full_precision == False and device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    all_samples = []
    seeds = ""
    with torch.no_grad():

        all_samples = list()
        for _ in trange(n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = None
                    if scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    shape = [batch_size, C, Height // f, Width // f]

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    samples_ddim = model.sample(
                        S=ddim_steps,
                        conditioning=c,
                        seed=seed,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        x_T=start_code,
                        sampler = sampler,
                    )

                    modelFS.to(device)
                    print("saving images")
                    for i in range(batch_size):

                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        all_samples.append(x_sample.to("cpu"))
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")

                        # Save images. Note - :05 pad to 5 with zeros
                        if sequence_step is None:
                            img_path = os.path.join(sample_path, "seed_" + str(seed) + "_" + f"{base_count:05}.{img_format}")
                        else:
                            img_path = os.path.join(sample_path, str(sequence_step) + "_" + str(seed) + f".{img_format}")

                        Image.fromarray(x_sample.astype(np.uint8)).save(img_path)

                        seeds += str(seed) + ","
                        seed += seed_step
                        base_count += 1

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    del samples_ddim
                    del x_sample
                    del x_samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    toc = time.time()

    time_taken = (toc - tic) / 60.0

    # # Making a grid
    # grid = torch.cat(all_samples, 0)
    # grid = make_grid(grid, nrow=n_iter)
    # grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()

    # Don't return text
    # txt = (
    #     "Samples finished in "
    #     + str(round(time_taken, 3))
    #     + " minutes and exported to "
    #     + sample_path
    #     + "\nSeeds used = "
    #     + seeds[:-1]
    # )

    # return Image.fromarray(grid.astype(np.uint8)), txt
    return all_samples


def sequence_gen(
    prompt,
    seq,
    project_name,
    ddim_steps=50,
    batch_size=9,
    Width = 512, 
    Height = 512,
    seed=12345,
    seed_step=10000,
    return_tensors=False,
    ):
    
    images = []

    for i in seq:
        image = generate(
            prompt = f"{prompt}, {i}",
            ddim_steps=ddim_steps,
            batch_size=batch_size, 
            Width = Width, 
            Height = Height,
            sample_path = f"../output/{project_name}",
            sequence_step=i,
            seed = seed,
            seed_step = seed_step
        )
        if return_tensors:
            images.append(image)
    if return_tensors:
        return images

def images2grid(image, nrow=3):
    """
    Convert a list of raw image values to a grid
    """
    grid = torch.cat(image, 0)
    grid = make_grid(grid, nrow=nrow)
    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
    return grid.astype(np.uint8)

def imageseq2grid(images, nrow=3):
    return [images2grid(i, nrow) for i in images]

def save_images(images, project_name):
    with open(f'../output/{project_name}.pkl', 'wb') as f:
        pickle.dump(images, f)
        
def make_video(grid, seq, project_name, Width=512, Height=512, nrow=3, fontsize=40):
    fig, ax = plt.subplots(
        figsize=(10, 11), 
        facecolor='black',
        )
    plt.tight_layout()

    ims=[]

    for n, i in enumerate(grid):
        im = ax.imshow(i, animated=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f"{project_name}", color='white', fontsize=fontsize)
        txt_x = Width*nrow // 2
        txt_y = Height*nrow + 100
        ttl = plt.text(txt_x, txt_y, seq[n], ha="center", color='white', fontsize=fontsize)
        ims.append([im, ttl])

    ani = animation.ArtistAnimation(fig, ims, interval=400, blit=True, repeat_delay=1000)
    ani.save(f"../output/{project_name}.mp4")

def folder2video(project_name, seq):
    project_name = project_name
    folder = glob.glob(f"../output/{project_name}/*")

    current = []
    for i in seq:
        # print(i)
        entry = []
        for j in folder:
            # print(j)
            if i in j:
                img = read_image(j)
                entry.append(img)
        grid = make_grid(entry, nrow=3)
        grid = rearrange(grid, "c h w -> h w c")

        current.append(grid)

    make_video(current, seq, project_name)
            