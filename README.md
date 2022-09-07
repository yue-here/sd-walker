# Stable diffusion walker

Notebook for generating sequences of images and rendering a video

To run:
* Clone basu jindal's [optimized stable diffusion](https://github.com/basujindal/stable-diffusion) repo, e.g. run ``git clone https://github.com/basujindal/stable-diffusion`` in empty folder
* Get the model weights (see below)
* Copy this notebook and txt2img_generator.py to the /optimzedSD/ subfolder.

Get model weights (~4GB):
* https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4.ckpt
* paste to (your SD root folder)/models/ldm/stable-diffusion-v1/
* rename the model to model.ckpt

Note: txt2img_generator.py is a modified version of txt2img_gradio.py that allows the generate function to be called externally with a module import.