# Stable diffusion walker

Notebook for generating sequences of images and rendering a video

To run:
(see e.g. https://rentry.org/SDInstallGuide for reference)
* Clone basu jindal's [optimized stable diffusion](https://github.com/basujindal/stable-diffusion) repo, e.g. run ``git clone https://github.com/basujindal/stable-diffusion`` in empty folder, or download and unzip
* Create and activate conda environment
* Get the model weights (see below)
* Copy this notebook and txt2img_generator.py to the /optimzedSD/ subfolder.

Get model weights (~4GB):
* https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4.ckpt
* paste to (your SD root folder)/models/ldm/stable-diffusion-v1/
* rename the model to model.ckpt