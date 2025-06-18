<h1 align="center"> Webcam Live Portrait Interface</h1>

This fork is designed to run the LivePortrait model live on webcam feed with a Flask API interface for Web GUI frontend.

`interference.py` holds the object where the API call flows are managed. 

# Prerequisites

For this repository to function properly, it needs to be a submodule / subdirectory of a main project where

* Main Project
  * Backgrounds
  * photos
  * Webcam_Live_Portrait_Flask

Where Backgrounds folder holds images which can be any size in JPG format.

Virtual background images can be any name which should be input into the `Inference.background_images` list in the `inference.py` script.

For image overlay over green screen images, images should be placed in Backgrounds folder with names `meeting-greenX.jpg` where X can be any integer.

# Running the API and the Inference

To start an inference session that will be controlled with API inputs, you require a running script. An example script can be found in this directory as `example_run.py`.

# Requirements 

* **Ubuntu 22.04**
* **Python 3.10**
* **Cuda 11.8** with **cuDNN 8.8** / Nvidia GPU
* **v4l2loopback** for Linux.
* [modnet_photographic_portrait_matting.onnx](https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/blob/034769305faf641ad94edfac654aba13be06e816/modnet_photographic_portrait_matting.onnx) in `pretrained_weights/modnet/` path
* [This drive](https://drive.google.com/drive/folders/1UtKgzKjFAOmZkhNK-OYT0caJ_w2XAnib) unzipped into the pretrained_weights folder (With all the subfolders).

# Install

* Install Cuda 11.8 Toolkit and place cuDNN 8.8  into the CUDA folder.
* `python -m venv .venv`

  `source .venv/bin/ activate`

  `pip install -r requirement.txt`

  `pip uninstall pytorch torch vision torchaudio`

  `pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118`

  `pip uninstall opencv-python`

  `pip install https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.6.0.20221107/opencv_contrib_python_rolling-4.6.0.20221107-cp36-abi3-win_amd64.whl`

  