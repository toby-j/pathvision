# Pathvision: An Explainable AI Method for Object Detection models

Using a Kalman Filter based object-spesific tracking algorithm with gradient-based 

![Pathvision gif demo](docs/assets/demo.gif)

## Prerequisites
- Python 3.9+
- Pathvision's only been developed and tested on **Microsoft Windows Only**.
- pip 22.3.1+
- setuptools 67.7.2+

## Setup
These instructions are for **Microsoft Windows 10/11** (However may also work on previous versions)

### **Step 0: Clone the repository**
Navigate to the root direction and run:
```bash
pip install .
```
This will run the `setup.py` file which'll install all of Pathvision's dependencies as spesified in `requirements.txt`.

### **Step 1: Build Detectron2**
This repository is using Detectron2 for image segmentation, this isn't built by default.
Navigate to the Detectron2 folder:
```
cd pathvision/models/detectron2
pip install .
```
We've now built and install the dependencies for Pathvision.

## Running
There's a `tester.py` file which creates an instance of Pathvision's Object Detection feature and runs a predefined test using images from `debug/test_data`. 
If you want to use your own test images, simply add a folder in `debug/test_data` and change the folder path of `frame_list_dir` in `tester.py` to the folder containing your test images (.jpg and .png are both supported).

You can select either `VanillaGradients`, `IntegratedGradients` or `Smoothgrad`. If you want fast results, usually a minute or so per frame, use vanilla gradients. Smoothgrad is usually a couple of minutes. Integrated Gradients is *much* more computationaly expensive, but is supported.

To view results, the output images will be written to `debug/final_output` and `debug/output_image`. Errors will be written in the `debug/log` folder.
