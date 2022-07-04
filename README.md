# robocup2022-cv-pipeline

## Environment setting
* create environments
```
$ conda create --name vild
```
* Next, install PyTorch 1.7.1 (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:
``` bash
$ conda install --yes pytorch torchvision cudatoolkit=11.6 cudnn=8.1.0 -c pytorch -c conda-forge
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```
> Replace `cudatoolkit=11.6` and `cudnn=8.1.0` above with the appropriate CUDA version on your machine or `cpuonly` when installing on a machine without a GPU.
* Then, install dependencies packages
``` 
$ pip install -r requirements.txt
```
* Lastly, install tensorflow
```
$ pip install tensorflow
```

## Useful Link
* Check TensorFlow version is compatible with your CUDA and cuDNN version: https://www.tensorflow.org/install/source_windows#gpu
* Install previous versions of PyTorch: https://pytorch.org/get-started/previous-versions/
* Chack your CUDA version: https://stackoverflow.com/a/9730706
