# VroomNet

Image Classification Model for the [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), and submission for the [Grab AI for S.E.A.](https://www.aiforsea.com/computer-vision) Computer Vision challenge. This solution is built in large part using the [fast.ai](https://docs.fast.ai/index.html) deep learning framework.

## Table of Contents

<!-- TOC -->

- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
    - [Download Trained Models](#download-trained-models)
    - [Inference](#inference)
    - [Training](#training)
- [Approach](#approach)
- [Credits](#credits)

<!-- /TOC -->
## Results
Summary of final results on the test dataset. The models achieve higher test accuracy with Test-time Augmentation.

| Model        | Accuracy | Accuracy w/ TTA |
|--------------|----------|-----------------|
| Ensemble     | 0.908    | 0.915           |
| SEResNeXt101 | 0.892    | 0.901           |

## Requirements
The easiest way to set up is to clone this repo, create a fresh Miniconda environment with Python 3.6,

```
$ conda create -n VroomNet python=3.6
```

Install the essential data science libraries,

```
$ conda install pandas scikit-learn matplotlib notebook
```

The latest versions of PyTorch and fastai,

```
$ conda install -c pytorch -c fastai fastai
```

And pretrained model implementations from [Cadene](https://github.com/Cadene/pretrained-models.pytorch). 

```
$ pip install pretrainedmodels
```

## Usage

### Download Trained Models
Download all my trained models from this [Dropbox](https://www.dropbox.com/sh/xtvbx7vj8ru9401/AAALEcsMZBDhhQpzYH6aQf0wa?dl=0), and place them in the `saved` folder.

```
├───saved
│   ├───dpn92.pkl
│   ├───inceptionv4.pkl
│   ├───se_resnext101.pkl
│   ├───...
```

### Inference
To perform inference on a batch of images, simply place all your images in the `data/test` folder,

```
├───data
│   ├───test    <- Here
│   ├───...
```

And run the `predict.py` script,

```
$ python -m src.predict -e -t
```

The probability scores for each class  will be saved in the root directory as `predictions.csv` by default, in this format:

| img_name  | AM General Hummer SUV 2000 | Acura Integra Type R 2001 | ... |
|-----------|----------------------------|---------------------------|-----|
| 02485.jpg | 0.000401                   | 0.00009.71                | ... |
| ...       | ...                        | ...                       | ... |

The highest accuracy on the test dataset is achieved with Ensembling and Test-time Augmentation. You can toggle the `-e` and `-t` flags to enable Ensembling and Test-time Augmentation respectively (inference time will be faster without them toggled on). Without Ensembling, only the SEResNeXt101 model will be used for inference.

### Training
If you wish to replicate my training results, first download the Stanford Cars [training](http://imagenet.stanford.edu/internal/car196/cars_train.tgz) and [testing](http://imagenet.stanford.edu/internal/car196/cars_test.tgz) images and place them in the `data/train` and `data/test` folders respectively.

```
├───data
│   ├───test    <- Testing images here 
│   ├───train   <- Training images here 
│   ├───...
```

All the configurations for my final models are stored in `src/configs`. 

```
└───src
    ├───configs
    │   ├───dpn92.json
    │   ├───inceptionv4.json
    │   ├───se_resnext101.json
    │   ├───...
```

To train each individual model, access the [training notebook](notebooks/train_expt.ipynb), specify the training configuration file to use and then run the entire notebook. 

## Approach
My final ensemble of models consists of SEResNeXt-101, Inception-v4 and DPN-92. Here is a quick listing of the techniques experimented with:
- Data augmentation
- Mixup augmentation
- One-cycle policy
- Transfer learning w/ discriminative layer training
- Test-time augmentation
- Ensembling using soft voting

See my [Report](report.md) for a more detailed discussion of my approach. 
    
## Credits
- [Cadene](https://github.com/Cadene/pretrained-models.pytorch) for pretrained model implementations in PyTorch.
