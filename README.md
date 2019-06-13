# VroomNet

Image Classification Model for the [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), and submission for the [Grab AI for S.E.A.](https://www.aiforsea.com/computer-vision) Computer Vision challenge. This solution is built in large part using the [fast.ai](https://docs.fast.ai/index.html) framework.

## Table of Contents

## Requirements
The easiest way to set up is to create a fresh Miniconda environment with Python 3.6,

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

And pretrained model implementations from [Cadene](https://github.com/Cadene/pretrained-models.pytorch) and [lukemelas](https://github.com/lukemelas/EfficientNet-PyTorch) if you wish to replicate my training results (not required for inference).

```
$ pip install pretrainedmodels
$ pip install efficientnet_pytorch
```

## Usage

### Download Trained Models

### Inference
To perform inference on a batch of images, simply place all your images in the `test` folder,

```
├───data
│   ├───test    <- Here
│   ├───train
│   ├───...
```

And run the `predict.py` script,

### Evaluation

### Training

## Approach

## Results

## Project Structure
```
├───report.md
│
├───data
│   ├───test
│   ├───train
│   ├───test_labels.csv
│   └───train_labels.csv
│
├───notebooks
│
├───saved
│ 
└───src
    ├───configs
    │   └───constants.py
    │   
    ├───dataloaders
    │   
    ├───evaluators
    │   
    ├───models
    │   
    ├───trainers
    │   
    ├───utils
    │   
    └───
```
    
## Credits
- Project Structure
- pretrainedmodels

## License
