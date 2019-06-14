# Grab AI for S.E.A.: Computer Vision Challenge

![Grab AI for S.E.A. Computer Vision Challenge](https://static.wixstatic.com/media/397bed_b98b08c6fc6848d1b280cc16d5462818~mv2.png/v1/fill/w_305,h_305,al_c,q_80,usm_0.66_1.00_0.01/Grab%20EDM_Computer%20Vision.webp)

*Taken from: [Grab AI for S.E.A.](https://www.aiforsea.com/computer-vision)*

**Problem Statement**  
> Given a dataset of distinct car images, can you automatically recognize the car model and make?

**Introduction**  
This challenge is a Fine-Grained Image Classification task; where a model has to be built to differentiate between hard-to-distinguish object classes, for e.g. the various makes or models of vehicles.

Such an endeavor can prove to be challenging as objects that belong to different classes may only contain subtle differences. In the context of this challenge, my trained models have the most difficulty in differentiating between cars of the same make and model, but produced in different years - a task only fit for the most die-hard of car enthusiasts.

**Dataset**  
The dataset for this challenge is the [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). It contains a total of *16,185* images, split into a training set of *8,144* images and a testing set of *8,041* images. The images are labeled at the level of Make, Model, Year, amounting to a total of *196* classes, and additionally contains bounding box labels that localizes the car in each image.

**Evaluation**  
The model should output a confidence score for every classification, and will be evaluated by *accuracy*, *precision*, and *recall*.

**Table of Contents**
<!-- TOC -->

- [Grab AI for S.E.A.: Computer Vision Challenge](#grab-ai-for-sea-computer-vision-challenge)
    - [Summary of Approach](#summary-of-approach)

<!-- /TOC -->    

## Approach    
    
### Summary 

I did not attempt to reinvent the wheel; rather, I took this challenge as a learning opportunity for me to, 
1. Familiarize myself with the popular and proven approaches for image classification, 
2. Successfully implement an end-to-end machine learning pipeline, and to
3. Attempt to document my process and results as closely as possible.

My solution was built in large part using the [fast.ai](https://docs.fast.ai) deep learning framework, utilizing their baked in "best practices" to whip out a well-performing model as quickly as possible.

I started off with a tried-and-tested ResNet50 model as a simple baseline, supplemented it with various approaches to get sense of what works, then finally proceeded to test out larger and more complicated architectures.

Here is a **TD;LR** of my final approach:

**Validation Scheme**  
- Simple 20% hold-out validation set.
- Split using stratified sampling to preserve class proportions. 

**Data Preprocessing**  
- Images resized to 224 x 244.
- Normalized using ImageNet statistics to feed into a pretrained model.
- Train-time traditional data augmentation + mixup data augmentation.

**Model Architectures**  
- ResNet50, ResNet101
- SEResNeXt101
- InceptionNet
- PNASNet

**Optimization**  
- Learning rate finder
- One-cycle policy
- Adam optimizer 

**Training Regime**  
- Stage-1: Transfer learning, only train classifier head.
- Stage-2: Fine-tune entire model, using discriminative layer training.

**Post-processing**  
- Test-time augmentation.

**Ensembling**  
- Ensembled using soft voting of classification scores.

### Data Preprocessing
My data preprocessing primarily comprises steps required to feed my images into a model pretrained using the [ImageNet](www.image-net.org) dataset, namely by resizing my images to 224 x 224, and the normalizing of pixel values using the ImageNet dataset statistics.

[insert examples of data augmentation here]

Data augmentation is carried out to artificially increase the size of the dataset and to act as a regularization technique. It works by performing random transformations (flips, rotations, zooms, lighting, etc.) to generate many more realistic variants of each training image  - the images fed into the model should still look like what the model would encounter during deployment. For example, an image of a car flipped horizontally is still a car, but an image of a car flipped vertically probably won't be a very useful training instance for the model; save for identifying overturned cars during accidents.

Another data augmentation technique already implemented by the [fast.ai]() library called [mixup](), is also experimented with. mixup is also a regularization technique that aims to improve the generalizability of models. It achieves this by "mixing up" two images in the form of a linear combination of their pixel values, and the target that is assigned to this new image is also the same combination of the two orginal targets.

### Model Architectures

#### SENet

[insert diagram of SE block here]

The [Squeeze-and-Excitation Network (SENet)]() is the winning architecture for ILSVRC 2017. It builds upon existing architectures like InceptionNet and ResNet by arming a mini neural network, called a SE Block, to every unit in the original architecture.

Roughly speaking, the SE Blocks improves performance by recalibrating the feature maps produced, through learning which features are most often activated together. For example, since mouths, noses and eyes frequently appear together in images, if the block sees that any two of these three features are strongly activated, it will act to increase the activation of the last feature map.

#### Inception

#### PNASNet 

### Optimization
The training technique adopted is the [1cycle policy]() proposed by Leslie Smith and evangelized by Jeremy Howard in his [fast.ai course](). This learning rate scheduler recommends a cycle with two steps of equal lengths over the total number of epochs, climbing from a lower learning rate to a higher one, and then back again to the minimum. 

[insert learning rate curve here]

The maximum learning rate value is chosen by performing a mock training session - gradually increasing the learning rate and plotting the losses after each iteration. The value that corresponds to the steepest slope on the loss curve, (somewhere before the minimum, where the loss is still improving) will be chosen.

[insert learning rate finder curve here]

The appeal behind that 1cycle policy is that it allows the model to converge to the same performance in a relatively lesser number of epochs.

The [Adam]() optimizer is chosen by default. 

### Training Regime
The de facto approach to obtain favourable results on your image classifier as quickly as possible is through transfer learning. The premise is simple: instead of starting from scratch, why not we grab a model already pretrained a large corpus of images as a starting point (the most accessible of which being [ImageNet]())? Of course, the more similar that image dataset is to your own problem task, the better. Even if your task is highly specific, transfer learning still works most of the time as it has been demonstrated that the earlier layers extract local, highly generic feature maps such as visual edges, colors, and textures. 

Transfer learning will not only speed up training considerably, but will also require much less training images.

In order to reuse a pretrained model and finetune it carefully, my training regime consists of two stages:

#### Stage-1: Transfer Learning

Selecting a model architecture with its weights already pretrained on the ImageNet dataset, the last few layers of the model are cut off to attach a custom classifier head for our problem task (i.e. 196 classes of cars instead of 1000 classes for ImageNet).

[insert example architecture of transfer learning]
