# classification-ensembles
The repository was created to experiment with using image augmentations for creating ensemble classification.
The ensemble can be used for improving model inference quality and for training using partially unsupervised data as suggested in (for detection and pose estimation) in :
https://arxiv.org/pdf/1712.04440.pdf

#### Experiments:
- Inference using existing network, translating ImageNet results to desired set of classes
- Inference using ensembles by augmenting evaluation images 
- Transfer learning training by changing last layer(s)
- Training with partially unsupervised data as suggested in the article (using an auxiliary trained network for tagging the unserperised data)
  - continue training
  - start training from scratch

#### Image augmentations:

A bit of experimentation was conducted with the following techniques and their combination:
- resize & rescale (keep aspect ratio, when changing size)
- flip
- crops
- color augmentations 

From some test it seems that:
* For a single net it seems that rescaling image is better then resizing to fit network input. 
* For an ensemble A combination of flip and crops using Pytorch's built in TenCrop from a rescaled resolution close to target resolution (e.g. rescale to 240x240 and apply TenCrop for 224x224 )

#### Ensemble aggregation methods:
- Majority vote: find most common index of argmax of each result
- Average: argmax of average of all classification results after softmax

No clear winner between the two.

#### Data set: Cats vs Dogs

From the Cat vs Dogs train dataset a few very small subsets were created for training, unsupervised training and evaluation.
https://www.kaggle.com/c/dogs-vs-cats

The smaller dataset were created as even the smallest classification networks trained on ImageNet get very high scores (>95% accuracy) even withou transfer learning, which didn't give a lot of margin to test improvements.

#### Network: squeezenet1_0
Again, very light network for lower initial accuracy. 

# Code and how to run
Currently from scripts, no point converting to args with the current code size.

## eval.py
Enables running a simple and augmentations ensemble inference.

## class_trainer.py
Enables running simple training for transfer learning and combined training with partial unsupervised data which is tagged using a data augmentation ensemble (with the same or different network)


## Results:
(2500 images sub set)

* ImageNet inference 84.36%
* ImageNet inference With TenCrop augmentations (rescale to 240 then ten crop to 224) and majority vote: 84.4%
* Transfer Learning training on 100 images: 91.04 %
* Transfer Learning training on 100 images with TneCrop and majority: 91.76% 
* Continue training with additional untagged 100 images: 91.88%
* Train from scrath with additional untagged 100 images: 91.36%