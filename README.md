# weakly supervised learning on images with inaccurate labels 
PyTorch implementation of influence function plugged into InceptionV3 models. The influence function plug-in can be applied to other deep convolutional neural networks(DCNN) for mislabeled imaging data identification.

## Setup
pip install --process-dependency-links -e .
Put the dataset into the data/ folder.

## Uage

When the ground truth labels exists, we manually create mislabels by flipping labels in a small portion of the training data. Here the percentage of mislabels is 20%, i.e. 0.2 in the commands below. 
```
cd weak_superviion/training
python flipped_label.py 0.2
```

A DCNN model can be trained with the following command,
```
cd weak_supervision/
python -m training.inceptionV3 0.2
```


To calculate the influence of training data on the model,
```
cd weak_superviion
python -m influence_function.inceptionV3_influence 0.2
```



### Reference 
*Inaccurate labels in weakly-supervised deep learning: Automatic identification and correction and their impact on classification performance*.


