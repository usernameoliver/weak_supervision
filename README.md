# Weakly supervised learning on medical images with inaccurate labels 
This repository is a PyTorch implementation of influence function applied to InceptionV3 models. The influence function plug-in can also be applied to other deep convolutional neural networks(DCNN) for mislabeled imaging data identification.

## Setup
First, install required Python packages with the require.txt in Anaconda Environment.
```
conda create --name myenv
pip install -r requirements.txt
```

Next, put the dataset into the data/ folder under weak\_supervision directory. Please change BASE\_DIR in each py file to your working directory.

## Uage
When the ground truth labels exists, we manually create mislabels by flipping labels in a small portion of the training data. Here the percentage of mislabels is 20%, i.e. 0.2 in the commands below. Note that 0.2 can be changed to any number between 0.0 and 1.0. 
```
cd weak_superviion/training
python flipped_label.py 0.2
```

A DCNN model can be trained with the following command:
```
cd weak_supervision/
python -m training.inceptionV3 0.2
```

To calculate the influence of training data on the model:
```
cd weak_superviion/
python -m influence_function.inceptionV3_influence 0.2
```

To calculate the percentage of mislabels identified by our method:
```
cd weak_superviion/
python -m influence_function.inceptionV3_influence 0.2
python mislabel_identification.py
```

To compare with a previous method, i.e., ensemble learning with multiple majority filter(mmf):
```
cd weak_superviion/evaluation/mmf/
python mmf_ddsm.py
```

### Reference 
*Inaccurate labels in weakly-supervised deep learning: Automatic identification and correction and their impact on classification performance*.


