'''
Demo usage of influence function with an inception V3 model. 
The influence of each imaging data is output into a json file. 

@Auther: Degan Hao
@Date: 02/03/2020

'''
import numpy as np
import torch
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from torch import nn as nn
from influence_function.misc_functions import preprocess_image, feed_data_to_model 
from influence_function.influence import layer_wise_influence
import os.path
import pickle
import json
import time
import sys


#BASE_DIR = '/pylon5/ca5phjp/deh95/mi/'#Change the BASE_DIR to your working directory
BASE_DIR = './'#Change the BASE_DIR to your working directory
def load_inceptionV3(flip_percentage):
    print('load the model...')
    #inceptionV3 model trained with mislabel loss, large dataset
    model_path = BASE_DIR +  'models/' + str(flip_percentage) + 'percent_flip_inceptionV3_influence_bc_entire_model.pt'

    model = torch.load(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

def preprocess_inceptionV3(img_path):
    #preprocessing
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    #load image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = Image.open(img_path)
    img = img.convert("RGB")
    img_tensor = preprocess(img)
    img_tensors = img_tensor.unsqueeze(0)
    inputs = Variable(img_tensors).cuda()
    return inputs

def calc_inf_inceptionV3(flip_percentage):
    #inceptionV3
    print('start loading model')
    inceptionV3_model = load_inceptionV3(flip_percentage)
    layer_options = [-1] 
    num_layers = len(layer_options) 
    verbose = False 
    
    print('start loading data')
    label_original_train = pickle.load(open(BASE_DIR + 'training/label_dict/' + str(flip_percentage) + 'percent_original_label_train.pkl', 'rb'))
    label_flipped_train = pickle.load(open(BASE_DIR + 'training/label_dict/' + str(flip_percentage) + 'percent_flipped_label_train.pkl', 'rb'))
    label_original_val = pickle.load(open(BASE_DIR + 'training/label_dict/' + str(flip_percentage) + 'percent_original_label_val.pkl', 'rb'))
    label_flipped_val = pickle.load(open(BASE_DIR + 'training/label_dict/' + str(flip_percentage) + 'percent_flipped_label_val.pkl', 'rb'))
    label_original_dict = {**label_original_train, **label_original_val}
    label_flipped_dict = {**label_flipped_train, **label_flipped_val}
    num_images = 5 
    count = 0
    infs = np.zeros(num_layers)
    data = []
    for img_name in label_flipped_train:
        if count % 100 == 0:
            print(str(count) + 'images processed')
        count += 1
        original_label = label_original_dict[img_name]
        flipped_label = label_flipped_dict[img_name]
        img_path_B = BASE_DIR + 'training/data_breast_density/train/B/' + img_name
        img_path_C = BASE_DIR + 'training/data_breast_density/train/C/' + img_name
        if os.path.exists(img_path_B):
            img_path = img_path_B
        elif os.path.exists(img_path_C):
            img_path = img_path_C
        else:
            print('image does not exist, will skip ' + img_path_B)
            continue

        inputs = preprocess_inceptionV3(img_path)

        img_path_test = BASE_DIR + 'training/data_breast_density/train/B/2_Case1000_breast_segmentation.jpg'
        img_name_test = '2_Case1000_breast_segmentation.jpg'  
        inputs_test = preprocess_inceptionV3(img_path)
        original_label_test = label_original_dict[img_name_test]
        loss_test, params_test = feed_data_to_model(inceptionV3_model, inputs_test, original_label_test)

        loss, params = feed_data_to_model(inceptionV3_model, inputs, flipped_label)
        for j in range(num_layers):
            nth_layer = layer_options[j] 
            infs[j] = layer_wise_influence(params, loss, nth_layer, verbose, params_test, loss_test) 
            print(str(nth_layer) + ' influence = ' + str(infs[j]))
            print('-' * 20)

        item = {
            'image_name' : img_name,
            'original_label' : original_label,
            'flipped_label' : flipped_label,
            'influence' : [infs[0]]
        }
        data.append(item)

    infs_path = '/pylon5/ca5phjp/deh95/mi/result/breast_density/' + str(flip_percentage) + 'percent_infs.json'
    print(data)
    with open(infs_path, 'w') as outfile:
        json.dump(data, outfile)
    print('influence saved at ' + infs_path)

if __name__ == "__main__":
    print('start calculating influence from inceptionV3')
    since = time.time()
    flip_ratio = float(sys.argv[1])
    flip_percentage = int(flip_ratio * 100)
    calc_inf_inceptionV3(flip_percentage)
    print('done')

    time_elapsed = time.time() - since
    print('Calculation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

