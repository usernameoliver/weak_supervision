'''
Given a training set with noisy data, extract the noise free data and leave the noisy data
Implementation of paper "Novel mislabeled training data detection algorithm"
doi:10.1007%2Fs00521-016-2589-9

@Code Author: Degan Hao
@Date: FEB 04, 2020


'''
import numpy as np
import os
import random
import pickle
from train_ad import train_model_without_subset, test_model_with_subset 
 

def create_flipped_label(data_dir, flip_ratio):
    path1 = data_dir + 'A/'
    path2 = data_dir + 'D/'
    class_1 = os.listdir(path1)
    class_2 = os.listdir(path2)
    label_for_class_1 = [0] * len(class_1)
    label_for_class_2 = [1] * len(class_2)
    random.shuffle(class_1)
    random.shuffle(class_2)
    class_1_flipped_number = int(flip_ratio * len(class_1))
    class_2_flipped_number = int(flip_ratio * len(class_2))
    label_for_class_1[:class_1_flipped_number] = [1] * class_1_flipped_number
    label_for_class_2[:class_2_flipped_number] = [0] * class_2_flipped_number
    x = dict(zip(class_1, label_for_class_1))
    y = dict(zip(class_2, label_for_class_2))
    z = {**x, **y}
    outfile = './flipped_label_partition.pkl'
    f = open(outfile, "wb")
    pickle.dump(z, f)
    return z 

'''
Given a list of image names, group them into n subsets using hash set
Return a list of n hashsets
'''
def partition_training_set(n, offset, training_list):
    list_size = len(training_list)
    sublist_size = list_size / n
    subsets = [set() for _ in range(n)]
    for i in range(list_size):
        subsets[i % n].add(training_list[(i + offset) % list_size])
    return subsets
        
def identify_noisy_data(data_dir, num_partition, num_subset):
    data_dir_train = data_dir + 'train/'
    data_dir_val = data_dir + 'val/'
    t = num_partition#number of times of subsets partitioning
    n = num_subset#number of disjoint almost equally sized subsets
    flip_ratio = 0.2 #percentage of flips
    model_names = ['resnext', 'googlenet', 'resnet18']
    y = len(model_names)#number of classifiers
    models = []

    mislabel_set_by_partitions = [set() for _ in range(t)]
    training_list_A = ['A' + str(e) + '.jpg' for e in range(51, 400, 1)]
    training_list_D = ['D' + str(e) + '.jpg' for e in range(51, 400, 1)]
    training_list = training_list_A + training_list_D  
    flipped_label_dict = create_flipped_label(data_dir_train, flip_ratio)
    flipped_label_dict_val = create_flipped_label(data_dir_val, flip_ratio)
    flipped_label_dict.update(flipped_label_dict_val)
    print('start training models')

    #for each round of partition, we cut the training_list into n subsets
    for p in range(t):
        #Todo the partition algorithm is currently same for all cuts
        subsets = partition_training_set(n, p, training_list)
        #image_name -> label(0,1) is created in flipped_label_dict
        #Loop through n subsets, each subset serve as a testing set and the rest serves as training set
        for i in range(n):
            #Train y classifiers using training dataset Et
            for j in range(y):
                print('-----' + model_names[j] +  '| subset ' + str(i) + '/' + str(n) + '| partition ' + str(p) + '/' + str(t) + '-----')
                model = train_model_without_subset(model_names[j], data_dir, tuple(subsets[i]), flipped_label_dict)
                models.append(model)
            
            #Test y classifiers on testing dataset Epi
            #identify_mislabel_on_subset(i)
            mislabels_in_subset = test_model_with_subset(data_dir, tuple(subsets[i]), flipped_label_dict, models)
            print('mislabels in subset = ')
            print(mislabels_in_subset) 
            for mislabel in mislabels_in_subset:
                mislabel_set_by_partitions[p].add(mislabel)
        print(mislabel_set_by_partitions[p])
        print('mislabel_set_partition ' + str(p)) 
        print('-' * 20)
    A = set()
    print(training_list)
    print('training_list')
    for e in training_list:
        ErrorCounter = 0
        for j in range(t):
            if e in mislabel_set_by_partitions[j]:
                ErrorCounter += 1
        if ErrorCounter > 0:
            A.add(e)

    print(A)
    #Evaluation of the mislabels identified
    num_correct_identified = 0
    for e in A:
        flipped_label = flipped_label_dict[e]
        if e.startswith('A'):
            true_label = 0
        else:
            true_label = 1 
        if flipped_label != true_label:
            num_correct_identified += 1
    num_flipped = len(training_list) * flip_ratio
    percent_identified = num_correct_identified / num_flipped
    print(str(percent_identified) + ' of the flipps are identified')

    return percent_identified
data_dir = './'#Todo: Add your data_dir 
num_partitions = [2,3,4,5,6] 
num_subsets = [2,3,4,5,6] 
result = np.zeros((len(num_partitions), len(num_subsets)))
for i in range(len(num_partitions)):
    num_partition = num_partitions[i]
    for j in range(len(num_subsets)):
        num_subset = num_subsets[j]
        result[i][j] = identify_noisy_data(data_dir, num_partition, num_subset)
        print('-----------------' + str(i)  + '_' + str(j) + '_' + str(result[i][j]) +  '----------------------')
print('result saved to result.npy')
print(result)
f = 'result.npy' 
np.save(f, result)
