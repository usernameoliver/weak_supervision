'''
Calculating the percentage of mislabels identified.

@Author: Degan Hao
@Date: Feb 05, 2020
'''
import numpy as np
import json
from sklearn import metrics
import matplotlib.pyplot as plt
import sys

def identify_by_nth(infs_path, nth):
    zero_as_zero = []
    zero_as_one = []
    one_as_one = []
    one_as_zero = []
    print('loading influence from ' + infs_path)
    with open(infs_path, 'r') as infile:
        data = json.load(infile)

        for item in data:
            print(item['original_label']) 
            if item['original_label'] == 0:
                if item['flipped_label'] == 0:
                    zero_as_zero.append(item['influence'][nth])
                else:
                    zero_as_one.append(item['influence'][nth])
            else:
                if item['flipped_label'] == 0:
                    one_as_zero.append(item['influence'][nth])
                else:
                    one_as_one.append(item['influence'][nth])
    return zero_as_zero, zero_as_one, one_as_one, one_as_zero

def sort_by_infs(data):
    data_sorted = sorted(data, key = lambda i : i['influence'][0])
    return data_sorted


def count_original_flip_difference(data_mislabeled):
    num_identified = 0
    for item in data_mislabeled:
        if not item['original_label'] == item['flipped_label']:
            num_identified += 1
    return num_identified

if __name__ == "__main__":
    print('start identifying mislabeled data from using influence')
    flip_ratio = float(sys.argv[1])
    flip_percentage = str(int(flip_ratio * 100)) 
    infs_path = '/pylon5/ca5phjp/deh95/mi/result/breast_density/' + flip_percentage + 'percent_infs.json'
    nth = 0
    data = json.load(open(infs_path, 'r'))
    data_sorted = sort_by_infs(data)
    x_percent = 0.02
    num_at_x_percent = int(len(data) * x_percent)
    data_mislabeled = data_sorted[-num_at_x_percent:-1] 
    num_identified = count_original_flip_difference(data_mislabeled)
    print('The percentage of identified mislabeled data = ' + str(num_identified * 100.0 / len(data_mislabeled)) + '%')
    print('done')

