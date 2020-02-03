import os
import random
import pickle
import sys


def main(path1, path2, flip_ratio, train_or_val):
    random.seed(9001)
    class_1 = os.listdir(path1)
    class_2 = os.listdir(path2)
    label_for_class_1 = [0] * len(class_1)
    label_for_class_2 = [1] * len(class_2)

    x = dict(zip(class_1, label_for_class_1))
    y = dict(zip(class_2, label_for_class_2))
    z = {**x, **y}
    f = open('./label_dict/' + str(int(flip_ratio * 100)) + 'percent_original_label_' + train_or_val + '.pkl', "wb")
    pickle.dump(z, f)

    random.shuffle(class_1)
    random.shuffle(class_2)
    class_1_flipped_number = int(flip_ratio * len(class_1))
    class_2_flipped_number = int(flip_ratio * len(class_2))
    label_for_class_1[:class_1_flipped_number] = [1] * class_1_flipped_number
    label_for_class_2[:class_2_flipped_number] = [0] * class_2_flipped_number

    x = dict(zip(class_1, label_for_class_1))
    y = dict(zip(class_2, label_for_class_2))
    z = {**x, **y}
    f = open('./label_dict/' + str(int(flip_ratio * 100)) + 'percent_flipped_label_' + train_or_val + '.pkl', "wb")
    pickle.dump(z, f)


if __name__ == '__main__':
    #main('images/B', 'images/C', 0.2)
    flip_ratio = float(sys.argv[1])

    train_or_val = 'val'
    main('data_breast_density/' + train_or_val + '/B', 'data_breast_density/' + train_or_val + '/C', flip_ratio, train_or_val)
    train_or_val = 'train'
    main('data_breast_density/' + train_or_val + '/B', 'data_breast_density/' + train_or_val + '/C', flip_ratio, train_or_val)
