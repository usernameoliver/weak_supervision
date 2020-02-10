import os
import random
import pickle


def main(path1, path2, flip_ratio):
    class_1 = os.listdir(path1)
    class_2 = os.listdir(path2)
    label_for_class_1 = [0] * len(class_1)
    label_for_class_2 = [1] * len(class_2)
    # randint(0, len(a) - 1)
    #
    random.shuffle(class_1)
    random.shuffle(class_2)
    class_1_flipped_number = int(flip_ratio * len(class_1))
    class_2_flipped_number = int(flip_ratio * len(class_2))
    label_for_class_1[:class_1_flipped_number] = [1] * class_1_flipped_number
    label_for_class_2[:class_2_flipped_number] = [0] * class_2_flipped_number
    x = dict(zip(class_1, label_for_class_1))
    y = dict(zip(class_2, label_for_class_2))
    z = {**x, **y}
    f = open('./flipped_label.pkl', "wb")
    pickle.dump(z, f)


if __name__ == '__main__':
    main('/pylon5/ca5phjp/deh95/mi/training/data_breast_density/train/B', '/pylon5/ca5phjp/deh95/mi/training/data_breast_density/train/C', 0.2)
