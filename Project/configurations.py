# Importing necessary libraries
import os
import operator
import glob
import numpy as np
import cv2
import sys
from imutils import paths
from Enhance_FP.image_enhance import image_enhance
from sklearn.metrics import accuracy_score, precision_score, f1_score
from finger_utils import *
from finger_preprocessing import *
from plots import *
# pyeer library: pip install pyeer
from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report, export_error_rates
from pyeer.plot import plot_eer_stats
from collections import defaultdict

MAX_FEATURES = 500
TRAIN_PER_CLASS = 6
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.25
DIST_THRESHOLD = 30

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RAW_DATASET_DIR = os.path.join(BASE_DIR, 'RawDataset')
ENHANCED_DIR = os.path.join(BASE_DIR, 'enhanced')
SPLIT_DIR = os.path.join(BASE_DIR, "split")

if not os.path.exists(RAW_DATASET_DIR):
    os.mkdir(BASE_DIR)
if not os.path.exists(ENHANCED_DIR):
    os.mkdir(ENHANCED_DIR)
if not os.path.exists(SPLIT_DIR):
    os.mkdir(SPLIT_DIR)


def get_image_label(filename):
    image = filename.split(os.path.sep)
    return image[len(image)-1]


def get_image_class(filename):
    return get_image_label(filename).split('_')[0]

# Splits the dataset on training and testing set
def split_dataset(data, test_size):
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    return train, test

def prepare_split(file_names):
    '''
    Coversion to grayscale and enhancement. Split into training and test set.
    :param file_names: All fingerprint images as file names
    :return: train_set, test_set: 2 dictionaries for training and test,
             where the key is the name of the image and the value is the image itself
    '''
    train_set = {}
    test_set = {}
    data = []  # list of tuples
    temp_label = get_image_class(file_names[0])  # sets the image class (101)
    for filename in file_names:
        img = cv2.imread(filename)
        label = get_image_label(filename)
        print('Processing image {} ...  '.format(label))
        if temp_label != get_image_class(filename):
            train, test = split_dataset(data, 0.2)
            train_set.update(train)
            test_set.update(test)
            temp_label = get_image_class(filename)
            data = []
        data.append((label, img))

        if filename == file_names[len(file_names) - 1]:
            train, test = split_dataset(data, 0.2)
            train_set.update(train)
            test_set.update(test)

    print('DONE')
    return train_set, test_set

