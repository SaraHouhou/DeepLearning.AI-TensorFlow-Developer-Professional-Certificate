# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:30:31 2022

@author: shouhou
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2



#Now the images are stored within the  directory. There is a subdirectory for each gestures class.

source_path = 'Datasets/gesteImages'

source_path_call = os.path.join(source_path, 'call')
source_path_dislike = os.path.join(source_path, 'dislike')
source_path_menu = os.path.join(source_path, 'menu')
source_path_like = os.path.join(source_path, 'like')
source_path_four = os.path.join(source_path, 'four')
source_path_ok = os.path.join(source_path, 'ok')
source_path_one = os.path.join(source_path, 'one')
source_path_palm = os.path.join(source_path, 'palm')
source_path_peace = os.path.join(source_path, 'peace')
source_path_peace_inverted = os.path.join(source_path, 'peace_inverted')
source_path_peace_rock = os.path.join(source_path, 'rock')
source_path_stop = os.path.join(source_path, 'stop')
source_path_stop_inverted = os.path.join(source_path, 'stop_inverted')
source_path_three = os.path.join(source_path, 'three')
source_path_two_up = os.path.josource_path_peace_stop = os.path.join(source_path, 'two_up')
source_path_two_up_inverted = os.path.join(source_path, 'two_up_inverted')
# Deletes all non-image files (there are two .db files bundled into the dataset)
#!find /Datasets/gesteImages -type f ! -name "*.jpg" -exec rm {} +

# os.listdir returns a list containing all files under the given path
print(f"There are {len(os.listdir(source_path_call))} images of call.")
print(f"There are {len(os.listdir(source_path_dislike))} images of dislike.")
print(f"There are {len(os.listdir(source_path_menu))} images of menu.")
print(f"There are {len(os.listdir(source_path_like))} images of like.")
print(f"There are {len(os.listdir(source_path_four))} images of four.")
print(f"There are {len(os.listdir(source_path_ok))} images of ok.")
print(f"There are {len(os.listdir(source_path_one))} images of one.")
print(f"There are {len(os.listdir(source_path_palm))} images of palm.")
print(f"There are {len(os.listdir(source_path_peace))} images of peace.")
print(f"There are {len(os.listdir(source_path_peace_inverted))} images of peace inverted.")
print(f"There are {len(os.listdir(source_path_peace_rock))} images of peace rock.")
print(f"There are {len(os.listdir(source_path_stop))} images of stop.")
print(f"There are {len(os.listdir(source_path_stop_inverted))} images of stop inverted.")
print(f"There are {len(os.listdir(source_path_three))} images of three.")
print(f"There are {len(os.listdir(source_path_two_up))} images of two up.")
print(f"There are {len(os.listdir(source_path_two_up_inverted))} images of two up inverted.")

