# -*- coding: utf-8 -*-
import os

directory = './images/'
images_folders = os.listdir(directory)
# for i in range(len(images_folders)):
#     images_folders[i] = int(images_folders[i])
# images_folders.sort()
# for i in range(len(images_folders)):
#     images_folders[i] = str(images_folders[i])
# print(images_folders)

for k, fldr in enumerate(images_folders):
    file_names = [os.path.basename(filename) for filename in os.listdir(directory + fldr)]
    for i, flnm in enumerate(file_names):
        os.rename(directory + fldr + '/' + flnm, directory + fldr + '/' + f'image_{k+1}_{i}.jpg')
