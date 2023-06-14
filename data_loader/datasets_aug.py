import os
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data

from data_loader.utils import *

class datasets(data.Dataset):
    def __init__(self, config, is_train):
        super(datasets, self).__init__()
        self.config = config
        self.is_train = is_train
        self.h = config.height
        self.w = config.width
        self.norm_val = config.norm_val
        self.max_sig = config.max_sig

        if is_train:
            self.c_folder_path_list, self.c_file_path_list, _ = load_file_list(config.c_path, config.input_path, is_flatten = True)
            self.gt_folder_path_list, self.gt_file_path_list, _ = load_file_list(config.c_path, config.gt_path, is_flatten = True)
            self.is_augment = True
        else:
            self.c_folder_path_list, self.c_file_path_list, _ = load_file_list(config.VAL.c_path, config.VAL.input_path, is_flatten = True)
            self.gt_folder_path_list, self.gt_file_path_list, _ = load_file_list(config.VAL.c_path, config.VAL.gt_path, is_flatten = True)

            self.is_augment = False

        self.len = int(np.ceil(len(self.c_file_path_list)))


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = index % self.len

        c_frame = read_frame(self.c_file_path_list[index], self.norm_val)
        gt_frame = read_frame(self.gt_file_path_list[index], self.norm_val)

        if self.is_augment:
            # Noise
            if random.uniform(0, 1) <= 0.05:
            # if random.uniform(0, 1) >= 0.0:
                row,col,ch = c_frame[0].shape
                mean = 0.0
                sigma = random.uniform(0.001, self.max_sig)
                gauss = np.random.normal(mean,sigma,(row,col,ch))
                gauss = gauss.reshape(row,col,ch)

                c_frame = np.expand_dims(np.clip(c_frame[0] + gauss, 0.0, 1.0), axis = 0)

            # Grayscale
            if random.uniform(0, 1) <= 0.3:
                c_frame = np.expand_dims(color_to_gray(c_frame[0]), axis = 0)
                gt_frame = np.expand_dims(color_to_gray(gt_frame[0]), axis = 0)

            # Scaling
            if random.uniform(0, 1) <= 0.5:
                scale = random.uniform(0.7, 1.0)
                row,col,ch = c_frame[0].shape

                c_frame = np.expand_dims(cv2.resize(c_frame[0], dsize=(int(col*scale), int(row*scale)), interpolation=cv2.INTER_AREA), axis = 0)
                gt_frame = np.expand_dims(cv2.resize(gt_frame[0], dsize=(int(col*scale), int(row*scale)), interpolation=cv2.INTER_AREA), axis = 0)

        cropped_frames = np.concatenate([c_frame, gt_frame], axis=3)

        if self.is_train:
            cropped_frames = crop_multi(cropped_frames, self.h, self.w, is_random = True)
        else:
            cropped_frames = cropped_frames

        c_patches = cropped_frames[:, :, :, :3]
        shape = c_patches.shape
        h = shape[1]
        w = shape[2]
        c_patches = c_patches.reshape((h, w, -1, 3))
        c_patches = torch.FloatTensor(np.transpose(c_patches, (2, 3, 0, 1)))

        gt_patches = cropped_frames[:, :, :, 3:6]
        gt_patches = gt_patches.reshape((h, w, -1, 3))
        gt_patches = torch.FloatTensor(np.transpose(gt_patches, (2, 3, 0, 1)))


        return { 'c': c_patches[0], 'gt': gt_patches[0]}


