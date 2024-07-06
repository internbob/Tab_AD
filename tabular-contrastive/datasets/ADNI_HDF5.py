import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from monai.transforms import (Compose, EnsureType, RandRotate, RandZoom, RandFlip, AddChannel, SpatialPad)


def get_data(data_path, task):
    data = []

    if task == 'ADCN':
        target_label = {'AD': 1, 'CN': 0}
    elif task == 'pMCIsMCI':
        target_label = {'pMCI': 1, 'sMCI': 0}
    else:
        target_label = {'AD': 2, 'pMCI': 1, 'sMCI': 1, 'CN': 0}

    with h5py.File(data_path, "r") as hf:
        for image_uid, g in hf.items():
            # get sub label
            DX = g.attrs['DX']
            # skip subjects
            if DX not in target_label.keys():
                continue
            # get sub image data and tabular data
            MRI = g['MRI'][:]
            FDG = g['FDG'][:]
            # get tabular data
            tabular = g['tabular'][:]
            # transform DX to num
            DX = target_label[DX]
            # append data
            data.append(tuple([MRI, FDG, tabular, DX]))
    return data


class ADNI(Dataset):
    def __init__(self, data_dict, data_transform):
        self.img_transform = data_transform
        self.data = data_dict

    def __len__(self):
        return len(self.data)

    def class_count(self):
        class_0 = 0
        class_1 = 0
        sample_weight = []
        for _, _, _, DX in self.data:
            if DX == 0:
                class_0 += 1
            elif DX == 1:
                class_1 += 1
        class_weight = [1000 / class_0, 1000 / class_1]
        for _, _, _, DX in self.data:
            sample_weight.append(class_weight[DX])
        return torch.Tensor(sample_weight)

    def __getitem__(self, index: int):
        MRI, FDG, tabular, DX = self.data[index]
        # transform image data
        MRI, FDG = self.img_transform(MRI), self.img_transform(FDG)
        data_point = [MRI, FDG, tabular, DX]
        return tuple(data_point)


def ADNI_transform(aug):
    if aug:
        train_transform = Compose([AddChannel(), SpatialPad(spatial_size=[128, -1, 128]),
                                   RandFlip(prob=0.3, spatial_axis=0), RandRotate(prob=0.3, range_x=0.05),
                                   RandZoom(prob=0.3, min_zoom=0.95, max_zoom=1), EnsureType()])
    else:
        train_transform = Compose([AddChannel(), SpatialPad(spatial_size=[128, -1, 128]), EnsureType()])
    test_transform = Compose([AddChannel(), SpatialPad(spatial_size=[128, -1, 128]), EnsureType()])
    return train_transform, test_transform

#
# test_transform = Compose([
#             ScaleIntensity(),
#             AddChannel(),
#             EnsureType()
#         ])
# ADNI_data = get_data('/home/kateridge/Projects/Projects/datasets/ADNI/ADNI_3class.csv', 'ADCN')
# ADNI_dataset = DatasetHeterogeneous(ADNI_data, test_transform)
# print(ADNI_dataset.__getitem__(0))
# a = DatasetHeterogeneous('D:\\datasets\\ADNI_ALL\\ADNI.hdf5', test_transform, 'ADCN')
# print(len(a))
