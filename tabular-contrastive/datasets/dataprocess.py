import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from monai.transforms import Compose, ScaleIntensity, CenterSpatialCrop, RandRotate, RandGaussianNoise, RandSpatialCrop


class TaskDataset(Dataset):
    def __init__(self, csv_path, data_dir, NC, modalities, is_train=True, num_augmented_samples=2):
        self.data = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.NC = NC
        self.modalities = modalities
        self.is_train = is_train
        self.num_augmented_samples = num_augmented_samples  # 每个样本生成的增强样本数量
        self.trans = Compose([ScaleIntensity(), CenterSpatialCrop([160, 192, 160])])
        self.trans_aug = Compose([
            ScaleIntensity(),
            RandSpatialCrop([160, 192, 160], random_size=False),
            RandGaussianNoise(prob=0.05)  # Add Gaussian noise with 50% probability
        ])
        
        print(f"Initial number of samples: {len(self.data)}")
        
        self.data = self.data[self.data.apply(lambda row: all(os.path.exists(os.path.join(self.data_dir, row['sub_ID'], 'ses-M00', modality, f"{row['sub_ID']}_ses-M00_smoothed.nii.gz")) for modality in ['fdg']), axis=1)]
        
        print(f"Filtered number of samples: {len(self.data)}")

        # 扩展数据集以包含增强样本（仅在训练集上）
        self.augmented_data = []
        for idx in range(len(self.data)):
            self.augmented_data.append((idx, False))  # 原始样本
            if self.is_train:
                for _ in range(self.num_augmented_samples):
                    self.augmented_data.append((idx, True))  # 增强样本

        print(f"Final number of samples: {len(self.augmented_data)}")

    def __len__(self):
        return len(self.augmented_data)

    def __getitem__(self, idx):
        original_idx, is_augmented = self.augmented_data[idx]

        filename = self.data.iloc[original_idx]['sub_ID']
        row = self.data.iloc[original_idx]

        data = {}
        
        for modality in self.modalities:
            if modality == 't1':
                modality_filepath = os.path.join(self.data_dir, row['sub_ID'], 'ses-M00', modality, f'{filename}_ses-M00.nii.gz')
                modality_data = nib.load(modality_filepath).get_fdata()
                modality_data = torch.from_numpy(np.expand_dims(modality_data, axis=0)).type(torch.float32)
                
                if is_augmented:
                    modality_img = self.trans_aug(modality_data)
                else:
                    modality_img = self.trans(modality_data)

                data[f'{modality}_img'] = modality_img

            elif modality == 'fdg':
                modality_filepath = os.path.join(self.data_dir, row['sub_ID'], 'ses-M00', modality, f'{filename}_ses-M00_smoothed.nii.gz')
                modality_data = nib.load(modality_filepath).get_fdata()
                modality_data = torch.from_numpy(np.expand_dims(modality_data, axis=0)).type(torch.float32)
                
                if is_augmented:
                    modality_img = self.trans_aug(modality_data)
                else:
                    modality_img = self.trans(modality_data)

                data[f'{modality}_img'] = modality_img

        label = float(row[self.NC])
        data['label'] = label

        # 提取表格信息（假设从第十九列开始）
        table_info = row.iloc[18:].values.astype(np.float32)
        table_info = torch.tensor(table_info)
        data['table_info'] = table_info

        return data
   