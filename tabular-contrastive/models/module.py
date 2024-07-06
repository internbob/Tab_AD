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

# 五层CNN编码器
# Define SimpleCNNEncoder with Batch Normalization
class SimpleCNNEncoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SimpleCNNEncoder, self).__init__()
        self.layer1 = nn.Conv3d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.layer2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.layer3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.layer4 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.layer5 = nn.Conv3d(128, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm3d(output_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.pool(self.layer1(x))))
        x = self.relu(self.bn2(self.pool(self.layer2(x))))
        x = self.relu(self.bn3(self.pool(self.layer3(x))))
        x = self.relu(self.bn4(self.pool(self.layer4(x))))
        x = self.relu(self.bn5(self.pool(self.layer5(x))))
        return x

# Define Classifier with Batch Normalization
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义多模态模型
class MultiModalCNNModel(nn.Module):
    def __init__(self, input_channels, output_channels, num_modalities, num_classes):
        super(MultiModalCNNModel, self).__init__()
        self.encoders = nn.ModuleList([SimpleCNNEncoder(input_channels, output_channels) for _ in range(num_modalities)])
        self.classifier = nn.ModuleList([Classifier(input_size=75*output_channels*num_modalities, num_classes=num_classes)for _ in range(num_modalities)])  # 修改输入尺寸

    def forward(self, modalities):
        features = []
        for encoder, modality in zip(self.encoders, modalities):
            encoded = encoder(modality)
            features.append(encoded)
        combined_features = torch.cat([feature.view(feature.size(0), -1) for feature in features], dim=1)
        output = self.classifier(combined_features)
        return output  # 返回分类结果和特征

class FDGFeatureGenerator(nn.Module):
    def __init__(self, input_feature_size, output_feature_size):
        super(FDGFeatureGenerator, self).__init__()
        self.fc = nn.Linear(input_feature_size, output_feature_size)
    
    def forward(self, x):
        #print("输入张量的原始形状:", x.shape)
        
        # 展平张量
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        #print("展平后的张量形状:", x.shape)
        return self.fc(x)

class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(input_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)
        self.final_layer = nn.Conv2d(64, output_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block
    
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        b = self.bottleneck(e4)
        d4 = self.decoder4(torch.cat((b, e4), dim=1))
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        out = self.final_layer(d1)
        return out

# 使用 UNet 作为 FDG 特征生成器
class UNetFDGFeatureGenerator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNetFDGFeatureGenerator, self).__init__()
        self.unet = UNet(input_channels, output_channels)
    
    def forward(self, x):
        return self.unet(x)