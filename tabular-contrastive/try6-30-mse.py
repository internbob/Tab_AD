import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from monai.transforms import Compose, ScaleIntensity, CenterSpatialCrop, RandRotate, RandGaussianNoise, RandSpatialCrop
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
import copy


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class FixedOrderSampler(SequentialSampler):
    def __init__(self, data_source, indices=None):
        super(FixedOrderSampler, self).__init__(data_source)
        self.indices = indices or list(range(len(data_source)))
        
    def __iter__(self):
        return iter(self.indices)



class TaskDataset(Dataset):
    def __init__(self, csv_path, data_dir, NC, modalities):
        self.data = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.NC = NC
        self.modalities = modalities
        self.trans = Compose([ScaleIntensity(), CenterSpatialCrop([160, 192, 160])])
        
        print(f"Initial number of samples: {len(self.data)}")
        
        # 过滤掉不存在的样本
        self.data = self.data[self.data.apply(
            lambda row: all(os.path.exists(
                os.path.join(self.data_dir, row['sub_ID'], 'ses-M00', modality, f"{row['sub_ID']}_ses-M00_smoothed.nii.gz")
            ) for modality in ['fdg']), axis=1)]
        
        print(f"Filtered number of samples: {len(self.data)}")

        # 重置索引并生成样本ID
        self.data = self.data.reset_index(drop=True)
        self.data['filtered_id'] = self.data['sub_ID']  # 保留原始的 sub_ID

        # 归一化表格信息（假设从第十九列开始）
        self.table_data = self.data.iloc[:, 18:-1].astype(np.float32)
        self.normalized_table_data = (self.table_data - self.table_data.min()) / (self.table_data.max() - self.table_data.min())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.iloc[idx]['sub_ID']
        row = self.data.iloc[idx]

        data = {}
        
        for modality in self.modalities:
            if modality == 't1':
                modality_filepath = os.path.join(self.data_dir, row['sub_ID'], 'ses-M00', modality, f'{filename}_ses-M00.nii.gz')
                modality_data = nib.load(modality_filepath).get_fdata()
                modality_data = torch.from_numpy(np.expand_dims(modality_data, axis=0)).type(torch.float32)
                modality_img = self.trans(modality_data)
                data[f'{modality}_img'] = modality_img
            if modality == 'fdg':
                modality_filepath = os.path.join(self.data_dir, row['sub_ID'], 'ses-M00', modality, f'{filename}_ses-M00_smoothed.nii.gz')
                modality_data = nib.load(modality_filepath).get_fdata()
                modality_data = torch.from_numpy(np.expand_dims(modality_data, axis=0)).type(torch.float32)
                modality_img = self.trans(modality_data)
                data[f'{modality}_img'] = modality_img
            else:
                continue

        label = float(row[self.NC])
        data['label'] = label

        # 提取归一化后的表格信息
        table_info = self.normalized_table_data.iloc[idx].values
        table_info = torch.tensor(table_info, dtype=torch.float32)
        data['table_info'] = table_info

        # 添加样本ID
        data['id'] = row['filtered_id']  # 使用保留的原始 sub_ID

        return data


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
    
def calculate_cosine_similarity(features):
    batch_size = features[0].size(0)
    num_modalities = len(features)
    similarity_matrix = torch.zeros((num_modalities, num_modalities, batch_size))

    for i in range(num_modalities):
        for j in range(num_modalities):
            if i != j:
                for k in range(batch_size):
                    cosine_sim = F.cosine_similarity(features[i][k].view(1, -1), features[j][k].view(1, -1), dim=1)
                    #print(f'cosine_sim:{cosine_sim}')
                    similarity_matrix[i, j, k] = torch.relu(cosine_sim)#限制在0-1
                    #print(f'sim:{similarity_matrix[i, j, k]}')
    return similarity_matrix

def cliploss(features, table_infos, labels, weights,device):
    similarity_matrix = calculate_cosine_similarity(features)
    
    total_loss = 0.0
    batch_size = similarity_matrix.size(2)
    num_modalities = similarity_matrix.size(0)
    
    for i in range(batch_size):
        for j in range(num_modalities):
            for k in range(num_modalities):
                if j != k:
                    if labels[i] == 1:
                        common_indices = torch.nonzero(torch.min(table_infos[0][j] > 0, table_infos[0][k] > 0)).squeeze().to(device)
                        if common_indices.numel() > 0:
                            table_info_similarity = F.cosine_similarity((table_infos[0][j][common_indices] * weights[common_indices]).unsqueeze(0), (table_infos[0][k][common_indices] * weights[common_indices]).unsqueeze(0), dim=1)
                            #print(f'target sim:{table_info_similarity}')
                            target_similarity = table_info_similarity.item()
                        else:
                            target_similarity = 0.0
                    else:
                        target_similarity = 0.0
                    loss = torch.abs(similarity_matrix[j, k, i] - target_similarity).mean()
                    total_loss += loss
    
    return total_loss

def train_first_stage(csv_paths, data_dir, NC, modalities, batch_size, num_epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 1

    datasets = [TaskDataset(csv_path, data_dir, NC, modalities) for csv_path in csv_paths]
    datasets = [dataset for dataset in datasets if len(dataset) > 0]

    if not datasets:
        raise ValueError("所有数据集都为空，请检查数据路径和CSV文件。")

    dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1) for dataset in datasets]

    val_results = []
    best_val_auc = 0.0
    best_epoch = -1
    t1_best_model_wts = None
    fdg_features_dict = {}

    for fold in range(len(datasets)):
        train_datasets = [datasets[i] for i in range(len(datasets)) if i != fold]
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        val_loader = dataloaders[fold]

        input_channels = 1
        output_channels = 16
        model = MultiModalCNNModel(input_channels, output_channels, len(modalities), num_classes).to(device)
        criterion = nn.BCEWithLogitsLoss()
      
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, last_epoch=-1)

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"checkpoint/{current_time}_first_stage"
        os.makedirs(save_dir, exist_ok=True)

        log_file = open(os.path.join(save_dir, f"training_log_fold{fold+1}.txt"), "w")

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            t1_running_loss = 0.0
            fdg_running_loss = 0.0
            mse_running_loss = 0.0
            clip_running_loss = 0.0
            t1_running_accuracy = 0.0
            fdg_running_accuracy = 0.0
            start_time = time.time()
            print(f'-------------start training epoch {epoch+1}/{num_epochs}-----------')
            log_file.write(f'-------------start training epoch {epoch+1}/{num_epochs}-----------\n')
            for batch in train_loader:
                fdg_images = batch['fdg_img']
                t1_images = batch['t1_img']
                labels = batch['label'].unsqueeze(1).to(device)
                sample_ids = batch['id']
                table_infos = [batch['table_info'].to(device)]

                modalities_data = []
                if t1_images is not None:
                    t1_images = t1_images.to(device)
                    modalities_data.append(t1_images)
                if fdg_images is not None:
                    fdg_images = fdg_images.to(device)
                    modalities_data.append(fdg_images)

                optimizer.zero_grad()

                t1_features = model.encoders[0](t1_images)
                fdg_features = model.encoders[1](fdg_images)
                #print(f'fdg_features:{fdg_features.size()}')
                
                # 保存生成的FDG特征
                
                for i, sample_id in enumerate(sample_ids):
                    fdg_features_dict[sample_id] = fdg_features[i].view(-1).clone().detach().cpu().numpy()

                t1_outputs = model.classifier[0](t1_features)
                fdg_outputs = model.classifier[1](fdg_features)

                t1_loss_bce = criterion(t1_outputs, labels)
                fdg_loss_bce = criterion(fdg_outputs, labels)

                loss_cl = cliploss([t1_features, fdg_features], table_infos, labels, weights, device)

                loss = t1_loss_bce + fdg_loss_bce + 0.1 * loss_cl

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                t1_running_loss += t1_loss_bce.item()
                fdg_running_loss += fdg_loss_bce.item()
                
                clip_running_loss += loss_cl.item()

                t1_preds = torch.round(torch.sigmoid(t1_outputs))
                fdg_preds = torch.round(torch.sigmoid(fdg_outputs))
                t1_running_accuracy += accuracy_score(labels.cpu().detach().numpy(), t1_preds.cpu().detach().numpy())
                fdg_running_accuracy += accuracy_score(labels.cpu().detach().numpy(), fdg_preds.cpu().detach().numpy())

            epoch_loss = running_loss / len(train_loader)
            t1_train_loss = t1_running_loss / len(train_loader)
            fdg_train_loss = fdg_running_loss / len(train_loader)
            clip_train_loss = clip_running_loss / len(train_loader)
            t1_epoch_accuracy = t1_running_accuracy / len(train_loader)
            fdg_epoch_accuracy = fdg_running_accuracy / len(train_loader)
            elapsed_time = time.time() - start_time
            print(f'Learning rate:{learning_rate}')
            log_file.write(f'Learning rate:{learning_rate}')
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f},t1 Loss: {t1_train_loss:.4f},clip Loss: {clip_train_loss:.4f},fdg Loss: {fdg_train_loss:.4f}, t1_Accuracy: {t1_epoch_accuracy:.4f},fdg_Accuracy: {fdg_epoch_accuracy:.4f}, Time: {elapsed_time:.2f}s')
            log_file.write(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, t1 Loss: {t1_train_loss:.4f},clip Loss: {clip_train_loss:.4f},fdg Loss: {fdg_train_loss:.4f},t1_Accuracy: {t1_epoch_accuracy:.4f},fdg_Accuracy: {fdg_epoch_accuracy:.4f}, Time: {elapsed_time:.2f}s\n')

            scheduler.step()

            

            model.eval()
            val_running_loss = 0.0
            t1_val_running_loss = 0.0
            fdg_val_running_loss = 0.0
            mse_val_running_loss = 0.0
            clip_val_running_loss = 0.0
            t1_val_running_accuracy = 0.0
            fdg_val_running_accuracy = 0.0
            val_labels = []
            t1_val_preds = []
            fdg_val_preds = []

            with torch.no_grad():
                for batch in val_loader:
                    fdg_images = batch['fdg_img']
                    t1_images = batch['t1_img']
                    labels = batch['label'].unsqueeze(1).to(device)
                    sample_ids = batch['id']
                    table_infos = batch['table_info'].to(device)

                    if t1_images is not None:
                        t1_images = t1_images.to(device)
                    if fdg_images is not None:
                        fdg_images = fdg_images.to(device)

                    t1_features = model.encoders[0](t1_images)
                    fdg_features = model.encoders[1](fdg_images)
                    #print(f'fdg_features:{fdg_features.size()}')

                    t1_outputs = model.classifier[0](t1_features)
                    fdg_outputs = model.classifier[1](fdg_features)

                    t1_loss_bce = criterion(t1_outputs, labels)
                    fdg_loss_bce = criterion(fdg_outputs, labels)

                    # 保存生成的FDG特征
                    # for sample_id in sample_ids:
                    #     fdg_features_dict[sample_id] = fdg_features.clone().detach().cpu().numpy()

                    loss_cl = cliploss([t1_features, fdg_features], table_infos, labels, weights, device)

                    loss = t1_loss_bce + fdg_loss_bce + 0.1 * loss_cl

                    val_running_loss += loss.item()
                    t1_val_running_loss += t1_loss_bce.item()
                    fdg_val_running_loss += fdg_loss_bce.item()
                   
                    clip_val_running_loss += loss_cl.item()

                    t1_preds = torch.round(torch.sigmoid(t1_outputs))
                    fdg_preds = torch.round(torch.sigmoid(fdg_outputs))

                    t1_val_running_accuracy += accuracy_score(labels.cpu().detach().numpy(), t1_preds.cpu().detach().numpy())
                    fdg_val_running_accuracy += accuracy_score(labels.cpu().detach().numpy(), fdg_preds.cpu().detach().numpy())
                    val_labels.extend(labels.cpu().detach().numpy())
                    t1_val_preds.extend(t1_preds.cpu().detach().numpy())
                    fdg_val_preds.extend(fdg_preds.cpu().detach().numpy())

            val_loss = val_running_loss / len(val_loader)
            t1_val_loss = t1_val_running_loss / len(val_loader)
            fdg_val_loss = fdg_val_running_loss / len(val_loader)
            mse_val_loss = mse_val_running_loss / len(val_loader)
            clip_val_loss = clip_val_running_loss / len(val_loader)

            t1_val_accuracy = t1_val_running_accuracy / len(val_loader)
            t1_val_auc = roc_auc_score(val_labels, t1_val_preds)
            t1_val_precision = precision_score(val_labels, t1_val_preds)
            t1_val_recall = recall_score(val_labels, t1_val_preds)
            t1_val_f1 = f1_score(val_labels, t1_val_preds)

            fdg_val_accuracy = fdg_val_running_accuracy / len(val_loader)
            fdg_val_auc = roc_auc_score(val_labels, fdg_val_preds)
            fdg_val_precision = precision_score(val_labels, fdg_val_preds)
            fdg_val_recall = recall_score(val_labels, fdg_val_preds)
            fdg_val_f1 = f1_score(val_labels, fdg_val_preds)

            cm_t1 = confusion_matrix(val_labels, t1_val_preds)
            tn, fp, fn, tp = cm_t1.ravel()
            t1_val_sensitivity = tp / (tp + fn)
            t1_val_specificity = tn / (tn + fp)

            cm_fdg = confusion_matrix(val_labels, fdg_val_preds)
            tn, fp, fn, tp = cm_fdg.ravel()
            fdg_val_sensitivity = tp / (tp + fn)
            fdg_val_specificity = tn / (tn + fp)

            print(f't1 Validation Loss: {t1_val_loss:.4f}, Validation Accuracy: {t1_val_accuracy:.4f}, Validation AUC: {t1_val_auc:.4f}, Precision: {t1_val_precision:.4f}, Recall: {t1_val_recall:.4f}, F1: {t1_val_f1:.4f}, Sensitivity: {t1_val_sensitivity:.4f}, Specificity: {t1_val_specificity:.4f}')
            log_file.write(f't1 Validation Loss: {t1_val_loss:.4f}, t1 Validation Accuracy: {t1_val_accuracy:.4f}, Validation AUC: {t1_val_auc:.4f}, Precision: {t1_val_precision:.4f}, Recall: {t1_val_recall:.4f}, F1: {t1_val_f1:.4f}, Sensitivity: {t1_val_sensitivity:.4f}, Specificity: {t1_val_specificity:.4f}\n')
            print(f'fdg Validation Loss: {fdg_val_loss:.4f}, fdg Validation Accuracy: {fdg_val_accuracy:.4f}, fdg Validation AUC: {fdg_val_auc:.4f}, fdg Precision: {fdg_val_precision:.4f}, fdg Recall: {fdg_val_recall:.4f}, fdg F1: {fdg_val_f1:.4f}, fdg Sensitivity: {fdg_val_sensitivity:.4f}, fdg Specificity: {fdg_val_specificity:.4f}')
            log_file.write(f'fdg Validation Loss: {fdg_val_loss:.4f}, fdg Validation Accuracy: {fdg_val_accuracy:.4f}, fdg Validation AUC: {fdg_val_auc:.4f}, fdg Precision: {fdg_val_precision:.4f}, fdg Recall: {fdg_val_recall:.4f}, fdg F1: {fdg_val_f1:.4f}, fdg Sensitivity: {fdg_val_sensitivity:.4f}, fdg Specificity: {fdg_val_specificity:.4f}\n')
            

            val_results.append([epoch, val_loss, t1_val_loss, mse_val_loss, fdg_val_loss, clip_val_loss, t1_val_accuracy, t1_val_auc, t1_val_precision, t1_val_recall, t1_val_f1, fdg_val_accuracy, fdg_val_auc, fdg_val_precision, fdg_val_recall, fdg_val_f1])

            if t1_val_auc > best_val_auc:
                best_val_auc = t1_val_auc
                best_epoch = epoch
                t1_best_model_wts = model.encoders[0].state_dict()
                torch.save(t1_best_model_wts, os.path.join(save_dir, f"best_model_fold{fold+1}.pt"))

        log_file.close()

    return val_results, best_val_auc, best_epoch, fdg_features_dict,t1_best_model_wts



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

#第二阶段
def train_second_stage(csv_paths, data_dir, NC, modalities, batch_size, num_epochs, learning_rate, best_t1_model_wts,fdg_features_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 1

    datasets = [TaskDataset(csv_path, data_dir, NC, modalities) for csv_path in csv_paths]
    datasets = [dataset for dataset in datasets if len(dataset) > 0]  # 过滤掉空数据集

    if not datasets:
        raise ValueError("所有数据集都为空，请检查数据路径和CSV文件。")

    dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1) for dataset in datasets]

    val_results = []
    best_val_auc = 0.0
    best_epoch = -1
    best_model_wts = None

    for fold in range(len(datasets)):
        train_datasets = [datasets[i] for i in range(len(datasets)) if i != fold]
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        
        train_sampler = FixedOrderSampler(train_dataset)
        val_sampler = FixedOrderSampler(datasets[fold])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=1)
        val_loader = DataLoader(datasets[fold], batch_size=batch_size, sampler=val_sampler, num_workers=1)

        input_channels = 1
        output_channels = 16
        t1_encoder = SimpleCNNEncoder(input_channels, output_channels).to(device)
        t1_encoder.load_state_dict(best_t1_model_wts)

        fdg_generator = FDGFeatureGenerator(2400, 2400).to(device)
        classifier = Classifier(input_size=300 * output_channels, num_classes=num_classes).to(device)

        criterion = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()
        optimizer = optim.Adam(list(fdg_generator.parameters()) + list(classifier.parameters()), lr=learning_rate)

        best_val_auc = 0.0

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"checkpoint/{current_time}_multimodal_batch_{train_loader.batch_size}"
        os.makedirs(save_dir, exist_ok=True)

        log_file = open(os.path.join(save_dir, "training_log.txt"), "w")

        for epoch in range(num_epochs):
            t1_encoder.eval()  # T1 encoder 在第二阶段不再训练
            fdg_generator.train()
            classifier.train()
            running_loss = 0.0
            mse_running_loss = 0.0
            clip_running_loss = 0.0
            running_accuracy = 0.0
            saved_fdg_features = {}
            start_time = time.time()
            print(f'-------------start stage 2 training epoch {epoch+1}/{num_epochs}-----------')
            log_file.write(f'-------------start stage 2 training epoch {epoch+1}/{num_epochs}-----------\n')
            for batch in train_loader:
                t1_images = batch['t1_img'].to(device)
                labels = batch['label'].unsqueeze(1).to(device)
                #print(labels)
                sample_ids = batch['id']
                #print(f'sample_ids:{sample_ids}')
                table_infos = [batch['table_info'].to(device)]

                optimizer.zero_grad()

                t1_features = t1_encoder(t1_images)
                fdg_features = fdg_generator(t1_features)
                #print(f'fdg_features size:{fdg_features.size()}')

                t1_features = t1_features.view(t1_features.size(0), -1)

                multimodal_features = torch.cat((t1_features, fdg_features), dim=1)
                outputs = classifier(multimodal_features)


                # 从字典中读取fdg_features
                saved_fdg_features = torch.stack([torch.tensor(fdg_features_dict[sample_id], device=device) for sample_id in sample_ids])
                print(f'saved_fdg_features size: {saved_fdg_features.size()}')
                
                
                mse_fdg_features = fdg_features.view(fdg_features.size(0), -1)
                #print(f'mse_fdg_features size: {mse_fdg_features.size()}')

                # 逐个样本计算MSE损失
                mse_loss_values = mse_loss(mse_fdg_features, saved_fdg_features) 
                mse_loss_value = torch.mean(mse_loss_values)



                #loss_cl = cliploss([t1_features, fdg_features], table_infos, labels, weights, device)
                
                loss_bce = criterion(outputs, labels)
                loss = mse_loss_value + loss_bce #+ loss_cl



                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                mse_running_loss += mse_loss_value.item()
                #clip_running_loss += loss_cl.item()
                preds = torch.round(torch.sigmoid(outputs))
                running_accuracy += accuracy_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy())

            epoch_loss = running_loss / len(train_loader)
            train_mse_loss = mse_running_loss / len(train_loader)
            #clip_loss = clip_running_loss / len(train_loader)
            epoch_accuracy = running_accuracy / len(train_loader)
            elapsed_time = time.time() - start_time

            print(f'stage 2 Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f},mse Loss: {train_mse_loss:.4f},Accuracy: {epoch_accuracy:.4f}, Time: {elapsed_time:.2f}s')
            log_file.write(f'stage 2 Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f},mse Loss: {mse_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Time: {elapsed_time:.2f}s\n')

            fdg_generator.eval()
            classifier.eval()
            val_running_loss = 0.0
            mse_val_running_loss = 0.0
            clip_val_running_loss = 0.0
            val_running_accuracy = 0.0
            val_labels = []
            val_preds = []

            with torch.no_grad():
                for batch in val_loader:
                    t1_images = batch['t1_img'].to(device)
                    labels = batch['label'].unsqueeze(1).to(device)
                    sample_ids = batch['id']
            

                    t1_features = t1_encoder(t1_images)
                    fdg_features = fdg_generator(t1_features)

                    t1_features = t1_features.view(t1_features.size(0), -1)

                    multimodal_features = torch.cat((t1_features, fdg_features), dim=1)
                    outputs = classifier(multimodal_features)

                    # 从 fdg_features_dict 中读取保存的特征并转换为张量
                    # 从字典中读取fdg_features
                    #saved_fdg_features = torch.stack([torch.tensor(fdg_features_dict[sample_id], device=device) for sample_id in sample_ids])
                    #print(f'saved_fdg_features dize:{saved_fdg_features.size()}')
                        
                    
                    # # 从 fdg_features_dict 中读取保存的特征并转换为张量
                    # saved_fdg_features = [torch.tensor(fdg_features_dict[sample_id], device=device) for sample_id in sample_ids]
                    
                    
                    
                    # if len(saved_fdg_features) > 0:
                    #     saved_fdg_features = torch.stack(saved_fdg_features)
                    #     print(f'saved_fdg_features size:{saved_fdg_features.size()}')
                    # else:
                    #     print("saved_fdg_features is empty.")

                    #mse_fdg_features = fdg_features.view(fdg_features.size(0), -1)
                    #print(f'mse_fdg_features size: {mse_fdg_features.size()}')

                    # 逐个样本计算MSE损失
                    #mse_loss_values = mse_loss(mse_fdg_features, saved_fdg_features)
                    #mse_loss_value = torch.mean(mse_loss_values)
                        
                    
                    #loss_cl = cliploss([t1_features, fdg_features], table_infos, labels, weights, device)
                
                    loss = criterion(outputs, labels)
                    #loss = mse_loss_value +  loss_bce#+ loss_cl

                    

                    val_running_loss += loss.item()
                    #mse_val_running_loss += mse_loss_value.item()
                    #clip_val_running_loss += loss_cl.item()
                    preds = torch.round(torch.sigmoid(outputs))
                    val_running_accuracy += accuracy_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy())

                    val_labels.extend(labels.cpu().detach().numpy())
                    val_preds.extend(preds.cpu().detach().numpy())

            val_loss = val_running_loss / len(val_loader)
            #mse_val_loss = mse_val_running_loss / len(val_loader)
            #clip_val_loss = clip_val_running_loss / len(val_loader)
            val_accuracy = val_running_accuracy / len(val_loader)
            val_auc = roc_auc_score(val_labels, val_preds)
            val_precision = precision_score(val_labels, val_preds)
            val_recall = recall_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds)
            
            cm = confusion_matrix(val_labels, val_preds)
            tn, fp, fn, tp = cm.ravel()
            val_sensitivity = tp / (tp + fn)
            val_specificity = tn / (tn + fp)

            print(f'Stage 2 Validation Loss: {val_loss:.4f},  Validation Accuracy: {val_accuracy:.4f}, Validation AUC: {val_auc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Sensitivity: {val_sensitivity:.4f}, Specificity: {val_specificity:.4f}')
            log_file.write(f'Stage 2 Validation Loss: {val_loss:.4f},  Validation Accuracy: {val_accuracy:.4f}, Validation AUC: {val_auc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Sensitivity: {val_sensitivity:.4f}, Specificity: {val_specificity:.4f}\n')

            if val_auc > best_val_auc:
                best_epoch = epoch
                best_model_wts = fdg_generator.state_dict()
                torch.save(best_model_wts, os.path.join(save_dir, "best_model.pth"))

        log_file.close()
        print("Training completed.")
        val_results.append({
            'fold': fold+1,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'val_sensitivity': val_sensitivity,
            'val_specificity': val_specificity,
            'val_auc': val_auc
        })

    return val_results, best_model_wts




if __name__=='__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "D:\\Datasets\\ADNI\\ADNI_Linear"
    NC = 'NC'
    modalities = ['t1', 'fdg']
    csv_paths = [f'C:/Users/idealab/Desktop/6-29/tabular-contrastive/lookupcsvs/test_fold_{i+1}.csv' for i in range(5)]
    batch_size = 2
    num_epochs = 1
    learning_rate = 1e-4
    feature_names = [ "age", "gender", "apoe", "education", "race", "trailA", "trailB", "boston",
                    "digitB", "digitBL", "digitF", "digitFL", "animal", "gds", "lm_imm", "lm_del", "mmse", 
                    "npiq_DEL", "npiq_HALL", "npiq_AGIT", "npiq_DEPD", "npiq_ANX", "npiq_ELAT", "npiq_APA",
                    "npiq_DISN", "npiq_IRR", "npiq_MOT", "npiq_NITE", "npiq_APP", "faq_BILLS", "faq_TAXES", 
                    "faq_SHOPPING", "faq_GAMES", "faq_STOVE", "faq_MEALPREP", "faq_EVENTS", "faq_PAYATTN",
                    "faq_REMDATES", "faq_TRAVEL", "his_NACCFAM", "his_CVHATT", "his_CVAFIB", "his_CVANGIO", 
                    "his_CVBYPASS", "his_CVPACE", "his_CVCHF", "his_CVOTHR", "his_CBSTROKE", "his_CBTIA", 
                    "his_SEIZURES", "his_TBI", "his_HYPERTEN", "his_HYPERCHO", "his_DIABETES", "his_B12DEF", 
                    "his_THYROID", "his_INCONTU", "his_INCONTF", "his_DEP2YRS", "his_DEPOTHR", "his_PSYCDIS", 
                    "his_ALCOHOL", "his_TOBAC100", "his_SMOKYRS", "his_PACKSPER", "his_ABUSOTHR"]
    weights = torch.tensor([ 0.08, 0.02, 0.10, 0.05, 0.01, 0.04, 0.06, 0.06, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.05, 0.05, 0.12, 
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 
            0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]).to(device)


    val_results, best_val_auc, best_epoch, fdg_features_dict,best_t1_model_wts = train_first_stage(csv_paths, data_dir, NC, modalities, batch_size, num_epochs, learning_rate)
    # 训练多模态模型
    val_results, best_model_wts, best_epoch = train_second_stage(csv_paths, data_dir, NC, modalities, batch_size, num_epochs, learning_rate, best_t1_model_wts,fdg_features_dict)


    # for result in val_results:
    #     print(f"Fold {result['fold']} - Validation Loss: {result['val_loss']:.4f}, Accuracy: {result['val_accuracy']:.4f}, Precision: {result['val_precision']:.4f}, Recall: {result['val_recall']:.4f}, F1-Score: {result['val_f1']:.4f}, Sensitivity: {result['val_sensitivity']:.4f}, Specificity: {result['val_specificity']:.4f}, AUC: {result['val_auc']:.4f}")
