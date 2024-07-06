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
from torch.optim.lr_scheduler import StepLR
from models.module import SimpleCNNEncoder,Classifier,MultiModalCNNModel,FDGFeatureGenerator,UNet,UNetFDGFeatureGenerator
from datasets.dataprocess import TaskDataset


def calculate_cosine_similarity(features):
    batch_size = features[0].size(0)
    num_modalities = len(features)
    similarity_matrix = torch.zeros((num_modalities, num_modalities, batch_size))

    for i in range(num_modalities):
        for j in range(num_modalities):
            if i != j:
                for k in range(batch_size):
                    cosine_sim = F.cosine_similarity(features[i][k].view(1, -1), features[j][k].view(1, -1), dim=1)
                    #print(cosine_sim)
                    similarity_matrix[i, j, k] = torch.sigmoid(cosine_sim)#限制在0-1
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
                            target_similarity = table_info_similarity.item()
                        else:
                            target_similarity = 0.0
                    else:
                        target_similarity = 0.0
                    loss = torch.abs(similarity_matrix[j, k, i] - target_similarity).mean()
                    total_loss += loss
    
    return total_loss

def train_and_evaluate_multimodal_model(csv_paths, data_dir, NC, modalities, batch_size, num_epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 1

    datasets = [TaskDataset(csv_path, data_dir, NC, modalities) for csv_path in csv_paths]
    datasets = [dataset for dataset in datasets if len(dataset) > 0]  # 过滤掉空数据集

    if not datasets:
        raise ValueError("所有数据集都为空，请检查数据路径和CSV文件。")

    dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1) for dataset in datasets]

    val_results = []
    print('loaded dataset')
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"checkpoint/{current_time}_modalities_{modalities}_batch_2"
    os.makedirs(save_dir, exist_ok=True)
    
    
    for fold in range(len(datasets)):
        train_datasets = [TaskDataset(csv_path, data_dir, NC, modalities, is_train=(i != fold)) for i, csv_path in enumerate(csv_paths)]
        #train_datasets = [datasets[i] for i in range(len(datasets)) if i != fold]
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
        val_loader = dataloaders[fold]

        input_channels = 1  # 假设输入通道数为1
        output_channels = 16  # 可以根据需要调整输出通道数
        model = MultiModalCNNModel(input_channels, output_channels, len(modalities), num_classes).to(device)
        print('initialized model')
        print(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)


        best_val_auc = 0.0
        t1_best_model_wts = None

        
        log_file = open(os.path.join(save_dir, f"training_log_fold{fold+1}.txt"), "w")
        print('del_classifier_batch/lr_e-4-change-per-5-epoch/data_aug')
        log_file.write('del_classifier_batch/lr_e-4-change-per-5-epoch/data_aug')


        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            t1_running_loss = 0.0
            fdg_running_loss = 0.0
            t1_running_accuracy = 0.0
            fdg_running_accuracy = 0.0
            start_time = time.time()
            print(f'-------------start training epoch {epoch+1}/{num_epochs}-----------')
            log_file.write(f'-------------start training epoch {epoch+1}/{num_epochs}-----------\n')
            for batch in train_loader:
                fdg_images = batch['fdg_img']
                t1_images = batch['t1_img']
                labels = batch['label'].unsqueeze(1).to(device)

                table_infos = [batch['table_info'].to(device)]

                modalities_data = []
                if t1_images is not None:
                    t1_images = t1_images.to(device)
                    modalities_data.append(t1_images)
                if fdg_images is not None:
                    fdg_images = fdg_images.to(device)
                    modalities_data.append(fdg_images)

                optimizer.zero_grad()

                #t1_encoder = SimpleCNNEncoder(input_channels, output_channels).to(device)
                #fdg_encoder = SimpleCNNEncoder(input_channels, output_channels).to(device)
                t1_features = model.encoders[0](t1_images)
                fdg_features = model.encoders[1](fdg_images)

                

                features = [t1_features, fdg_features]
                #combined_features = torch.cat([feature.view(feature.size(0), -1) for feature in features], dim=1)

                t1_outputs = model.classifier[0](t1_features)
                fdg_outputs = model.classifier[1](fdg_features)

                t1_loss_bce = criterion(t1_outputs, labels)
                fdg_loss_bce = criterion(fdg_outputs, labels)
                loss_cl = cliploss(features, table_infos, labels, weights,device)
                
                loss = t1_loss_bce  + fdg_loss_bce  + 0.1*loss_cl
                #print(f'loss: {loss}, t1_Loss: {t1_loss_bce:.4f},fdg_Loss: {fdg_loss_bce:.4f},Loss_cl: {loss_cl:.4f}')
                #print(f'loss: {loss}, t1_Loss: {t1_loss_bce:.4f},fdg_Loss: {fdg_loss_bce:.4f}', flush=True)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                t1_running_loss += t1_loss_bce.item()
                fdg_running_loss += fdg_loss_bce.item()

                t1_preds = torch.round(torch.sigmoid(t1_outputs))
                fdg_preds = torch.round(torch.sigmoid(fdg_outputs))
                t1_running_accuracy += accuracy_score(labels.cpu().detach().numpy(), t1_preds.cpu().detach().numpy())
                fdg_running_accuracy += accuracy_score(labels.cpu().detach().numpy(), fdg_preds.cpu().detach().numpy())
                
                
                

            epoch_loss = running_loss / len(train_loader)
            t1_train_loss = t1_running_loss / len(train_loader)
            fdg_train_loss = fdg_running_loss / len(train_loader)
            t1_epoch_accuracy = t1_running_accuracy / len(train_loader)
            fdg_epoch_accuracy = fdg_running_accuracy / len(train_loader)
            elapsed_time = time.time() - start_time

            print(f'Learning rate:{learning_rate}')
            log_file.write(f'Learning rate:{learning_rate}')
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f},t1 Loss: {t1_train_loss:.4f},fdg Loss: {fdg_train_loss:.4f}, t1_Accuracy: {t1_epoch_accuracy:.4f},fdg_Accuracy: {fdg_epoch_accuracy:.4f}, Time: {elapsed_time:.2f}s')
            log_file.write(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, t1 Loss: {t1_train_loss:.4f},fdg Loss: {fdg_train_loss:.4f},t1_Accuracy: {t1_epoch_accuracy:.4f},fdg_Accuracy: {fdg_epoch_accuracy:.4f}, Time: {elapsed_time:.2f}s\n')
            scheduler.step()  # 更新学习率

            model.eval()
            val_running_loss = 0.0
            t1_val_running_loss = 0.0
            fdg_val_running_loss = 0.0
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

                    table_infos = [batch['table_info'].to(device)]

                    modalities_data = []
                    if t1_images is not None:
                        t1_images = t1_images.to(device)
                        modalities_data.append(t1_images)
                    if fdg_images is not None:
                        fdg_images = fdg_images.to(device)
                        modalities_data.append(fdg_images)

                    t1_features = model.encoders[0](t1_images)
                    fdg_features = model.encoders[1](fdg_images)

                    #features = [t1_features, fdg_features]
                    #combined_features = torch.cat([feature.view(feature.size(0), -1) for feature in features], dim=1)
                    t1_outputs = model.classifier[0](t1_features)
                    fdg_outputs = model.classifier[1](fdg_features)

                    t1_loss_bce = criterion(t1_outputs, labels)
                    fdg_loss_bce = criterion(fdg_outputs, labels)
                    loss_cl = cliploss(features, table_infos, labels, weights,device)
                    loss = t1_loss_bce + fdg_loss_bce + 0.1*loss_cl 
                    t1_val_running_loss += t1_loss_bce.item()
                    fdg_val_running_loss += fdg_loss_bce.item()
                    val_running_loss += loss.item()

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
            # 保存最佳模型
            if t1_val_auc > best_val_auc:
                best_val_auc = t1_val_auc
                t1_best_model_wts = model.encoders[0].state_dict()
                torch.save(t1_best_model_wts, os.path.join(save_dir, "t1_best_model.pth"))

        log_file.close()
        print("Training completed.")
        val_results.append({
            'fold': fold+1,
            'val_loss': val_loss,
            't1_val_accuracy': t1_val_accuracy,
            't1_val_precision': t1_val_precision,
            't1_val_recall': t1_val_recall,
            't1_val_f1': t1_val_f1,
            't1_val_sensitivity': t1_val_sensitivity,
            't1_val_specificity': t1_val_specificity,
            't1_val_auc': t1_val_auc
        })

    return val_results, t1_best_model_wts

#第二阶段
def train_multimodal_with_generated_fdg(csv_paths, data_dir, NC, modalities, batch_size, num_epochs, learning_rate, best_t1_model_wts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 1

    datasets = [TaskDataset(csv_path, data_dir, NC, modalities) for csv_path in csv_paths]
    datasets = [dataset for dataset in datasets if len(dataset) > 0]  # 过滤掉空数据集

    if not datasets:
        raise ValueError("所有数据集都为空，请检查数据路径和CSV文件。")

    dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1) for dataset in datasets]

    val_results = []

    for fold in range(len(datasets)):
        train_datasets = [datasets[i] for i in range(len(datasets)) if i != fold]
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        val_loader = dataloaders[fold]

        input_channels = 1
        output_channels = 16
        t1_encoder = SimpleCNNEncoder(input_channels, output_channels).to(device)
        t1_encoder.load_state_dict(best_t1_model_wts)

        fdg_generator = FDGFeatureGenerator(2400, 960).to(device)
        classifier = Classifier(input_size=210 * output_channels, num_classes=num_classes).to(device)

        criterion = nn.BCEWithLogitsLoss()
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
            running_accuracy = 0.0
            start_time = time.time()
            print(f'-------------start stage 2 training epoch {epoch+1}/{num_epochs}-----------')
            log_file.write(f'-------------start stage 2 training epoch {epoch+1}/{num_epochs}-----------\n')
            for batch in train_loader:
                t1_images = batch['t1_img'].to(device)
                labels = batch['label'].unsqueeze(1).to(device)

                optimizer.zero_grad()

                t1_features = t1_encoder(t1_images)
                print(t1_features.size())
                fdg_features = fdg_generator(t1_features)

                multimodal_features = torch.cat((t1_features, fdg_features), dim=1)
                outputs = classifier(multimodal_features)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = torch.round(torch.sigmoid(outputs))
                running_accuracy += accuracy_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy())

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = running_accuracy / len(train_loader)
            elapsed_time = time.time() - start_time

            print(f'stage 2 Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Time: {elapsed_time:.2f}s')
            log_file.write(f'stage 2 Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Time: {elapsed_time:.2f}s\n')

            fdg_generator.eval()
            classifier.eval()
            val_running_loss = 0.0
            val_running_accuracy = 0.0
            val_labels = [] 
            val_preds = [] 

            with torch.no_grad():
                for batch in val_loader:
                    t1_images = batch['t1_img'].to(device)
                    labels = batch['label'].unsqueeze(1).to(device)

                    t1_features = t1_encoder(t1_images)
                    fdg_features = fdg_generator(t1_features)

                    multimodal_features = torch.cat((t1_features, fdg_features), dim=1)
                    outputs = classifier(multimodal_features)

                    loss = criterion(outputs, labels)

                    val_running_loss += loss.item()
                    preds = torch.round(torch.sigmoid(outputs))
                    val_running_accuracy += accuracy_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy())

                    val_labels.extend(labels.cpu().detach().numpy())
                    val_preds.extend(preds.cpu().detach().numpy())

            val_loss = val_running_loss / len(val_loader)
            val_accuracy = val_running_accuracy / len(val_loader)
            val_auc = roc_auc_score(val_labels, val_preds)
            val_precision = precision_score(val_labels, val_preds)
            val_recall = recall_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds)
            
            cm = confusion_matrix(val_labels, val_preds)
            tn, fp, fn, tp = cm.ravel()
            val_sensitivity = tp / (tp + fn)
            val_specificity = tn / (tn + fp)

            print(f'Stage 2 Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation AUC: {val_auc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Sensitivity: {val_sensitivity:.4f}, Specificity: {val_specificity:.4f}')
            log_file.write(f'Stage 2 Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation AUC: {val_auc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Sensitivity: {val_sensitivity:.4f}, Specificity: {val_specificity:.4f}\n')

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_wts = (fdg_generator.state_dict(), classifier.state_dict())
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


    val_results,best_t1_model_wts = train_and_evaluate_multimodal_model(csv_paths, data_dir, NC, modalities, batch_size, num_epochs, learning_rate)
    # 训练多模态模型
    val_results_multimodal, best_multimodal_model_wts = train_multimodal_with_generated_fdg(csv_paths, data_dir, NC, modalities, batch_size, num_epochs, learning_rate, best_t1_model_wts)


    # for result in val_results:
    #     print(f"Fold {result['fold']} - Validation Loss: {result['val_loss']:.4f}, Accuracy: {result['val_accuracy']:.4f}, Precision: {result['val_precision']:.4f}, Recall: {result['val_recall']:.4f}, F1-Score: {result['val_f1']:.4f}, Sensitivity: {result['val_sensitivity']:.4f}, Specificity: {result['val_specificity']:.4f}, AUC: {result['val_auc']:.4f}")
