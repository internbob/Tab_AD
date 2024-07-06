import os
import csv
import numpy as np

# 文件夹路径
folder_path = r"D:\Datasets\ADNI"
csv_file_path = os.path.join(folder_path, "patient_statistics.csv")

# 读取patient_statistics.csv文件，获取所有患者信息
patients_info = {}
with open(csv_file_path, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        patient_id = row["sub_ID"]
        patients_info[patient_id] = row

# 读取所有患者名字
patient_names = list(patients_info.keys())

# 随机打乱患者名字列表
np.random.shuffle(patient_names)

# 创建交叉验证文件夹
for i in range(5):
    cross_folder = os.path.join(folder_path, "lookupcsvs", f"cross{i}")
    os.makedirs(cross_folder, exist_ok=True)
    for csv_name in ["train.csv", "valid.csv", "test.csv"]:
        csv_path = os.path.join(cross_folder, csv_name)
        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=patients_info[patient_names[0]].keys())
            writer.writeheader()

# 分成五折并将患者分配到不同的折中
for i in range(5):
    train_patients = patient_names[:3 * len(patient_names) // 5]
    valid_patients = patient_names[3 * len(patient_names) // 5:4 * len(patient_names) // 5]
    test_patients = patient_names[4 * len(patient_names) // 5:]

    for j, patients in enumerate([train_patients, valid_patients, test_patients]):
        cross_folder = os.path.join(folder_path, "lookupcsvs", f"cross{i}")
        csv_name = ["train.csv", "valid.csv", "test.csv"][j]
        csv_path = os.path.join(cross_folder, csv_name)
        
        with open(csv_path, 'a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=patients_info[patient_names[0]].keys())
            for patient_id in patients:
                writer.writerow(patients_info[patient_id])

print("已生成五折交叉验证所需的文件夹结构和CSV文件。")