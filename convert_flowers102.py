"""
Oxford Flowers-102 数据集转换脚本
将MATLAB格式的数据转换为LIFT可用的格式
直接生成train.txt和test.txt文件
"""

import os
import scipy.io
import numpy as np

# 配置路径
DATA_DIR = r"D:\1B.毕业设计\数据集"
OUTPUT_DIR = r"D:\1B.毕业设计\数据集\oxford_flowers"
TXT_DIR = r"D:\1B.毕业设计\Code - 副本\LIFT-main"

# 读取MATLAB文件
print("正在读取标签文件...")
labels_mat = scipy.io.loadmat(os.path.join(DATA_DIR, "imagelabels.mat"))
setid_mat = scipy.io.loadmat(os.path.join(DATA_DIR, "setid.mat"))

# 获取标签和划分信息
labels = labels_mat['labels'][0]  # 8189个标签，从1到102
train_ids = setid_mat['trnid'][0]  # 训练集索引
val_ids = setid_mat['valid'][0]    # 验证集索引  
test_ids = setid_mat['tstid'][0]   # 测试集索引

print(f"总图像数: {len(labels)}")
print(f"训练集: {len(train_ids)} 张")
print(f"验证集: {len(val_ids)} 张")
print(f"测试集: {len(test_ids)} 张")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
splits = ['train', 'val', 'test']
for split in splits:
    for class_id in range(1, 103):  # 1-102
        os.makedirs(os.path.join(OUTPUT_DIR, split, f"class_{class_id:03d}"), exist_ok=True)

print("\n正在转换数据...")

# 图像文件基础路径
jpg_dir = os.path.join(DATA_DIR, "jpg")

# 创建txt文件
train_txt_path = os.path.join(TXT_DIR, "oxford_flowers_train.txt")
val_txt_path = os.path.join(TXT_DIR, "oxford_flowers_val.txt")
test_txt_path = os.path.join(TXT_DIR, "oxford_flowers_test.txt")

train_txt = open(train_txt_path, 'w')
val_txt = open(val_txt_path, 'w')
test_txt = open(test_txt_path, 'w')

# 处理训练集
print("处理训练集...")
for idx in train_ids:
    img_idx = idx - 1  # MATLAB索引从1开始
    label = labels[img_idx] - 1  # 转换为0-101索引
    
    src = os.path.join(jpg_dir, f"image_{img_idx+1:05d}.jpg")
    dst = os.path.join(OUTPUT_DIR, "train", f"class_{label+1:03d}", f"image_{img_idx+1:05d}.jpg")
    
    if os.path.exists(src):
        import shutil
        shutil.copy2(src, dst)
        # 写入txt文件：相对路径 标签(0-101)
        rel_path = os.path.join("oxford_flowers", "train", f"class_{label+1:03d}", f"image_{img_idx+1:05d}.jpg")
        train_txt.write(f"{rel_path} {label}\n")

train_txt.close()
print(f"训练集txt已保存: {train_txt_path}")

# 处理验证集
print("处理验证集...")
for idx in val_ids:
    img_idx = idx - 1
    label = labels[img_idx] - 1
    
    src = os.path.join(jpg_dir, f"image_{img_idx+1:05d}.jpg")
    dst = os.path.join(OUTPUT_DIR, "val", f"class_{label+1:03d}", f"image_{img_idx+1:05d}.jpg")
    
    if os.path.exists(src):
        import shutil
        shutil.copy2(src, dst)
        rel_path = os.path.join("oxford_flowers", "val", f"class_{label+1:03d}", f"image_{img_idx+1:05d}.jpg")
        val_txt.write(f"{rel_path} {label}\n")

val_txt.close()
print(f"验证集txt已保存: {val_txt_path}")

# 处理测试集
print("处理测试集...")
for idx in test_ids:
    img_idx = idx - 1
    label = labels[img_idx] - 1
    
    src = os.path.join(jpg_dir, f"image_{img_idx+1:05d}.jpg")
    dst = os.path.join(OUTPUT_DIR, "test", f"class_{label+1:03d}", f"image_{img_idx+1:05d}.jpg")
    
    if os.path.exists(src):
        import shutil
        shutil.copy2(src, dst)
        rel_path = os.path.join("oxford_flowers", "test", f"class_{label+1:03d}", f"image_{img_idx+1:05d}.jpg")
        test_txt.write(f"{rel_path} {label}\n")

test_txt.close()
print(f"测试集txt已保存: {test_txt_path}")

# 统计
print("\n数据集统计:")
for split in splits:
    split_dir = os.path.join(OUTPUT_DIR, split)
    total = 0
    for class_id in range(1, 103):
        class_dir = os.path.join(split_dir, f"class_{class_id:03d}")
        count = len(os.listdir(class_dir)) if os.path.exists(class_dir) else 0
        total += count
    print(f"  {split}: {total} 张图像")

print(f"\n数据转换完成！")
print(f"图像目录: {OUTPUT_DIR}")
print(f"训练txt: {train_txt_path}")
print(f"验证txt: {val_txt_path}")
print(f"测试txt: {test_txt_path}")
