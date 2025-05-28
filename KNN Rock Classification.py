import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 参数设置
DATA_PATH = "Rock Data/train"  # 数据集路径
IMG_SIZE = (64, 64)          # 统一图像尺寸
TEST_RATIO = 0.2             # 测试集比例
RANDOM_STATE = 42            # 随机种子
CLASS_NAMES = ['Basalt', 'chert', 'Clay', 'Conglomerate', 'Diatomite', 'gypsum', 'olivine-basalt', 'Shale-(Mudstone)', 'Siliceous-sinter']  #现实标签

def extract_features(img_path):
    """提取图像特征（RGB颜色直方图）"""
    img = Image.open(img_path).resize(IMG_SIZE).convert("RGB")
    # 计算RGB三通道的直方图（各通道32个bin）
    hist = np.concatenate([
        np.histogram(np.array(img.getchannel(i)), bins=32, range=(0, 255))[0]
        for i in range(3)
    ])
    return hist / hist.sum()  # 归一化

def load_data(data_path):
    """加载岩石数据集"""
    features, labels = [], []
    sample_images = []  # 存储用于可视化的样本图像
    for class_idx, class_name in enumerate(sorted(os.listdir(data_path))):
        class_dir = os.path.join(data_path, class_name)
        # 每类只取第一个图像用于展示
        img_file = os.listdir(class_dir)[0]
        img_path = os.path.join(class_dir, img_file)
        sample_images.append(Image.open(img_path).resize(IMG_SIZE))
        
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            features.append(extract_features(img_path))
            labels.append(class_idx)
    return np.array(features), np.array(labels), sample_images

def plot_sample_images(images, class_names):
    """展示每类样本图像"""
    plt.figure(figsize=(15, 3))
    for i, (img, name) in enumerate(zip(images, class_names)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img)
        plt.title(name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_feature_distribution(X, y, class_names):
    """展示特征分布"""
    plt.figure(figsize=(10, 6))
    for class_idx in range(len(class_names)):
        # 取每个类别的第一个特征值（直方图的第一个bin）
        class_data = X[y == class_idx][:, 0]  
        sns.kdeplot(class_data, label=class_names[class_idx])
    plt.title('Feature Distribution (First Histogram Bin)')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def main():
    # 1. 加载数据
    print("正在加载数据...")
    X, y, sample_images = load_data(DATA_PATH)
    
    # 2. 可视化样本图像
    plot_sample_images(sample_images, CLASS_NAMES)
    
    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE
    )
    
    # 4. 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 5. 可视化特征分布
    plot_feature_distribution(X_train, y_train, CLASS_NAMES)
    
    # 6. 训练KNN分类器
    print("训练KNN分类器...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # 7. 评估模型
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n模型准确率: {accuracy:.2%}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    # 8. 可视化混淆矩阵
    plot_confusion_matrix(y_test, y_pred, CLASS_NAMES)

if __name__ == "__main__":
    main()
