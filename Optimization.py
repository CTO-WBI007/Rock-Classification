import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.feature import local_binary_pattern

# 参数设置
DATA_PATH = "Rock Data/train"  # 数据集路径
IMG_SIZE = (64, 64)          # 统一图像尺寸
TEST_RATIO = 0.2             # 测试集比例
RANDOM_STATE = 42            # 随机种子
CLASS_NAMES = ['Basalt', 'chert', 'Clay', 'Conglomerate', 'Diatomite', 'gypsum', 'olivine-basalt', 'Shale-(Mudstone)', 'Siliceous-sinter']  #现实标签

# 数据增强参数
AUGMENTATION_PARAMS = {
    'rotation_range': 30,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

def extract_lbp_features(img):
    """提取LBP纹理特征"""
    gray_img = np.array(img.convert('L'))
    lbp = local_binary_pattern(gray_img, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp, bins=32, range=(0, 32))
    return hist / hist.sum()

def extract_combined_features(img_path):
    """组合颜色直方图和纹理特征"""
    img = Image.open(img_path).resize(IMG_SIZE).convert("RGB")
    
    # 颜色特征
    color_hist = np.concatenate([
        np.histogram(np.array(img.getchannel(i)), bins=32, range=(0, 255))[0]
        for i in range(3)
    ])
    
    # 纹理特征
    texture_feat = extract_lbp_features(img)
    
    return np.concatenate([color_hist, texture_feat])

def load_data_with_augmentation(data_path, augment=False):
    """加载数据并可选地进行增强"""
    features, labels = [], []
    
    if augment:
        datagen = ImageDataGenerator(**AUGMENTATION_PARAMS)
    
    for class_idx, class_name in enumerate(sorted(os.listdir(data_path))):
        class_dir = os.path.join(data_path, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = Image.open(img_path).resize(IMG_SIZE)
            
            # 原始图像特征
            features.append(extract_combined_features(img_path))
            labels.append(class_idx)
            
            if augment:
                # 数据增强
                img_array = np.array(img)
                img_array = img_array.reshape((1,) + img_array.shape)
                for batch in datagen.flow(img_array, batch_size=1):
                    aug_img = Image.fromarray(batch[0].astype('uint8'))
                    features.append(extract_combined_features_from_image(aug_img))
                    labels.append(class_idx)
                    break  # 每种增强只生成一个样本
    
    return np.array(features), np.array(labels)

def optimize_model(X_train, y_train):
    """模型选择和参数优化"""
    # 定义候选模型和参数网格
    models = {
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        },
        'SVM': {
            'model': SVC(),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10]
            }
        }
    }
    
    best_model = None
    best_score = 0
    
    for name, config in models.items():
        print(f"\n正在优化 {name}...")
        grid = GridSearchCV(config['model'], config['params'], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
        
        print(f"最佳参数: {grid.best_params_}")
        print(f"最佳交叉验证分数: {grid.best_score_:.2%}")
    
    return best_model

def main():
    # 1. 加载原始数据
    print("加载原始数据...")
    X, y = load_data_with_augmentation(DATA_PATH, augment=False)
    
    # 2. 加载增强后的数据
    print("\n加载增强后的数据...")
    X_aug, y_aug = load_data_with_augmentation(DATA_PATH, augment=True)
    print(f"原始数据量: {len(y)} | 增强后数据量: {len(y_aug)}")
    
    # 3. 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, test_size=TEST_RATIO, random_state=RANDOM_STATE
    )
    
    # 4. 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 5. 模型优化
    print("\n开始模型优化...")
    best_model = optimize_model(X_train, y_train)
    print(f"\n选择的最佳模型: {type(best_model).__name__}")
    
    # 6. 评估最佳模型
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n测试集准确率: {accuracy:.2%}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    # 7. 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix (Optimized Model)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    main()
