import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 参数配置
DATA_PATH = "?????"  # 替换路径
IMG_SIZE = (224, 224)  # ResNet默认输入尺寸
TEST_RATIO = 0.2
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001  # 迁移学习建议更小的学习率

# 加载图像数据
def load_images(data_path):
    features, labels = [], []
    for class_idx, class_name in enumerate(os.listdir(data_path)):
        class_dir = os.path.join(data_path, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = Image.open(img_path).resize(IMG_SIZE).convert("RGB")
            features.append(np.array(img))
            labels.append(class_idx)
    return np.array(features), np.array(labels)

# 加载数据并划分
X, y = load_images(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=42)

# 数据增强（针对训练集）
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 标准化（ResNet的预处理）
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 加载预训练ResNet50（不包含顶层分类器）
base_model = ResNet50(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# 冻结预训练层（迁移学习第一阶段）
base_model.trainable = False

# 自定义顶层分类器
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # 替代Flatten，减少参数量
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')
])

# 编译模型
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 回调函数
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

# 训练模型（第一阶段：仅训练自定义层）
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, checkpoint]
)
# 微调（第二阶段：解冻部分层）
base_model.trainable = True
for layer in base_model.layers[:100]:  # 冻结前100层（根据需求调整）
    layer.trainable = False

# 重新编译（更低的学习率）
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 继续训练
history_fine = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=5,  # 微调阶段epoch较少
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"测试集准确率: {test_acc:.4f}")

# 可视化训练过程
import matplotlib.pyplot as plt
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'])
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Initial Training', 'Fine-tuning'])
plt.show()
