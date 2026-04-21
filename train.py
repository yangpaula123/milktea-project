import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


# 0. 设置随机种子（确保结果复现）
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# 1. 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 2. 数据增强
# ImageNet 标准化参数
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)), #  使用PIL的resize方法将输入图像调整为256x256的固定尺寸
    transforms.RandomCrop(224), #  对图像进行随机裁剪，将图像裁剪为224x224的大小
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.3, #  亮度调整因子，取值范围通常为0-1，值越大图像越亮
        contrast=0.3, #  对比度调整因子，取值范围通常为0-1，值越大对比度越明显
        saturation=0.3, #  饱和度调整因子，取值范围通常为0-1，值越大色彩越鲜艳
        hue=0.1 #  色调调整因子，取值范围通常为0-0.5，用于调整图像的色相
    ),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), #  使用RandomAffine变换进行随机仿射变换     degrees=0 表示不进行旋转     translate=(0.1, 0.1) 表示在水平和垂直方向上进行最多10%的平移
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])


# 3. 加载数据
train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    "dataset/val",
    transform=val_transform
)

# CPU 训练建议使用较小的 batch_size
BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0  # CPU 环境设为 0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = train_dataset.classes
num_classes = len(class_names)

print("类别:", class_names)


# 4. 加载ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 解冻最后几个层进行微调
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name: #  检查当前层名称是否包含'layer4'或'fc'关键字
        param.requires_grad = True
    else:
        param.requires_grad = False

# 修改最后一层，添加dropout
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

model = model.to(device)


# 5. 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([ #  创建随机梯度下降(SGD)优化器，为模型的不同层设置不同的学习率
    {'params': model.conv1.parameters(), 'lr': 0.001}, #  为第一个卷积层(conv1)设置学习率为0.001
    {'params': model.bn1.parameters(), 'lr': 0.001}, #  为第一个批归一化层(bn1)设置学习率为0.001
    {'params': model.layer1.parameters(), 'lr': 0.001}, #  为第一个残差块层(layer1)设置学习率为0.001
    {'params': model.layer2.parameters(), 'lr': 0.001}, #  为第二个残差块层(layer2)设置学习率为0.001
    {'params': model.layer3.parameters(), 'lr': 0.001}, #  为第三个残差块层(layer3)设置学习率为0.001
    {'params': model.layer4.parameters(), 'lr': 0.01}, #  为第四个残差块层(layer4)设置学习率为0.01
    {'params': model.fc.parameters(), 'lr': 0.01} #  为全连接层(fc)设置学习率为0.01
], lr=0.001, momentum=0.9, weight_decay=1e-4)

# 使用ReduceLROnPlateau调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
# mode='max' 表示当验证准确率最高时，学习率保持不变，当验证准确率不再提高时，学习率会降低

# 6. 开始训练
NUM_EPOCHS = 30
best_acc = 0

for epoch in range(NUM_EPOCHS):
    model.train() #  将模型设置为训练模式
    running_loss = 0
    
    # 使用 tqdm 显示进度条
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")

    # ===== 训练 =====
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"Train Loss": f"{running_loss / (progress_bar.n + 1) :.4f}"}) #  更新进度条显示，显示平均训练损失，保留4位小数

    # ===== 验证 =====
    model.eval() #  将模型设置为评估模式
    correct = 0 
    total = 0
    val_loss = 0

    with torch.no_grad():    #  禁用梯度计算，减少内存消耗并加速计算
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0) #  更新总样本数
            correct += (predicted == labels).sum().item() #  更新正确预测的样本数

    acc = 100 * correct / total
    avg_train_loss = running_loss / len(train_loader) #  计算平均训练损失
    avg_val_loss = val_loss / len(val_loader) #  计算平均验证损失

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {acc:.2f}%")

    # 更新学习率调度器
    scheduler.step(acc)

    # 保存最佳模型
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth") #  保存模型状态
        print(f"✓ 保存模型，当前最佳准确率: {best_acc:.2f}%")

print("训练完成！最佳准确率:", best_acc) #  训练完成后打印最佳准确率