import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 1. 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 2. 类别名称（顺序必须和训练时一致）
classes = ['coco', 'heytea', 'mixue', 'naixue']

# 3. 加载模型（与 train.py 保持一致）
model = models.resnet18(weights=None)  # 使用 weights=None 替代 deprecated 的 pretrained=False
model.fc = nn.Sequential(
    nn.Dropout(0.5), 
    nn.Linear(model.fc.in_features, 4)
)
model.load_state_dict(torch.load("best_model.pth", map_location=device))  # 加载到指定设备
model.to(device)  # 移到设备
model.eval() #  将模型设置为评估模式

# 5. 图片预处理（与 train.py 的 val_transform 保持一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 标准化
])

# 6. 预测函数
def predict_image(img_path):
    if not os.path.exists(img_path):  # 检查文件是否存在
        print(f"错误：文件 {img_path} 不存在！")
        return #  提前终止函数执行
    
    try:
        img = Image.open(img_path).convert("RGB") #  使用PIL库打开图片，并转换为RGB格式
        img = transform(img).unsqueeze(0).to(device)  # 移到设备 #  对图片进行预处理，并添加批次维度，最后将图片移动到指定设备
        
        with torch.no_grad():
            outputs = model(img)
            prob = torch.softmax(outputs, dim=1) #  对模型输出应用softmax函数，获取每个类别的概率
            pred = torch.argmax(prob, dim=1).item() #  获取预测结果（概率最高的类别索引）
            score = prob[0][pred].item() #  获取预测结果的置信度（概率值）
        
        print("预测结果：", classes[pred])
        print("置信度：", round(score * 100, 2), "%") #  打印预测的置信度，保留两位小数
    
    except Exception as e: #  捕获并处理可能发生的异常
        print(f"预测失败：{e}") 

# 7. 输入图片路径（使用项目目录中的 test.jpg）
predict_image("test.jpg")

