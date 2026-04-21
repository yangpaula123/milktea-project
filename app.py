from flask import Flask, render_template, request
import os #  导入操作系统接口模块
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image #  导入Python图像处理库

app = Flask(__name__) #  创建Flask应用实例
UPLOAD_FOLDER = "uploads" #  设置上传文件夹路径
os.makedirs(UPLOAD_FOLDER, exist_ok=True) #  创建上传文件夹，如果不存在则创建

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别名称（顺序与训练一致）
classes = ['coco', 'heytea', 'mixue', 'naixue']

# 加载模型（与 train.py 保持一致）
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 4)
)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval() #  将模型设置为评估模式 在评估模式下，Dropout层会被禁用，确保模型在推理时保持一致性

# 预处理（与 train.py 的 val_transform 保持一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            prob = torch.softmax(outputs, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            score = prob[0][pred].item() #  获取预测类别的置信度分数

        return classes[pred], round(score * 100, 2), {classes[i]: round(prob[0][i].item() * 100, 2) for i in range(len(classes))}
    except Exception as e:
        return None, None, str(e)

@app.route("/", methods=["GET", "POST"]) #  定义一个路由，处理GET和POST请求
def index():
    result = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            label, score, probs = predict(path)
            if label:
                result = {
                    "filename": file.filename,
                    "label": label,
                    "score": score,
                    "probs": probs
                }
            else:
                error = probs  # probs here is error message
        else:
            error = "请上传图片文件"

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)