import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, vit_b_16
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap, QImage
from PIL import Image
import numpy as np
import os

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自动获取类别名称（文件夹名称）
train_dir = "./data/train"
classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# print("加载的类别标签:", classes)  # 确保类别正确



# CNN+ViT 模型定义（与训练时相同）
class CNN_ViT_Fusion(nn.Module):
    def __init__(self, num_classes):
        super(CNN_ViT_Fusion, self).__init__()

        self.cnn = resnet50(weights=None)  # 不加载预训练权重
        self.cnn.fc = nn.Identity()
        cnn_out_dim = 2048

        self.vit = vit_b_16(weights=None)
        self.vit.heads = nn.Identity()
        vit_out_dim = 768

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + vit_out_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_features = self.cnn(x)
        vit_features = self.vit(x)
        fusion = torch.cat((cnn_features, vit_features), dim=1)
        return self.fc(fusion)


# 加载模型
num_classes = 22
model = resnet50(weights=None)
model.fc = nn.Linear(2048, num_classes)  # 修改分类头
model.to(device)
model.load_state_dict(torch.load("./save_model/ResNet_classification.pth", map_location=device))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# PyQt 界面
class ImageClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("卫星图像分类")
        self.setGeometry(100, 100, 500, 600)

        self.label = QLabel(self)
        self.label.setFixedSize(400, 400)

        self.btn_load = QPushButton("导入图片", self)
        self.btn_load.clicked.connect(self.load_image)

        self.result_label = QLabel("分类结果: ", self)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            # 显示图片
            pixmap = QPixmap(file_path)
            self.label.setPixmap(pixmap.scaled(400, 400))

            # 进行分类
            predicted_class = self.classify_image(file_path)
            self.result_label.setText(f"分类结果: {predicted_class}")

    def classify_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            _, pred = torch.max(output, 1)
            return classes[pred.item()]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifier()
    window.show()
    sys.exit(app.exec())
