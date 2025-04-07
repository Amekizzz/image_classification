import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, vit_b_16, ResNet50_Weights, ViT_B_16_Weights
import matplotlib.pyplot as plt
import os

# 数据增强 & 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据
train_dataset = torchvision.datasets.ImageFolder(root="./data/train", transform=transform)
val_dataset = torchvision.datasets.ImageFolder(root="./data/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

num_classes = len(train_dataset.classes)  # 获取类别数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN_ViT_Fusion(nn.Module):
    def __init__(self, num_classes):
        super(CNN_ViT_Fusion, self).__init__()

        # 加载 ResNet50 (去掉最后的全连接层)
        self.cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.cnn.fc = nn.Identity()  # 移除 ResNet50 的全连接层
        cnn_out_dim = 2048  # ResNet50 的特征维度

        # 加载 Vision Transformer（去掉分类头）
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()  # 移除 ViT 的分类层
        vit_out_dim = 768  # ViT-B/16 的特征维度

        # 组合特征 (ResNet + ViT)
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + vit_out_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_features = self.cnn(x)  # ResNet 提取的特征
        vit_features = self.vit(x)  # ViT 提取的特征

        fusion = torch.cat((cnn_features, vit_features), dim=1)  # 拼接两个特征
        out = self.fc(fusion)  # 通过全连接层分类
        return out


# 实例化模型
model = CNN_ViT_Fusion(num_classes).to(device)


def train(model, train_loader, val_loader, num_epochs=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        # 验证模型
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        val_loss /= len(val_loader)

        # 记录结果
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 画 loss/accuracy 曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss_list, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_loss_list, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_acc_list, label="Train Acc")
    plt.plot(range(1, num_epochs + 1), val_acc_list, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.savefig("./plot/cnn_vit_curves.png")
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), "./save_model/cnn_vit_classification.pth")


# 训练模型
if __name__ == "__main__":
    train(model, train_loader, val_loader, num_epochs=200, lr=0.00001)

