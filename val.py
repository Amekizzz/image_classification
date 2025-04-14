import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def val():
    # ====================
    # 1. 参数设置
    # ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./data/val"
    batch_size = 32

    # 自动获取类别
    classes = sorted(os.listdir("./data/train"))
    num_classes = len(classes)

    # ====================
    # 2. 定义模型（以 CNN+ViT 融合模型为例）
    # ====================
    from train_CNN_VIT import CNN_ViT_Fusion  # 替换为你的模型定义

    model = CNN_ViT_Fusion(num_classes=num_classes)
    model.load_state_dict(torch.load("./save_model/cnn_vit_classification.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # ====================
    # 3. 数据加载
    # ====================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    val_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ====================
    # 4. 推理并收集结果
    # ====================
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Binarize 真实标签
    y_true = label_binarize(all_labels, classes=range(num_classes))
    y_score = np.array(all_probs)

    # ====================
    # 5. 混淆矩阵
    # ====================
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig("./Curves/confusion_matrix.png")
    plt.close()

    # ====================
    # 6. PR 曲线
    # ====================
    plt.figure(figsize=(12, 8))
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        ap = average_precision_score(y_true[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f'{classes[i]} (AP={ap:.2f})')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc='lower left', fontsize="small")
    plt.grid()
    plt.savefig("./Curves/pr_curve.png")
    plt.close()

    # ====================
    # 7. ROC 曲线
    # ====================
    plt.figure(figsize=(12, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{classes[i]} (AUC={roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc='lower right', fontsize="small")
    plt.grid()
    plt.savefig("./Curves/roc_curve.png")
    plt.close()



if __name__ == "__main__":
    val()
