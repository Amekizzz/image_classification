import os
import shutil
import random

# 设置路径
train_dir = './data/train'
test_dir = './data/test'

# 遍历每个训练子文件夹
for class_name in os.listdir(train_dir):
    class_train_path = os.path.join(train_dir, class_name)

    # 确保是目录
    if not os.path.isdir(class_train_path):
        continue

    # 获取所有图片文件
    all_images = [f for f in os.listdir(class_train_path)
                  if os.path.isfile(os.path.join(class_train_path, f))]

    # 随机选择200张图片
    selected_images = random.sample(all_images, 200)

    # 创建对应的测试目录
    class_test_path = os.path.join(test_dir, class_name)
    os.makedirs(class_test_path, exist_ok=True)

    # 移动文件
    for img in selected_images:
        src = os.path.join(class_train_path, img)
        dst = os.path.join(class_test_path, img)
        shutil.move(src, dst)
        print(f'Moved: {src} -> {dst}')

print("All files moved successfully!")