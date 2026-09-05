from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

# 下载训练数据
train_data = FashionMNIST(root="./data",
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                          download=True)

# 加载训练数据
    # shuffle为是否打乱
    # num_workers为加载数据进程
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

# 获得一个Batch的数据
    # enumerate：遍历序列，同时拿索引+元素
    # 元素形式为(image_tensor,label)
for step,(b_x, b_y) in enumerate(train_loader):
    if step > 0 :
        break
# 移除四维张量里大小为1的维度
    # 四维张量：[batch,channel,H,W]
batch_x = b_x.squeeze().numpy()
# 将张量转换成Numpy数组
batch_y = b_y.numpy()
# 训练集的标签
class_label = train_data.classes
# print(class_label)

# 可视化一个Batch的图像
plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]], size=10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()