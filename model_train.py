import time
import torch
import copy
import pandas as pd
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import LeNet

# 训练集，验证集处理
def train_val_data_process():
    train_data = FashionMNIST(root="./data",
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)

    # 分割数据集，0.8训练，0.2验证
    train_data,val_data = Data.random_split(train_data,[round(len(train_data)*0.8),round(len(train_data)*0.2)])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                   batch_size=128,
                                   shuffle=True,
                                   num_workers=8)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                   batch_size=128,
                                   shuffle=True,
                                   num_workers=8)

    return train_dataloader,val_dataloader

# 模型训练
def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 优化器（本质梯度下降法更新参数）（Adam）
        # lr为学习率
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    # 损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()

    # 将模型放在训练设备中
    model = model.to(device)

    # 复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0

    # 训练集每epoch损失函数列表
    train_loss_all = []

    # 验证集每epoch损失函数列表
    val_loss_all = []

    # 训练集准确度函数列表
    train_acc_all = []

    # 验证集准确度函数列表
    val_acc_all = []

    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("-" * 10)
        print("Epoch {}/{}".format(epoch+1,num_epochs))

        # 初始化参数
        # 训练集损失
        train_loss = 0.0
        # 训练集准确度
        train_correct = 0.0
        # 测试集损失
        val_loss = 0.0
        # 测试机准确度
        val_correct = 0.0
        # 已训练轮次
        train_num = 0
        # 已测试轮次
        val_num = 0

        # 对每一个batch训练和计算
        for step,(b_x,b_y) in enumerate(train_dataloader):
            # 将特征放入训练设备中
            b_x = b_x.to(device)
            # 将标签放入训练设备中
            b_y = b_y.to(device)
            # 模型设置为训练模式
            model.train()

            # 前向传播，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的索引
            pre_lab = torch.argmax(output,dim=1)
            # 计算每一个batch的损失
            loss = criterion(output,b_y)

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 网络参数更新
            optimizer.step()

            # 对损失进行累加，为求每epoch的平均样本损失
            train_loss += loss.item() * b_x.size(0)
            # 对准确度进行累加，为求每epoch的平均样本准确度
            train_correct += torch.sum(pre_lab == b_y)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)

        # 对每一个batch验证
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征放入验证设备中
            b_x = b_x.to(device)
            # 将标签放入验证设备中
            b_y = b_y.to(device)
            # 模型设置为验证模式
            model.eval()

            # 前向传播，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的索引
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失
            loss = criterion(output, b_y)

            # 对损失进行累加，为求每epoch的平均样本损失
            val_loss += loss.item() * b_x.size(0)
            # 对准确度进行累加，为求每epoch的平均样本准确度
            val_correct += torch.sum(pre_lab == b_y)
            # 当前用于验证的样本数量
            val_num += b_x.size(0)

        # 计算每epoch的平均训练损失
        train_loss_all.append(train_loss / train_num)
        # 计算每epoch的平均验证损失
        val_loss_all.append(val_loss / val_num)
        # 计算每epoch的平均训练准确度
        train_acc_all.append(train_correct.double().item() / train_num)
        # 计算每epoch的平均验证准确度
        val_acc_all.append(val_correct.double().item() / val_num)

        print("[{}] train loss:{:.4f} train acc: {:.4f}".format(epoch+1, train_loss_all[-1], train_acc_all[-1]))
        print("[{}] val loss:{:.4f} val acc: {:.4f}".format(epoch+1, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前最高准绝度对应的参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 训练+验证耗时
        time_use = time.time() - since
        print("训练和验证耗时{:.0f}m{:.0f}s".format(time_use//60, time_use%60))

    # 保存最高准确率下的模型参数
    torch.save(best_model_wts,"best_model.pth")


    train_process = pd.DataFrame(data = {"epoch":range(1,num_epochs+1),
                                         "train_loss_all":train_loss_all,
                                         "val_loss_all":val_loss_all,
                                         "train_acc_all":train_acc_all,
                                         "val_acc_all":val_acc_all})

    return train_process

# 可视化绘图
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))

    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, "ro-", label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, "bs-", label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, "ro-", label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, "bs-", label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()

# 主函数
if __name__ == "__main__":
    # 将模型实例化
    LeNet = LeNet()
    # 加载数据集
    train_dataloader, val_dataloader = train_val_data_process()
    # 模型训练
    train_process = train_model_process(LeNet,train_dataloader,val_dataloader,20)
    # 可视化
    matplot_acc_loss(train_process)









