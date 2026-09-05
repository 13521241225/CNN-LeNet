import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    # 初始化
    def __init__(self):
        super(LeNet, self).__init__()
        # 卷积层1
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,padding=2)
        # 激活函数
        self.sig = nn.Sigmoid()
        # 平均池化层1
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层2
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 平均池化层2
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 平展层
        self.flatten = nn.Flatten()
        # 线性全连接层1
        self.f5 = nn.Linear(in_features=400, out_features=120)
        # 线性全连接层2
        self.f6 = nn.Linear(in_features=120, out_features=84)
        # 线性全连接层3
        self.f7 = nn.Linear(in_features=84, out_features=10)

    # 前向传播
    def forward(self, x):
        x = self.sig(self.c1(x))
        x = self.s2(x)
        x = self.sig(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型放到设备里
    model = LeNet().to(device)
    print(summary(model, input_size=(1, 28, 28)))
