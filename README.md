# CNN-LeNet

基于 PyTorch 实现的经典 **LeNet** 卷积神经网络，用于 **FashionMNIST** 服装图像分类（10 类）。

## 项目结构

| 文件 | 说明 |
|------|------|
| `model.py` | LeNet 网络结构定义 |
| `model_train.py` | 模型训练与验证，自动保存最优权重 `best_model.pth` |
| `model_test.py` | 模型测试，输出测试集准确率与逐样本预测结果 |
| `plot.py` | FashionMNIST 数据可视化 |
| `best_model.pth` | 训练后生成的最优模型权重（不纳入版本控制） |
| `data/` | FashionMNIST 数据集（运行脚本时自动下载，不纳入版本控制） |

## 环境依赖

- Python 3.x
- PyTorch
- torchvision
- pandas
- matplotlib
- numpy
- torchsummary

```bash
pip install torch torchvision pandas matplotlib numpy torchsummary
```

## 网络结构

LeNet 变体（输入 28×28 灰度图）：

1. Conv2d(1→6, 5×5, padding=2) + Sigmoid
2. AvgPool2d(2×2, stride=2)
3. Conv2d(6→16, 5×5) + Sigmoid
4. AvgPool2d(2×2, stride=2)
5. Flatten → Linear(400→120) → Linear(120→84) → Linear(84→10)

## 使用方法

### 训练

```bash
python model_train.py
```

训练集按 8:2 划分训练/验证集，Adam 优化器（lr=0.001），交叉熵损失，训练过程中自动保存验证集准确率最高的权重到 `best_model.pth`。

### 测试

```bash
python model_test.py
```

加载 `best_model.pth`，在测试集上评估，输出整体准确率并逐样本打印预测类别与真实类别。

### 数据可视化

```bash
python plot.py
```

展示一批 FashionMNIST 训练图像及其标签。

## 数据集

[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)：28×28 灰度服装图像，共 10 个类别：

T-shirt/top、Trouser、Pullover、Dress、Coat、Sandal、Shirt、Sneaker、Bag、Ankle boot
