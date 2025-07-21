import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class BaseConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BaseConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))
    
class InceptionModule(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, npool, **kwargs):
        super(InceptionModule, self).__init__()

        self.branch1 = BaseConv2D(in_channels=in_channels, out_channels=n1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BaseConv2D(in_channels=in_channels, out_channels=n3x3red, kernel_size=1),
            BaseConv2D(in_channels=n3x3red, out_channels=n3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BaseConv2D(in_channels=in_channels, out_channels=n5x5red, kernel_size=1),
            BaseConv2D(in_channels=n5x5red, out_channels=n5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BaseConv2D(in_channels=in_channels, out_channels=npool, kernel_size=1)
        )

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        y = torch.cat([y1, y2, y3, y4], dim=1)
        return y
    
class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.7):
        super(AuxiliaryClassifier, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.relu(self.conv(x))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        y = self.fc2(x)
        return y
    
class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000, use_aux=True):
        super(GoogleNet, self).__init__()
        self.use_aux = use_aux
        self.conv1 = BaseConv2D(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn1 = nn.LocalResponseNorm(size=5)

        self.conv2 = BaseConv2D(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = BaseConv2D(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.lrn2 = nn.LocalResponseNorm(size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionModule(in_channels=192, n1x1=64, n3x3red=96, n3x3=128, n5x5red=16, n5x5=32, npool=32)
        self.inception3b = InceptionModule(in_channels=256, n1x1=128, n3x3red=128, n3x3=192, n5x5red=32, n5x5=96, npool=64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = InceptionModule(in_channels=480, n1x1=192, n3x3red=96, n3x3=208, n5x5red=16, n5x5=48, npool=64)
        self.inception4b = InceptionModule(in_channels=512, n1x1=160, n3x3red=112, n3x3=224, n5x5red=24, n5x5=64, npool=64)
        self.inception4c = InceptionModule(in_channels=512, n1x1=128, n3x3red=128, n3x3=256, n5x5red=24, n5x5=64, npool=64)
        self.inception4d = InceptionModule(in_channels=512, n1x1=112, n3x3red=144, n3x3=288, n5x5red=32, n5x5=64, npool=64)
        self.inception4e = InceptionModule(in_channels=528, n1x1=256, n3x3red=160, n3x3=320, n5x5red=32, n5x5=128, npool=128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionModule(in_channels=832, n1x1=256, n3x3red=160, n3x3=320, n5x5red=32, n5x5=128, npool=128)
        self.inception5b = InceptionModule(in_channels=832, n1x1=384, n3x3red=192, n3x3=384, n5x5red=48, n5x5=128, npool=128)
        self.pool5 = nn.AvgPool2d(kernel_size=7, stride=1)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)

        if self.use_aux:
            self.aux_classifier1 = AuxiliaryClassifier(in_channels=512, num_classes=num_classes)
            self.aux_classifier2 = AuxiliaryClassifier(in_channels=528, num_classes=num_classes)

    def forward(self, x):
        x = self.conv1(x)   # 112x112x64
        x = self.pool1(x)   # 56x56x64
        x = self.lrn1(x)    # 56x56x64

        x = self.conv2(x)   # 56x56x64
        x = self.conv3(x)   # 56x56x192
        x = self.lrn2(x)    # 56x56x192
        x = self.pool2(x)   # 28x28x192

        x = self.inception3a(x) # 28x28x256
        x = self.inception3b(x) # 28x28x480
        x = self.pool3(x)   # 14x14x480

        x = self.inception4a(x) # 14x14x512
        x = self.inception4b(x) # 14x14x512
        aux1_x = x

        x = self.inception4c(x) # 14x14x512
        x = self.inception4d(x) # 14x14x528
        aux2_x = x

        x = self.inception4e(x) # 14x14x832
        x = self.pool4(x)   # 7x7x832

        x = self.inception5a(x) # 7x7x832
        x = self.inception5b(x) # 7x7x1024
        x = self.pool5(x)   # 1x1x1024

        x = self.flatten(x)  # 1024
        y = self.fc(self.dropout(x))  # 1000

        if self.use_aux:
            aux1_y = self.aux_classifier1(aux1_x)
            aux2_y = self.aux_classifier2(aux2_x)
            return y, aux1_y, aux2_y
        else:
            return y

if __name__=="__main__":

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # model = InceptionModule(in_channels=3, n1x1=64, n3x3red=64, n3x3=128, n5x5red=64, n5x5=128, npool=64).to(device)
    # model = BaseConv2D(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1).to(device)
    model = GoogleNet(use_aux=True).to(device)

    train_dataset = datasets.FakeData(size=10000, num_classes=1000, transform=transforms.ToTensor())
    test_dataset = datasets.FakeData(size=1000, num_classes=1000, transform=transforms.ToTensor())

    train_dataloader = DataLoader(train_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    for i, batch in enumerate(train_dataloader):
        image, label = batch
        image = image.to(device)
        label = label.to(device)
        print(image.shape)

        pred, pred_aux1, pred_aux2 = model(image)
        print(pred.shape)
        print(pred_aux1.shape)
        print(pred_aux2.shape)
        exit(0)
