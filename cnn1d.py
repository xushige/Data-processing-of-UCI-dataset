import torch
import torch.nn as nn
import torch.utils.data as d
from collections import Counter
import numpy as np
import os
import matplotlib.pyplot as plt
from rexnetv1 import*
X_train = torch.from_numpy(np.load('UCI/x_train.npy'))
X_test = torch.from_numpy(np.load('UCI/x_test.npy'))
Y_train = torch.from_numpy(np.load('UCI/y_train.npy'))
Y_test = torch.from_numpy(np.load('UCI/y_test.npy'))

X_train = X_train.reshape(X_train.size(0), 1, X_train.size(1), X_train.size(2))
X_test = X_test.reshape(X_test.size(0), 1, X_test.size(1), X_test.size(2))

category = len(Counter(Y_train.tolist()))
print(X_train.size(), X_test.size(), 'classes=', category)
train_data = d.TensorDataset(X_train, Y_train)
test_data = d.TensorDataset(X_test, Y_test)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((5, 1)),

            nn.Conv2d(64, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((5, 1)),

            nn.Conv2d(128, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((5, 1))
        )
        self.flc = nn.Linear(256*3, category)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.flc(x)
        return x

net = ReXNetV1(input_ch=1,final_ch=180,classes=category).cuda()
EP = 100
B_S = 128

train_loader = d.DataLoader(train_data, batch_size=B_S, shuffle=True)
test_loader = d.DataLoader(test_data, batch_size=B_S, shuffle=True)

LR = 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

for i in range(EP):
    cor = 0
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        net.train()
        out = net(data)
        loss = loss_fn(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for data, label in test_loader:
        data, label = data.cuda(), label.cuda()
        net.eval()
        out = net(data)
        _, pre = torch.max(out, 1)
        cor += (pre == label).sum()

    acc = cor.cpu().numpy()/len(test_data)
    print('epoch:%d, loss:%f, acc:%f' % (i, loss, acc))
