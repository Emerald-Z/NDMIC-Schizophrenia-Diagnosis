import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

'''
    Takes in 64 x 64 x 40 shape activity maps
'''
class CNNLSTM(nn.Module):
    def __init__(self):
        # TODO:: code style can make conv blocks
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, 3)# check input layers isn't it 40?
        self.conv2 = nn.Conv3d(32, 64, 3) # 0 padding?
        self.conv3 = nn.Conv3d(64, 128, 3) # TODO: initialize weights

        nn.init.xavier_uniform(self.conv1.weight) #???

        self.maxpool = nn.MaxPool3d(2) 
        self.dropout = nn.Dropout3d(0.5) # TODO: strides

        self.lstm = nn.LSTM(128, 128, batch_first=True) # what should the sizes be

        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(self.maxpool(out))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(self.maxpool(out))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.dropout(self.maxpool(out))

        # TODO: reshape
        x = x.view(x.size(0), 128, 36) # 128 x 36
        out = self.dropout(out)
        out, _ = self.lstm(out)
        return out

# can move this to another file
    
model = CNNLSTM()
criterion = nn.CrossEntropyLoss(); # TODO: no idea
optimizer = optim.Adam(model.parameters(), lr=0.0005)

N_EPOCHS=50
BATCH_SIZE=16

# create dataloader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(N_EPOCHS):
    for id_batch, (x_batch, y_batch) in enumerate(dataloader):

        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if id_batch % 100 == 0:
            print()