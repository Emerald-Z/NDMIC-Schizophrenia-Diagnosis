import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import zoom


'''
    Takes in 64 x 64 x 40 shape activity maps
'''
class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, 3)# check input layers isn't it 40???
        self.conv2 = nn.Conv3d(32, 64, 3) 
        self.conv3 = nn.Conv3d(64, 128, 3)

        nn.init.xavier_uniform_(self.conv1.weight) #???
        nn.init.xavier_uniform_(self.conv2.weight) 
        nn.init.xavier_uniform_(self.conv3.weight) 

        self.maxpool = nn.MaxPool3d(2) 
        self.dropout = nn.Dropout(0.5)

        self.lstm = nn.LSTM(128, 36, batch_first=True) # what should the sizes be

        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)

        self.relu = nn.ReLU()
        
        # fully connected layer for binary classification
        self.fc = nn.Linear(36, 2) 


    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(self.maxpool(out))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(self.maxpool(out))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.dropout(self.maxpool(out))

        # Flatten the output for LSTM
        out = out.view(out.size(0), -1, 128)  # Adjust the shape for LSTM
        out = self.dropout(out)
        out, _ = self.lstm(out)
        #out = out[:, -1, :] # Get last hidden state for each example in batch
        out = torch.mean(out, dim=1)  # Average across the sequence maybe?

        # FC
        out = self.fc(out)
        
        return out

# can move this to another file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
model = CNNLSTM().to(device)
criterion = nn.CrossEntropyLoss(); # TODO: no idea
optimizer = optim.Adam(model.parameters(), lr=0.0005)

N_EPOCHS=50
BATCH_SIZE=16

# create dataloader
#dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
# STEP 1: training on full dataset
def train(model, device, dataloader, optimizer, criterion, n_epochs, save=True):
    model.train()
    for epoch in range(n_epochs):
        print("\nEpoch:", epoch)
        for id_batch, (x_batch, y_batch) in enumerate(dataloader):
            if id_batch in [70, 83]:
                continue
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = x_batch.unsqueeze(1)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if save:
        torch.save(model.state_dict(), 'model_checkpoint_e')
        