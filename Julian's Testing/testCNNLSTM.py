import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
import numpy as np
from NeuroimagingDataset import NeuroimagingDataset
from sklearn.model_selection import KFold

# For command line calling. Use --epochs for nubmer of epochs arg
# Otherwise, default # of epochs is 100
import argparse
parser = argparse.ArgumentParser(description="CNN_LSTM for Schizophrenia Project")
parser.add_argument("--epochs", type=int, help="Number of Epochs")
cmd_args = parser.parse_args()



# 3DCNN part ===================================================================
class Conv3DNet(nn.Module):
    def __init__(self):
        super(Conv3DNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight) # Xavier? Really?
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv3.weight)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(self.pool(F.relu(self.bn3(self.conv3(x)))))
        return x

# LSTM part --------------------------------------------------------------------
class LSTMNet(nn.Module):
    def __init__(self, num_classes=2):
        super(LSTMNet, self).__init__()
        
        self.lstm = nn.LSTM(input_size=128, hidden_size=36, batch_first=True)
        # Xavier init for LSTM
        # for name, param in self.lstm.named_parameters():
            # if 'weight_ih' in name:
                # nn.init.xavier_uniform_(param.data)
            # elif 'weight_hh' in name:
                # nn.init.xavier_uniform_(param.data)
                
        self.fc = nn.Linear(36, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.dropout(x)
        #x = self.fc(x[:, -1, :]) # bad
        
        # Account for all time steps?
        x = torch.mean(x, dim=1)  # Averaging across time steps
        x = self.fc(x)

        return x

# Add them together ------------------------------------------------------------
class CNNLSTMModel(nn.Module):
    def __init__(self):
        super(CNNLSTMModel, self).__init__()
        self.Conv3DNet = Conv3DNet()
        self.LSTMNet = LSTMNet()

    def forward(self, x):
        x = self.Conv3DNet(x)
        x = x.view(x.size(0), -1, 128)  # Adjust shape for LSTM
        x = self.LSTMNet(x)
        return x



# Train the model ==============================================================
def train_model(model, criterion, optimizer, train_loader, val_loader,
                num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        size = 0 # For correct accuracy calculation

        # Iterate over data
        for inputs, labels in train_loader:
            size += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero parameter gradients
            optimizer.zero_grad()

            # Forward
            inputs = inputs.unsqueeze(1) #TODO
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            # Stats
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / size
        epoch_acc = running_corrects.double() / size

        print(f'Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss:.4f}, 
              Acc: {epoch_acc:.4f}')

        # Call eval loop for validation
        eval_model(model, criterion, val_loader)

# Evaluate the model -----------------------------------------------------------
def eval_model(model, criterion, val_loader):
    model.eval()  # Set model to eval
    running_loss = 0.0
    running_corrects = 0
    size = 0
    
    # Iterate over data
    for inputs, labels in val_loader:
        size += len(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward
        with torch.no_grad():
            inputs = inputs.unsqueeze(1) #TODO
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

        # Stats
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / size
    epoch_acc = running_corrects.double() / size

    print(f'Validation Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')



# Let's run the model! =========================================================

# Init device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Path to data folder, path to phenotypic data file, feature types (WIP)
train_dataset = NeuroimagingDataset(
    '/Users/weavejul/Documents/Class_and_Lab/My_Lab_Spring_24/Data/COBRE/ResultsS',
    '/Users/weavejul/Documents/Class_and_Lab/My_Lab_Spring_24/Data/COBRE/phenotypic_data.csv',
    feature_types='ALFF')

# Config options
# Define # of folds and batch size
k_folds = 5
batch_size = 16
kf = KFold(n_splits=k_folds, shuffle=True)


# K-fold cross validation ------------------------------------------------------
for fold, (train_idx, test_idx) in enumerate(kf.split(train_dataset)):
    
    # Init model
    model = CNNLSTMModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Get rid of bad indices
    train_idx = np.array([idx for idx in train_idx if idx not in [70, 83]])
    test_idx = np.array([idx for idx in test_idx if idx not in [70, 83]])
    
    # Define data loaders for current fold
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
    )
    test_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx),
    )

    if(cmd_args.epochs):
        # If specified, use terminal arg for number of epochs
        train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=cmd_args.epochs)
    else:
        # Otherwise default to 10 epochs
        train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=100)

    eval_model(model, criterion, test_loader)
    
    
"""
# Code for one fold


# Define the size of the dataset
total_size = len(train_dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

# Generate a list of all indices
indices = list(range(total_size))

# Bad indices to remove
bad_indices = {70, 83}

# Filter out bad indices
filtered_indices = [i for i in indices if i not in bad_indices]

# Now, split the filtered indices list into train and validation/test parts
split = int(np.floor(0.8 * len(filtered_indices)))
train_indices = filtered_indices[:split]
test_indices = filtered_indices[split:]

# Create data samplers and loaders
#train_sampler = SubsetRandomSampler(train_indices)
#test_sampler = SubsetRandomSampler(test_indices)"""