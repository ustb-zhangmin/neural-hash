import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import hashlib
import random
import matplotlib.pyplot as plt
import torchvision
# Define the neural network model

class HashFunction(nn.Module):
    def __init__(self):
        super(HashFunction, self).__init__()
        
        self.conv1=nn.Sequential(
            nn.Conv1d(64,512,2),
            nn.ReLU(),
            nn.Conv1d(512,256,2),
            nn.ReLU(),
            nn.Conv1d(256,128,2),
            nn.ReLU(),
            nn.MaxPool1d(5)
        )

        self.Lin = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            
            #nn.BatchNorm1d(128),
            #nn.ReLU(),
            #nn.Linear(128, 64),
            #nn.ReLU(),
            #nn.Linear(64, 128),
            
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(len(x), -1)
        x = self.Lin(x)
        return x



# Define the dataset
class HashDataset(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return self.data1[idx],self.data2[idx]

# Contrastive loss
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
    def forward(self, output1, output2):
        loss_mse = torch.mean(1 - torch.nn.MSELoss(reduction='mean')(output1,output2))

        return loss_mse


train_num, test_num = 690000, 10000
data_path = 'data.npy'

# load data
data_dict = np.load(data_path, allow_pickle=True).item()
data = data_dict['data']
similar_data = data_dict['similar_data']

print(data.shape,similar_data.shape)

# Create the dataset and dataloader
train_dataset = HashDataset(data[:train_num],similar_data[:train_num])
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = HashFunction().to(device)
#optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = MSELoss()

# Train the model
num_epochs = 50
test_data = data[train_num:].to(device)
s_test_data = similar_data[train_num:].to(device)
threshold = 10
loss_list = []
dist_list = []
rate = []
for epoch in range(num_epochs):
    for batch_idx, (p_data, s_data) in enumerate(train_dataloader):
        p_data = p_data.to(device)
        s_data = s_data.to(device)
        
        
        outputs1 = model(p_data)
        outputs2 = model(s_data)
        loss = criterion(outputs1, outputs2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_list.append(loss.item())
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        #Ensure that any two outputs are not equal
        distances = []
        qualified = 0
        for i in range(len(test_data)):
            output1 = model(test_data[i].unsqueeze(0).to(device))
            output2 = model(s_test_data[i].unsqueeze(0).to(device))
            distance = torch.dist(output1, output2, p=2)
            distances.append(distance.item())
            if distance > threshold:
                qualified += 1

        avg_distance = np.mean(distances)
        dist_list.append(avg_distance)
        rate.append(qualified/test_num)
        print(f"Average Euclidean distance: {avg_distance}, Rate of pass: {qualified/test_num}")

plt.figure(figsize=(18,6))

plt.subplot(1, 3, 1)
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(num_epochs), loss_list, label='train')
plt.grid(color='k', linestyle=':')
plt.legend()

plt.subplot(1, 3, 2)
plt.title('Average Euclidean distance')
plt.xlabel('epoch')
plt.ylabel('Euclidean distance')
plt.plot(range(num_epochs), dist_list, label='test')
plt.grid(color='k', linestyle=':')
plt.legend()

plt.subplot(1, 3, 3)
plt.title('Rate of pass')
plt.xlabel('epoch')
plt.ylabel('Rate')
plt.plot(range(num_epochs), rate, label='test')
plt.grid(color='k', linestyle=':')
plt.legend()

plt.imshow
plt.show()      
        
