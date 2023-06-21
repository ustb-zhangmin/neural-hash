import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import hashlib
import random
import matplotlib.pyplot as plt
# Define the neural network model
class HashFunction(nn.Module):
    def __init__(self):
        super(HashFunction, self).__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.sigmoid((self.fc4(x)))

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
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2):
        euclidean_distance = torch.pairwise_distance(output1, output2)
        #loss_contrastive = torch.mean(torch.relu(self.margin - euclidean_distance))
        loss_contrastive = torch.mean(torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        #loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


train_num, test_num = 390000, 10000
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
optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.SGD(model.parameters(), lr=0.001)
margin = 128
criterion = ContrastiveLoss(margin)

# Train the model
num_epochs = 100
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
