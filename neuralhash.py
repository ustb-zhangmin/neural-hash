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
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 128)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = torch.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
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

def dis_criterion(output1,output2):
    distance = torch.dist(output1, output2,p = 2)
    loss = - torch.mean(distance)
    return loss
'''
# Generate random data and labels
train_num,test_num = 990000, 10000 
changenum = 5
data = [''.join(np.random.choice(['0','1'], 512)) for _ in range(train_num + test_num)]

similar_data = []
#random choose a bit to change
for d in data:
    indices = random.sample(list(range(512)),changenum)
    d_list = list(d)
    for i in indices:
        d_list[i] = '0' if d[i] == '1' else '1'
    similar_data.append(''.join(d_list))

data = torch.tensor([list(map(int, d)) for d in data], dtype=torch.float32)
similar_data = torch.tensor([list(map(int, d)) for d in similar_data], dtype=torch.float32)
'''
train_num, test_num = 1990000, 10000
data_path = 'data.npy'

# load data
data_dict = np.load(data_path, allow_pickle=True).item()
data = data_dict['data']
similar_data = data_dict['similar_data']

print(data.shape,similar_data.shape)

# Create the dataset and dataloader
train_dataset = HashDataset(data[:train_num],similar_data[:train_num])
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = HashFunction().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# Train the model
num_epochs = 1
test_data = data[train_num:].to(device)
s_test_data = similar_data[train_num:].to(device)
threshold = 10
loss_list = []
dist_list = []
rate = []
for epoch in range(num_epochs):
    #optimizer.zero_grad()
    for batch_idx, (p_data, s_data) in enumerate(train_dataloader):
        p_data = p_data.to(device)
        s_data = s_data.to(device)
        
        optimizer.zero_grad()
        outputs1 = model(p_data)
        outputs2 = model(s_data)
        loss = dis_criterion(outputs1, outputs2)
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
            if distance > 128:
                print(output1)
                print(output2)
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
