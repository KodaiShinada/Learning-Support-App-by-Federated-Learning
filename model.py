import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

class QuizDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.questions = [d['question_id'] for d in data]
        self.labels = [d['label'] for d in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question_id = self.questions[idx]
        label = self.labels[idx]
        return question_id, label
    
class QuizNet(nn.Module):
    def __init__(self):
        super(QuizNet, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x.squeeze()

def train_model(model, data_loader, epochs=10):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss}')
    return model

