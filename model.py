import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        x = self.fc(x)
        return torch.sigmoid(x)

def train_model(model, data_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        for questions, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(questions.float())
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

