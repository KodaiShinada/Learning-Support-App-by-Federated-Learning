import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import json

def load_questions(file_path='English_Quiz.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    return questions

def load_user_data(file_path='user_data.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        user_data = json.load(f)
    return user_data

def filter_incorrect_questions(questions, user_data):
    incorrect_ids = set(user_data['incorrect'])
    incorrect_questions = [question for question in questions if question['id'] in incorrect_ids]
    return incorrect_questions


# 予測機能
def predict(model, questions_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(questions_tensor)
        confidences, predicted_indices = torch.max(F.softmax(outputs, dim=1), dim=1)
        return predicted_indices.tolist(), confidences.tolist()
    
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def load_model(path):
    model = SimpleNN()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

inputs = torch.randn(100, 10)
labels = torch.randint(0, 10, (100,))

dataset = torch.utils.data.TensorDataset(inputs, labels)
loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    for data, target in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

torch.save(model.state_dict(), 'model.pth')