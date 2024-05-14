import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer

class QuizNet(nn.Module):
    def __init__(self):
        super(QuizNet, self).__init__()
        self.fc1 = nn.Linear(21, 50)  # 入力層の次元数を21に変更
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def prepare_data(questions, results):
    vectorizer = TfidfVectorizer(max_features=20)  # 入力サイズに合わせる
    texts = [q['question'] for q in questions]
    tfidf_features = vectorizer.fit_transform(texts).toarray()

    labels = [1 if q['result'] == 'correct' else 0 for q in results]
    ids = [[q['id']] for q in results]

    ids = np.array(ids)
    features = np.hstack((ids, tfidf_features))  # IDとTF-IDF特徴の結合

    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)  # ラベルを2次元に変換

    return features_tensor, labels_tensor

def train_model(model, data_loader, criterion, optimizer):
    model.train()
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'model.pth')
    return model
