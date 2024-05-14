from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from model import QuizNet
from utils import load_data, prepare_data, load_model
from torch.utils.data import TensorDataset, DataLoader
import json
import random
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

model = QuizNet()
questions = load_data('English_Quiz.json')

vectorizer = TfidfVectorizer(max_features=100)

def prepare_features(questions):
    vectorizer = TfidfVectorizer(max_features=100)
    corpus = [q['question'] for q in questions]
    tfidf_features = vectorizer.fit_transform(corpus).toarray()
    ids = np.array([[q['id']] for q in questions])
    combined_features = np.hstack((ids, tfidf_features[:, :20]))  # 先頭の20特徴量を使用して次元数を21に
    print("Combined features shape:", combined_features.shape)  # 確認のため出力
    return torch.tensor(combined_features, dtype=torch.float32)

def train_and_save_model(features, labels):
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)

    dataset = TensorDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    model = QuizNet()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 学習率を調整

    model.train()
    for epoch in range(50):  # エポック数を増やして学習を強化
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/50, Loss: {loss.item()}")

    torch.save(model.state_dict(), 'model.pth')

def prepare_data_for_training(correct_ids, incorrect_ids, questions):
    texts = [q['question'] for q in questions]
    features = vectorizer.fit_transform(texts).toarray()

    labels = np.zeros(len(questions))
    for idx in correct_ids:
        if idx < len(questions):
            labels[idx] = 1
    for idx in incorrect_ids:
        if idx < len(questions):
            labels[idx] = 0

    print(f"Correct IDs count: {len(correct_ids)}")
    print(f"Incorrect IDs count: {len(incorrect_ids)}")

    ids = np.array([[q['id']] for q in questions])
    combined_features = np.hstack((ids, features[:, :20]))  # 先頭の20特徴量を使用して次元数を21に

    features_tensor = torch.tensor(combined_features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    
    return features_tensor, labels_tensor

@app.route('/')
def home():
    if 'current_index' not in session or 'question_order' not in session:
        session['current_index'] = 0
        session['correct_answers'] = 0
        session['correct_ids'] = []
        session['incorrect_ids'] = []
        
        full_question_list = list(range(len(questions)))
        random.shuffle(full_question_list)
        session['question_order'] = full_question_list[:min(30, len(questions))]

    if session['current_index'] < len(session['question_order']):
        question_id = session['question_order'][session['current_index']]
        if question_id < len(questions):
            question = questions[question_id]
            return render_template('quiz.html', question=question, question_id=question_id, index=session['current_index'], total_questions=len(session['question_order']), correct_answers=session['correct_answers'])
        else:
            print(f"Invalid question_id: {question_id} - Out of range.")
            return render_template('error.html', message=f"An error occurred: Question ID {question_id} is out of valid range.")

    return redirect(url_for('results'))

@app.route('/submit', methods=['POST'])
def submit():
    if 'question_order' not in session or 'current_index' not in session:
        return redirect(url_for('home'))

    question_id = session['question_order'][session['current_index']]
    user_answer = request.form['answer']
    correct_answer = questions[question_id]['answer']

    if user_answer == correct_answer:
        session['correct_answers'] += 1
        session.get('correct_ids', []).append(question_id)
    else:
        session.get('incorrect_ids', []).append(question_id)

    session.modified = True
    session['current_index'] += 1

    if session['current_index'] >= len(session['question_order']):
        return redirect(url_for('results'))
    else:
        return redirect(url_for('home'))

@app.route('/results', methods=['GET'])
def results():
    correct_answers = session.get('correct_answers', 0)
    correct_ids = session.get('correct_ids', [])
    incorrect_ids = session.get('incorrect_ids', [])
    total_questions = len(session['question_order'])
    
    print("Correct IDs:", correct_ids)
    print("Incorrect IDs:", incorrect_ids)

    features, labels = prepare_data_for_training(correct_ids, incorrect_ids, questions)
    
    # モデルのトレーニングと保存
    train_and_save_model(features, labels)
    session.clear()
    return render_template('results.html', correct_answers=correct_answers, total_questions=total_questions)

@app.route('/suggest_questions', methods=['GET'])
def suggest_questions():
    model = load_model('model.pth')
    questions = load_data('English_Quiz.json')
    question_features = prepare_features(questions)

    # モデルによる予測を取得
    predictions = model(question_features).detach().numpy().flatten()
    sorted_indices = np.argsort(predictions)[::-1]

    # デバッグのため、予測値とそのインデックスを出力
    print("Predictions:", predictions)
    print("Sorted Indices:", sorted_indices)

    # 予測に基づいて問題を選択
    suggested_question_ids = [int(questions[i]['id']) for i in sorted_indices[:30]]
    print("Suggested Question IDs:", suggested_question_ids)

    with open('suggested_questions.json', 'w') as f:
        json.dump(suggested_question_ids, f)

    return redirect(url_for('suggested_quiz'))

@app.route('/suggested_quiz', methods=['GET'])
def suggested_quiz():
    with open('suggested_questions.json', 'r') as f:
        suggested_question_ids = json.load(f)

    if 'current_index' not in session or 'suggested_questions' not in session:
        session['current_index'] = 0
        session['correct_answers'] = 0
        session['correct_ids'] = []
        session['incorrect_ids'] = []
        session['suggested_questions'] = suggested_question_ids

    if session['current_index'] < len(session['suggested_questions']):
        question_id = session['suggested_questions'][session['current_index']]
        if question_id < len(questions):
            question = questions[question_id]
            return render_template('quiz.html', question=question, question_id=question_id, index=session['current_index'], total_questions=len(session['suggested_questions']), correct_answers=session['correct_answers'])
        else:
            print(f"Invalid question_id: {question_id} - Out of range.")
            return render_template('error.html', message=f"An error occurred: Question ID {question_id} is out of valid range.")

    return redirect(url_for('results'))

@app.route('/submit_suggested', methods=['POST'])
def submit_suggested():
    if 'suggested_questions' not in session or 'current_index' not in session:
        return redirect(url_for('suggested_quiz'))

    question_id = session['suggested_questions'][session['current_index']]
    user_answer = request.form['answer']
    correct_answer = questions[question_id]['answer']

    if user_answer == correct_answer:
        session['correct_answers'] += 1
        session['correct_ids'].append(question_id)
        print(f"Question ID {question_id}: Correct")
    else:
        session['incorrect_ids'].append(question_id)
        print(f"Question ID {question_id}: Incorrect (User answer: {user_answer}, Correct answer: {correct_answer})")

    session.modified = True
    session['current_index'] += 1

    if session['current_index'] >= len(session['suggested_questions']):
        return redirect(url_for('results'))
    else:
        return redirect(url_for('suggested_quiz'))

if __name__ == '__main__':
    app.run(debug=True)
