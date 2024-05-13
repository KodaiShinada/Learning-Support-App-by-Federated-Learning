from flask import Flask, render_template, request, session, redirect, url_for
from model import QuizNet, train_model
from utils import load_data, prepare_data
from torch.utils.data import TensorDataset, DataLoader
import json
import random

app = Flask(__name__)
app.secret_key = 'your_secret_key'

model = QuizNet()
questions = load_data('English_Quiz.json')

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
            return render_template('quiz.html', question=question, index=session['current_index'], total_questions=len(session['question_order']), correct_answers=session['correct_answers'])
        else:
            print(f"Invalid question_id: {question_id} - Out of range.")
            return render_template('error.html', message=f"An error occurred: Question ID {question_id} is out of valid range.")

    return redirect(url_for('results'))


@app.route('/submit', methods=['POST'])
def submit():
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
    global model
    correct_answers = session.get('correct_answers', 0)
    correct_ids = session.get('correct_ids', [])
    incorrect_ids = session.get('incorrect_ids', [])
    total_questions = len(session['question_order'])

    print("Session correct_ids:", correct_ids)
    print("Session incorrect_ids:", incorrect_ids)

    features, labels = prepare_data(correct_ids, incorrect_ids, questions)
    if features is None or labels is None:
        print("No data available to train the model.")
        return render_template('error.html', message="No sufficient data to train the model.")

    dataset = TensorDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # モデルを更新する
    model = train_model(model, data_loader)

    # 結果をJSONファイルに保存
    results_data = {
        "correct_answers": correct_answers,
        "total_questions": total_questions
    }
    with open('results.json', 'w') as f:
        json.dump(results_data, f, indent=4)

    session.clear()
    return render_template('results.html', correct_answers=correct_answers, total_questions=total_questions)


if __name__ == '__main__':
    app.run(debug=True)
