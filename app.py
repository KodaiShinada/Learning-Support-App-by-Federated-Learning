import json
import torch
import random 
from flask import Flask, render_template, request, session, redirect, url_for
from model import QuizNet, train_model, QuizDataset
from utils import load_data, select_questions

app = Flask(__name__)

num_questions = 30
model = QuizNet(num_questions)

questions = load_data('English_Quiz.json')

@app.route('/', methods=['GET', 'POST'])
def home():
    if 'current_index' not in session or 'question_order' not in session:
        session['current_index'] = 0
        session['correct_answers'] = 0
        session['correct_ids'] = []
        session['incorrect_ids'] = []
        
        full_question_list = list(range(len(questions)))
        random.shuffle(full_question_list)
        session['question_order'] = full_question_list[:30]

    if request.method == 'POST':
        user_answer = request.form.get('answer')
        if session['current_index'] < len(session['question_order']):
            question_id = session['question_order'][session['current_index']]
            correct_answer = questions[question_id]['answer']

            if user_answer == correct_answer:
                session['correct_answers'] += 1
                session['correct_ids'].append(question_id)
            else:
                session['incorrect_ids'].append(question_id)
        
        session['current_index'] += 1
        if session['current_index'] >= len(session['question_order']):
            return redirect(url_for('results'))

    if session['current_index'] < len(session['question_order']):
        question_id = session['question_order'][session['current_index']]
        question = questions[question_id]
        return render_template('quiz.html', question=question, index=session['current_index'], total_questions=len(session['question_order']), correct_answers=session['correct_answers'])
    else:
        return redirect(url_for('results'))

@app.route('/results')
def results():
    correct_answers = session.get('correct_answers', 0)
    total_questions = 30
    correct_ids = session.get('correct_ids', [])
    incorrect_ids = session.get('incorrect_ids', [])

    results_data = {
        "correct": correct_ids,
        "incorrect": incorrect_ids
    }
    with open('results.json', 'w') as f:
        json.dump(results_data, f, indent=4)
    
    session.clear()
    return render_template('results.html', correct_answers=correct_answers, total_questions=total_questions)

if __name__ == '__main__':
    app.run(debug=True)
