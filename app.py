import json
from flask import Flask, render_template, request, session, redirect, url_for
from model import load_questions

app = Flask(__name__)
app.secret_key = 'your_secret_key'

questions = load_questions()

def save_quiz_results(correct_ids, incorrect_ids):
    results = {
        "correct": correct_ids,
        "incorrect": incorrect_ids
    }
    with open('quiz_results.json', 'w') as f:
        json.dump(results, f)

@app.route('/', methods=['GET', 'POST'])
def home():
    if 'current_index' not in session:
        session['current_index'] = 0
        session['correct_answers'] = 0
        session['correct_ids'] = []
        session['incorrect_ids'] = []

    if request.method == 'POST':
        if 'correct_ids' not in session:
            session['correct_ids'] = []
        if 'incorrect_ids' not in session:
            session['incorrect_ids'] = []

        user_answer = request.form.get('answer')
        current_index = session['current_index']
        correct_answer = questions[current_index]['answer']

        if user_answer == correct_answer:
            session['correct_answers'] += 1
            session['correct_ids'].append(questions[current_index]['id'])
        else:
            session['incorrect_ids'].append(questions[current_index]['id'])

        session['current_index'] += 1

        if session['current_index'] >= len(questions):
            save_quiz_results(session['correct_ids'], session['incorrect_ids'])  # 結果を保存
            return redirect(url_for('results'))

    if session['current_index'] < len(questions):
        current_question = questions[session['current_index']]
        return render_template('quiz.html', question=current_question, index=session['current_index'], total_questions=len(questions), correct_answers=session['correct_answers'])
    else:
        return redirect(url_for('results'))

@app.route('/results')
def results():
    correct_answers = session.get('correct_answers', 0)
    total_questions = len(questions)
    session.clear()
    return render_template('results.html', correct_answers=correct_answers, total_questions=total_questions)


if __name__ == '__main__':
    app.run(debug=True)
