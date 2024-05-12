import torch
import json

def load_data():
    with open('English_Quiz.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]

    questions_tensor = torch.tensor(questions, dtype=torch.long)
    answers_tensor = torch.tensor(answers, dtype=torch.long)

    return questions_tensor, answers_tensor
