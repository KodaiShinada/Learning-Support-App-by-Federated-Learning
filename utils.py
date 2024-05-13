import torch
import json
import random

def get_random_questions(data, num_questions=30):
    return random.sample(data, num_questions)

def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def select_questions(model, all_questions, num_questions=30):
    probabilities = model(all_questions)  # 誤答は正の重み
    weights = probabilities + (all_questions * -0.5)  # 正解は負の重み
    selected_questions = torch.topk(weights, num_questions)
    return selected_questions.indices
