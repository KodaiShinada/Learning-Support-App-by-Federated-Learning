import torch
import json
import random

def get_random_questions(data, num_questions=30):
    return random.sample(data, num_questions)

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def select_questions(model, all_questions, num_questions=30):
    probabilities = model(all_questions)  # 誤答は正の重み
    weights = probabilities + (all_questions * -0.5)  # 正解は負の重み
    selected_questions = torch.topk(weights, num_questions)
    return selected_questions.indices

def prepare_data(correct_ids, incorrect_ids, all_questions):
    # 特徴ベクトルとラベルのリストを初期化
    features = []
    labels = []
    
    # 合計IDリストを作成し、それぞれのIDでループ
    total_ids = correct_ids + incorrect_ids
    for q_id in total_ids:
        # 質問IDを特徴として使用（例示）
        # 実際にはここで質問テキストから特徴を抽出する処理が必要
        feature = [q_id]  # 単純化のため、特徴ベクトルはIDのリスト
        features.append(feature)
        
        # ラベルを生成（1: 正解, 0: 不正解）
        if q_id in correct_ids:
            labels.append(1)
        else:
            labels.append(0)
    
    # 特徴ベクトルとラベルをTensorに変換
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    return features_tensor, labels_tensor



