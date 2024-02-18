import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Загрузка предварительно обученного биэнкодера DPR и токенизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base").to(device)
tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

# Загрузка данных
knowledge_df = pd.read_csv('./data/knowledge_df.csv')

# Загрузка токенизированного вектора контекста
context_vectors = np.load('./data/context_vectors.npy')

# Функция получения ответа на запрос пользователя
def get_response(query):
    query_vector = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = context_encoder(**query_vector)
    query_vector = outputs.pooler_output[0].cpu().numpy()
    
    similarity_scores = cosine_similarity([query_vector], context_vectors)
    most_similar_index = similarity_scores.argmax()
    col_index = most_similar_index % len(knowledge_df.columns[1:])
    row_index = most_similar_index // len(knowledge_df.columns[1:])
    response = knowledge_df.iloc[row_index, 0]
    
    return response



# Главный цикл чат-бота
print("Чат-бот запущен. Напишите 'пока', чтобы завершить. Общаемся на английском")
while True:
    user_input = input("User: ")
    if user_input.lower() == 'пока':
        print("Чат-бот завершает работу.")
        break
    else:
        response = get_response(user_input)
        print("Ricky-The-Bot:", response)
