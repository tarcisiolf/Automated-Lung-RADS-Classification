import json
import numpy as np
import os
from simcse import SimCSE
import faiss
    
from transformers import AutoModel

def read_data(file_name):
    return json.load(open(file_name, encoding="utf-8"))


def write_file(dir_, data):
    file = open(dir_, "w")
    for item in data:
        file.write(json.dumps(item, ensure_ascii=False)+'\n')
    file.close()

def compute_simcse_knn(test_data, train_data, knn_num, test_index=None):

    sim_model = SimCSE(r"C:\Users\tarci\OneDrive\Área de Trabalho\Mestrado\roberta_large")

    train_sentence = []
    train_sentence_index = []

    for idx_, item in enumerate(train_data):
        text = item["text"]
        train_sentence.append(text)
        train_sentence_index.append(idx_)

    
    embeddings = sim_model.encode(train_sentence, batch_size=128, normalize_to_unit=True, return_numpy=True)
    quantizer = faiss.IndexFlatIP(embeddings.shape[1])
    index = quantizer
    index.add(embeddings.astype(np.float32))
    # 10 is a default setting in simcse
    index.nprobe = min(10, len(train_sentence))
    index = faiss.index_gpu_to_cpu(index)
    train_index = index

    example_idx = []
    example_value = []

    if test_index is None:
        for idx_ in range(len(test_data)):
            context = test_data[idx_]["text"]

            embedding = sim_model.encode([context], batch_size=128, normalize_to_unit=True, keepdim=True, return_numpy=True)
            top_value, top_index = train_index.search(embedding.astype(np.float32), knn_num)

            example_idx.append([train_sentence_index[int(i)] for i in top_index[0]])
            example_value.append([float(value) for value in top_value[0]])
        
        return example_idx, example_value

if __name__ == '__main__':
    test_data = read_data(r"data\all_lung_nodule\few_shot\test_samples.json")
    train_data = read_data(r"data\all_lung_nodule\few_shot\train_samples.json")
    index_, value_ = compute_simcse_knn(test_data=test_data, train_data=train_data, knn_num=8)
    write_file(dir_= r"data\all_lung_nodule\few_shot\test.100.simcse.dev.8.knn.jsonl", data=index_)
