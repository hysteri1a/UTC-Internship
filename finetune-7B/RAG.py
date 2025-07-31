import json
import random
import re
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim



class RAG:
    def __init__(self, similarity_threshold=0.86):
        self.model = SentenceTransformer('thenlper/gte-large-zh', cache_folder=r'./similarity')
        self.similarity_threshold = similarity_threshold

    def cos_similarity(self, target):
        embeddings = self.model.encode(target)
        print(cos_sim(embeddings[0], embeddings[1]))


if __name__ == "__main__":
    augmenter = RAG()
    augmenter.cos_similarity(['客户端证书格式异常','PDF签章报错'])

