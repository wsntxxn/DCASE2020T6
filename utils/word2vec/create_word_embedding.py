# coding=utf-8
#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from tqdm import tqdm
import fire

import sys
import os
sys.path.append(os.getcwd())
from utils.build_vocab import Vocabulary

def create_embedding(caption_file: str,
                     vocab_file: str,
                     embed_size: int,
                     output: str,
                     **word2vec_kwargs):
    caption_df = pd.read_json(caption_file)
    caption_df["tokens"] = caption_df["tokens"].apply(lambda x: ["<start>"] + [token for token in x] + ["<end>"])

    sentences = list(caption_df["tokens"].values)
    vocabulary = torch.load(vocab_file, map_location="cpu")

    model = Word2Vec(sentences=sentences,
                     size=embed_size,
                     min_count=1,
                     **word2vec_kwargs)
    
    word_embeddings = np.zeros((len(vocabulary), embed_size))
    
    with tqdm(total=len(vocabulary), ascii=True) as pbar:
        for word, idx in vocabulary.word2idx.items():
            if word == "<pad>" or word == "<unk>":
                continue
            word_embeddings[idx] = model.wv[word]
            pbar.update()

    np.save(output, word_embeddings)

    print("Finish writing word2vec embeddings to " + output)


if __name__ == "__main__":
    fire.Fire(create_embedding)



