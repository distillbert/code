from collections import defaultdict
import argparse
import functools
import random
import sys

from gensim.models import KeyedVectors
from tqdm import tqdm
import nltk
import pandas as pd
import torch.nn as nn

import bert_distill.data as dat
import bert_distill.model.bert as bt


class ContrastiveSampler(object):

    def __init__(self, word_vectors):
        self.word_vectors = word_vectors

    @functools.lru_cache(maxsize=100000)
    def most_similar(self, word, topn=15):
        return self.word_vectors.most_similar(word, topn=topn)

    def __hash__(self):
        return hash(type(self))

    def __call__(self, word, topn=3):
        try:
            vecs = self.most_similar(word, topn=topn)
            return random.choice(vecs)[0].replace("_", " ")
        except:
            return word


def process_gen_batch(bert, gen_batch, dataset, single=True):
    if single:
        continue_txts, fin_txts = bert.iterative_batch_mask_predict(gen_batch, single=True)
    else:
        fin_txts = bert.iterative_batch_mask_predict(gen_batch, single=False)
    for sent in fin_txts:
        if sent not in dataset:
            dataset.add(sent)
            print(sent)
    if single:
        return continue_txts


def reconstruct_allennlp(gen_words, tokens):
    offset_idx = 0
    join_words = []
    for gen_word, token in zip(gen_words, tokens):
        if token.idx != offset_idx:
            join_words.append(" ")
        offset_idx = token.idx + len(token.text)
        join_words.append(gen_word)
    return "".join(join_words)


def main():
    path = "/mnt/nvme/Castor-data/embeddings/word2vec/GoogleNews-vectors-negative300.bin"
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_prob", default=0.1, type=float)
    parser.add_argument("--random_prob", default=0.1, type=float)
    parser.add_argument("--contrastive_prob", default=0, type=float)
    parser.add_argument("--bert_gen_prob", default=0, type=float)
    parser.add_argument("--window_prob", default=0, type=float)
    parser.add_argument("--window_lengths", default=[1, 2, 3, 4, 5], nargs="+", type=int)
    parser.add_argument("--n_iter", default=20, type=int)
    parser.add_argument("--dataset_file", type=str)
    parser.add_argument("--word2vec_file", type=str, default=path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", type=str, default="nltk", choices=["allennlp", "nltk"], 
        help="Use NLTK for now.")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    random.seed(args.seed)
    if args.contrastive_prob:
        wvs = KeyedVectors.load_word2vec_format(args.word2vec_file, binary=True)
        sampler = ContrastiveSampler(wvs)

    if args.bert_gen_prob:
        bert = bt.BertMaskedLMWrapper.load('bert-large-uncased')
    df = pd.read_csv(args.dataset_file, sep="\t")
    vocab = set()
    dataset = set()
    pos_dict = defaultdict(list)
    for sentence in tqdm(df["sentence"]):
        if args.tokenizer == "nltk":
            words = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
        else:
            toks = dat.allennlp_full_tokenize(sentence)
            words = [tok.text for tok in toks]
            pos_tags = [(tok.text, tok.pos) for tok in toks]
        for word, pos_tag in pos_tags:
            vocab.add(word)
            pos_dict[pos_tag].append(word)
        dataset.add(" ".join(words))

    vocab = list(vocab)
    gen_batch = []
    sorted_sents = sorted(df["sentence"], key=lambda x: len(x.split()), reverse=True)

    for sentence in tqdm(df["sentence"]):
        while len(gen_batch) >= args.batch_size:
            gen_batch = process_gen_batch(bert, gen_batch, dataset)
        if args.tokenizer == "nltk":
            words = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
        else:
            toks = dat.allennlp_full_tokenize(sentence)
            words = [tok.text for tok in toks]
            pos_tags = [(tok.text, tok.pos) for tok in toks]
        for _ in range(args.n_iter):
            use_bert_gen = False
            use_windowing = random.random() < args.window_prob
            mask_prob = args.mask_prob
            gen_words = []
            all_replaced = True
            for idx, (word, pos_tag) in enumerate(zip(words, pos_tags)):
                roll = random.random()
                if roll < mask_prob:
                    word = "[MASK]"
                elif roll < mask_prob + args.random_prob:
                    word = random.choice(pos_dict[pos_tag[1]])
                elif roll < mask_prob + args.random_prob + args.bert_gen_prob:
                    word = "[UNK]"
                    use_bert_gen = True
                else:
                    all_replaced = False
                gen_words.append(word)

            if use_windowing:
                window_len = random.choice(args.window_lengths)
                try:
                    idx = random.randrange(len(gen_words) - window_len)
                    gen_words = gen_words[idx:idx + window_len]
                except ValueError:
                    break
            if args.tokenizer == "nltk":
                gen_sentence = " ".join(gen_words)
            else:
                gen_sentence = reconstruct_allennlp(gen_words, toks)
            if use_bert_gen:
                gen_batch.append(gen_sentence)
                continue
            if not all_replaced and gen_sentence not in dataset:
                dataset.add(gen_sentence)
                print(gen_sentence)
    if len(gen_batch) > 0:
        process_gen_batch(bert, gen_batch, dataset, single=False)


if __name__ == "__main__":
    main()
