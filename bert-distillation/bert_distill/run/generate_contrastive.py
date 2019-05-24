import argparse

from gensim.models import KeyedVectors


def main():
    path = "/mnt/nvme/Castor-data/embeddings/word2vec/GoogleNews-vectors-negative300.bin"
    parser = argparse.ArgumentParser()
    parser.add_argument("--word2vec_file", type=str, default=path)
    args = parser.parse_args()

    word_vecs = KeyedVectors.load_word2vec_format(args.word2vec_file, binary=True)


if __name__ == "__main__":
    main()
