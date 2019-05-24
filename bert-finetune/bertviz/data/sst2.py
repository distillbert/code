import hashlib
import os

import torch
import torch.data as data

import bertviz.utils as utils


def load_file(data_file):
    with open(data_file) as f:
        lines = f.readlines()
    input_examples = []
    for line in lines:
        label, text = line.strip().split(" ", 1)
        guid = int(hashlib.md5(text).hexdigest(), 16)
        input_examples.append(utils.InputExample(guid, text, None, label))
    return input_examples


def sst2_splits(data_folder):
    filenames = list(map(lambda x: os.path.join(data_folder, x), ("stsa.binary.phrases.train", "stsa.binary.dev", "stsa.binary.test")))
    return list(map(load_file, filenames))
