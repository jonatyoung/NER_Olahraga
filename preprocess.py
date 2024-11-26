import torch
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np


# Dataset Class
class NERDataset(Dataset):
    def __init__(self, path_to_data):
        train = ""
        sentences = []
        labels = []

        with open(path_to_data,"r") as f:
            train = f.read().split("\n\n")
            train = [x.split("\n") for x in train]

        for i in range(len(train)):
            temp_sentence = []
            temp_label = []
            for j in range(len(train[i])):
                train[i][j] = tuple(train[i][j].split())
                temp_sentence.append(train[i][j][0])
                temp_label.append(train[i][j][1])
            sentences.append(temp_sentence)
            labels.append(temp_label)

        # Create Vocabulary for words and labels
        word2idx = defaultdict(lambda: len(word2idx))
        label2idx = defaultdict(lambda: len(label2idx))

        # Add <PAD> token to the vocab
        word2idx["<PAD>"] = 0
        label2idx["O"] = 0

        # Convert words and labels to indices
        input_data = [[word2idx[word] for word in sentence] for sentence in sentences]
        label_data = [[label2idx[label] for label in sentence] for sentence in labels]

        # Padding the sequences
        MAX_LEN = 10
        input_data = [sentence + [0]*(MAX_LEN - len(sentence)) for sentence in input_data]
        label_data = [label + [0]*(MAX_LEN - len(label)) for label in label_data]
        self.inputs = torch.tensor(input_data)
        self.labels = torch.tensor(label_data)
        self.word2idx = word2idx
        self.label2idx = label2idx

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


