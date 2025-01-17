import os
from collections import Counter
import pickle
from pycocotools.coco import COCO
import nltk


class Vocabulary:
    def __init__(self,
                 vocab_threshold,
                 annotations_path,
                 vocab_file_path="./vocab.pkl",
                 start_word="<start>",
                 end_word="<end>",
                 unk_word="<unk>"):

        self.vocab_threshold = vocab_threshold
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_path = annotations_path
        self.vocab_file_path = vocab_file_path
        self.load_vocab()

    def load_vocab(self):
        if os.path.exists(self.vocab_file_path):
            with open(self.vocab_file_path, "rb") as f:
                vocabulary = pickle.load(f)
                self.word2idx = vocabulary.word2idx
                self.idx2word = vocabulary.idx2word
        else:
            self.build_vocabulary()
            with open(self.vocab_file_path, "wb") as f:
                pickle.dump(self, f)

    def init_vocabulary(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        coco = COCO(self.annotations_path)
        counter = Counter()
        ids = list(coco.anns.keys())

        for _id in ids:
            captions = coco.anns[_id]['caption']
            tokens = nltk.tokenize.word_tokenize(captions.lower())
            counter.update(tokens)

        words = [word for word, count in counter.items() if count >= self.vocab_threshold]
        for word in words:
            self.add_word(word)

    def build_vocabulary(self):
        self.init_vocabulary()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


if __name__ == "__main__":
    vocabulary = Vocabulary(5, "/Users/dsparch/Workspace/Data/COCO/annotations-2/captions_train2014.json")

