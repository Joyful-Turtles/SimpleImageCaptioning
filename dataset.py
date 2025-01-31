import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
from vocabulary import Vocabulary
from tqdm import tqdm
import nltk
nltk.download('punkt_tab')


class COCODataset(Dataset):

    def __init__(self, transform, annotations_path, img_dir_path, batch_size):
        super().__init__()
        self.transform = transform
        self.batch_size = batch_size
        self.img_dir_path = img_dir_path
        self.coco = COCO(annotations_path)
        self.ids = list(self.coco.anns.keys())
        self.vocab = Vocabulary(5, annotations_path)

        all_tokens = []
        for index in tqdm(range(len(self.ids))):
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption'].lower()
            tokens = nltk.tokenize.word_tokenize(caption)
            all_tokens.append(tokens)

        self.caption_lengths = [len(token) for token in all_tokens]

    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        image_id = self.coco.anns[ann_id]['image_id']
        image_path = self.coco.loadImgs(image_id)[0]['file_name']

        image = Image.open(os.path.join(self.img_dir_path, image_path)).convert('RGB')
        image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption.lower()))
        caption = []
        caption.append(self.vocab(self.vocab.start_word))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab(self.vocab.end_word))
        caption = torch.Tensor(caption).long()

        return image, caption

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    train = COCODataset(transform_train,
                       annotations_path="/Users/dsparch/Workspace/Data/COCO/annotations-2/captions_train2014.json",
                       img_dir_path="/Users/dsparch/Workspace/Data/COCO/train2014",
                       batch_size=32)

    image, caption = train[0]
    print(image.size, caption)
