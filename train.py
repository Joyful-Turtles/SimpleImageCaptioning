from time import time
import math
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, sampler
from dataset import COCODataset
from vocabulary import Vocabulary
from tqdm import tqdm
from model import Encoder, Decoder, ImageCaption
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image


def show_image_with_caption(image_tensor, caption_tensor, vocab):
    """
    이미지와 캡션을 시각화하는 함수.

    Args:
        image_tensor: torch.Tensor, 이미지 텐서 (shape: [3, H, W])
        caption_tensor: torch.Tensor, 캡션 텐서 (정수 인덱스)
        vocab: Vocabulary 객체, 정수 인덱스를 단어로 변환하는 데 사용
    """
    # 1. 이미지 텐서를 복원
    unnormalize = transforms.Compose([
        transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.ToPILImage()  # 텐서를 PIL 이미지로 변환
    ])
    image = unnormalize(image_tensor.cpu())  # GPU에서 CPU로 이동 후 복원

    # 2. 캡션 텍스트로 변환
    caption = [vocab.idx2word[idx.item()] for idx in caption_tensor if idx.item() in vocab.idx2word]
    caption_text = " ".join(caption)

    image.save("./test.png")
    with open("./out.txt", "w") as f:
        f.write(caption_text)


class RandomCaptionLengthSampler(sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.length_to_indices = {}
        for idx, length in enumerate(self.dataset.caption_lengths):
            if length not in self.length_to_indices:
                self.length_to_indices[length] = []
            self.length_to_indices[length].append(idx)

    def __iter__(self):
        lengths = list(self.length_to_indices.keys())

        # 무한 반복 대신 길이를 반복적으로 샘플링
        for random_length in np.random.permutation(lengths):
            all_indices = self.length_to_indices[random_length]

            # 배치 생성
            np.random.shuffle(all_indices)  # 인덱스를 무작위로 섞기
            for i in range(0, len(all_indices), self.batch_size):
                batch = all_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # 배치 크기를 만족할 때만 반환
                    yield batch

    def __len__(self):
        return sum(len(indices) // self.batch_size for indices in self.length_to_indices.values())


def train(device, epoch=10, batch_size=32):
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    transform_validation = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    train = COCODataset(transform_train,
                       annotations_path="/Users/dsparch/Workspace/Data/COCO/annotations-2/captions_train2014.json",
                       img_dir_path="/Users/dsparch/Workspace/Data/COCO/train2014",
                        batch_size=batch_size)

    validation = COCODataset(transform_validation,
                             annotations_path="/Users/dsparch/Workspace/Data/COCO/annotations-2/captions_val2014.json",
                             batch_size=batch_size,
                             img_dir_path="/Users/dsparch/Workspace/Data/COCO/val2014")


    data_loader = DataLoader(dataset=train,
                             batch_sampler=RandomCaptionLengthSampler(train, batch_size),
                             num_workers=4)

    validation_data_loader = DataLoader(dataset=validation,
                                        num_workers=4)

    embed_size = 256
    encoder = Encoder(embed_size)
    decoder = Decoder(embed_size, 512, len(train.vocab))
    model = ImageCaption(encoder, decoder)
    model.load_state_dict(torch.load("./epoch_4.pth"))
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(5, 5 + epoch + 1):
        total_loss = 0.0
        model.train()
        start = time()
        for images, captions in data_loader:
            #show_image_with_caption(images[0], captions[0], vocab=train.vocab)

            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions)
            loss = criterion(
                outputs.view(-1, len(train.vocab)),
                captions[:, 1:].contiguous().view(-1)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        print(f"TRAIN Total Loss for Epoch {epoch}: {total_loss:.4f} time: {time() - start}")
        torch.save(model.state_dict(), f"epoch_{epoch}.pth")

        validations_loss = 0.0
        model.eval()
        for images, captions in tqdm(validation_data_loader):
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions)
            loss = criterion(
                outputs.view(-1, len(train.vocab)),
                captions[:, 1:].contiguous().view(-1)
            )

            validations_loss += loss.item()
            #print(f"Epoch {epoch}, Validation Loss: {loss.item():.4f}")
        print(f"VALIDATION Epoch {epoch} Total loss {validations_loss}")


if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    train(device)
