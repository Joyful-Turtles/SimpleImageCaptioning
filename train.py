import torch
import numpy as np
import torchvision.transforms as transforms

from time import time
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm
from model import Encoder, Decoder, ImageCaption
from dataset import COCODataset


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


fix_seed(0)


def show_image_with_caption(image_tensor, caption_tensor, vocab):
    """
    이미지와 캡션을 시각화하는 함수.

    Args:
        image_tensor: torch.Tensor, 이미지 텐서 (shape: [3, H, W])
        caption_tensor: torch.Tensor, 캡션 텐서 (정수 인덱스)
        vocab: Vocabulary 객체, 정수 인덱스를 단어로 변환하는 데 사용
    """
    # 1. 이미지 텐서를 복원
    unnormalize = transforms.Compose(
        [
            transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
            transforms.ToPILImage(),  # 텐서를 PIL 이미지로 변환
        ]
    )
    image = unnormalize(image_tensor.cpu())  # GPU에서 CPU로 이동 후 복원

    # 2. 캡션 텍스트로 변환
    caption = [
        vocab.idx2word[idx.item()]
        for idx in caption_tensor
        if idx.item() in vocab.idx2word
    ]
    caption_text = " ".join(caption)

    image.save("./test.png")
    with open("./out.txt", "w") as f:
        f.write(caption_text)


class RandomCaptionLengthSampler(sampler.Sampler):
    def __init__(self, dataset, batch_size):
        """
        샘플러 초기화 메서드
        :param dataset: 데이터셋 객체 (caption_lengths 속성 필요)
        :param batch_size: 배치 크기
        """
        self.dataset = dataset
        self.batch_size = batch_size

        # 캡션 길이에 따라 인덱스를 그룹화
        self.length_to_indices = {}
        for idx, length in enumerate(self.dataset.caption_lengths):
            if length not in self.length_to_indices:
                self.length_to_indices[length] = []
            self.length_to_indices[length].append(idx)

        # 캡션 길이를 무작위 순서로 섞습니다.
        self.random_lengths = list(self.length_to_indices.keys())
        np.random.shuffle(self.random_lengths)

    def __iter__(self):
        """샘플러의 반복자를 정의"""
        for random_length in self.random_lengths:
            all_indices = self.length_to_indices[random_length]
            np.random.shuffle(all_indices)

            # 배치 단위로 인덱스 묶기
            for i in range(0, len(all_indices), self.batch_size):
                batch = all_indices[i : i + self.batch_size]
                if len(batch) == self.batch_size:  # 배치 크기를 만족할 때만 반환
                    yield batch

    def __len__(self):
        return sum(
            len(indices) // self.batch_size
            for indices in self.length_to_indices.values()
        )


def train(device, epoch=10, batch_size=256):
    transform_train = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    transform_validation = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # 모든 이미지를 256x256으로 리사이즈
            transforms.CenterCrop(224),  # 224x224로 중앙 크롭
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train = COCODataset(
        transform_train,
        annotations_path="/home/dkdk/data/coco2014/coco2014/annotations/annotations/captions_train2014.json",
        img_dir_path="/home/dkdk/data/coco2014/coco2014/images/train2014",
        batch_size=batch_size,
    )

    validation = COCODataset(
        transform_validation,
        annotations_path="/home/dkdk/data/coco2014/coco2014/annotations/annotations/captions_val2014.json",
        img_dir_path="/home/dkdk/data/coco2014/coco2014/images/val2014",
        batch_size=batch_size,
    )

    data_loader = DataLoader(
        dataset=train,
        batch_sampler=RandomCaptionLengthSampler(train, batch_size),
        num_workers=4,
        pin_memory=True,
    )

    validation_data_loader = DataLoader(
        dataset=validation,
        batch_sampler=RandomCaptionLengthSampler(validation, batch_size),
        num_workers=4,
        pin_memory=True,
    )

    embed_size = 512
    encoder = Encoder(embed_size)
    decoder = Decoder(embed_size, 512, len(train.vocab), num_layers=1)
    model = ImageCaption(encoder, decoder)
    # model.load_state_dict(torch.load("./epoch_1.pth"))
    model = model.to(device)

    # Ref: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(1, epoch + 1):
        total_loss = 0.0
        model.train()
        start = time()

        for images, captions in tqdm(data_loader):
            # show_image_with_caption(images[0], captions[0], vocab=train.vocab)

            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions)

            # Origin)
            # outputs = outputs.view(-1, len(train.vocab))
            # captions = captions.contiguous().view(-1)

            # New)
            outputs = outputs
            captions = captions
            # outputs: [B, word_len, vocab_size] -> [B, vocab_size, word_len]
            # captions: [B, word_len]
            outputs = outputs.permute(0, 2, 1)

            loss = criterion(outputs, captions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            word_len = captions.shape[1]
            print(
                f"\rEpoch {epoch}, Loss: {loss.item():.4f}, word_len: {word_len}",
            )

            # First image debug
            outputs = outputs.permute(0, 2, 1)
            outputs = outputs[0]
            outputs = outputs.argmax(dim=1)
            logits = outputs.cpu().detach().numpy()
            print(f"logits: {logits.shape}")

            predictions = ""
            for logit in logits:
                predictions += train.vocab.idx2word[logit] + " "
            print(f" = RET-1 : {len(logits)} : {predictions}")

            # Caption debug
            captions_str = ""
            for caption in captions[0]:
                captions_str += train.vocab.idx2word[caption.item()] + " "
            print(f" = CAPTI : {len(captions[0])} : {captions_str}")

            # First image debug
            features = model.encoder(images)
            # [B, 256] -> [1, 256] first image
            features = features[0].unsqueeze(0)
            caption = model.decoder.predict(features, train.vocab, device)
            caption_join = " ".join(caption[0])
            print(f" = RET-2 : {len(caption[0])} : {caption_join}")

        print(
            f"TRAIN Total Loss for Epoch {epoch}: {total_loss:.4f} time: {time() - start}"
        )
        torch.save(model.state_dict(), f"epoch_{epoch}.pth")

        if epoch % 5 != 0:
            print(f"[INFO] skip validation for epoch {epoch}")
            continue

        validations_loss = 0.0
        model.eval()
        for images, captions in tqdm(validation_data_loader):
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions)

            # Origin)
            # outputs = outputs.view(-1, len(train.vocab))
            # captions = captions.contiguous().view(-1)

            # New)
            outputs = outputs
            captions = captions
            # outputs: [B, word_len, vocab_size] -> [B, vocab_size, word_len]
            # captions: [B, word_len]
            outputs = outputs.permute(0, 2, 1)

            loss = criterion(outputs, captions)

            validations_loss += loss.item()
            print(f"Epoch {epoch}, Validation Loss: {loss.item():.4f}")
        print(f"VALIDATION Epoch {epoch} Total loss {validations_loss}")


if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    train(device)
