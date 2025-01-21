import sys
import numpy as np
import torch
import torchvision.transforms as transforms
from time import time
from torch.utils.data import DataLoader, sampler
from dataset import COCODataset
from tqdm import tqdm
from model import Encoder, Decoder, ImageCaption


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


fix_seed(0)


def seed_fix(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


seed_fix(0)


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

        # TODO: Remove
        # self.random_lengths = [10, 10]
        print(f"self.random_lengths : {self.random_lengths}")

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


def train(device, epoch=3, batch_size=256):
    transform_train = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    transform_validation = transforms.Compose(
        [
            transforms.Resize((224, 224)),
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
        batch_size=batch_size,
        img_dir_path="/home/dkdk/data/coco2014/coco2014/images/val2014",
    )

    data_loader = DataLoader(
        dataset=train,
        batch_sampler=RandomCaptionLengthSampler(train, batch_size),
        num_workers=4,
    )

    validation_data_loader = DataLoader(
        dataset=validation,
        batch_sampler=RandomCaptionLengthSampler(validation, batch_size),
        num_workers=4,
    )

    embed_size = 512
    encoder = Encoder(embed_size)
    decoder = Decoder(embed_size, 512, len(train.vocab), num_layers=1)
    model = ImageCaption(encoder, decoder)
    # model.load_state_dict(torch.load("./epoch_1.pth"))
    model = model.to(device)

    # Ref: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

    test_image = Image.open(
        "/home/dkdk/data/coco2014/coco2014/images/train2014/COCO_train2014_000000000081.jpg"
    ).convert("RGB")
    test_image_tensor = transform_validation(test_image).unsqueeze(0)

    for epoch in range(1, epoch + 1):
        total_loss = 0.0
        model.train()
        start = time()
        for i, (images, captions) in enumerate(data_loader):
            # show_image_with_caption(images[0], captions[0], vocab=train.vocab)

            images = images.to(device)
            captions = captions.to(device)

            model.zero_grad()

            print(f"[DEBUG] captions shape: {captions.shape}")

            outputs = model(images, captions[:, :-1])

            loss = criterion(
                outputs.view(-1, len(train.vocab)),
                captions[:, 1:].contiguous().view(-1),
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            temp = " ".join(
                [
                    data_loader.dataset.vocab.idx2word[i]
                    for i in captions[0, 1:].cpu().tolist()
                ]
            )
            if "big plane" in temp:
                print(temp)
                exit()

            print(
                f"\rEpoch {epoch}, {i}/{len(data_loader)} Loss: {loss.item():.4f}",
                flush=True,
            )
            sys.stdout.flush()

            if i % 50 == 0:
                features = model.encoder(images)
                print(
                    "\rpred : ",
                    " ".join(
                        [
                            data_loader.dataset.vocab.idx2word[i]
                            for i in model.decoder.predict(features)[0]
                        ]
                    ),
                    flush=True,
                )
                print(
                    "\rmodel : ",
                    " ".join(
                        [
                            data_loader.dataset.vocab.idx2word[i]
                            for i in outputs[0].argmax(dim=1).cpu().tolist()
                        ]
                    ),
                    flush=True,
                )
                print(
                    "\ranswer : ",
                    " ".join(
                        [
                            data_loader.dataset.vocab.idx2word[i]
                            for i in captions[0, 1:].cpu().tolist()
                        ]
                    ),
                    flush=True,
                )
                with torch.no_grad():
                    model.encoder.eval()
                    test_features = model.encoder(test_image_tensor.to(device))
                    model.encoder.train()

                    model.decoder.eval()
                    print(
                        "\rtest : ",
                        " ".join(
                            [
                                data_loader.dataset.vocab.idx2word[i]
                                for i in model.decoder.predict(test_features)[0]
                            ]
                        ),
                        flush=True,
                    )
                    model.decoder.train()
                print()
                print(time() - start)
                start = time()

            # sys.stdout.flush()

        print(
            f"\nTRAIN Total Loss for Epoch {epoch}: {total_loss:.4f} time: {time() - start}"
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

            outputs = model(images, captions[:, :-1])
            loss = criterion(
                outputs.view(-1, len(train.vocab)),
                captions[:, 1:].contiguous().view(-1),
            )

            validations_loss += loss.item()
            # print(f"Epoch {epoch}, Validation Loss: {loss.item():.4f}")
        print(f"VALIDATION Epoch {epoch} Total loss {validations_loss}")


if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    train(device)
