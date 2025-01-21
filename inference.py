import torch
from torchvision import transforms
from model import Encoder, Decoder, ImageCaption
from vocabulary import Vocabulary
from PIL import Image

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backendsd.mps.is_available():
        device = "mps"

    vocab = Vocabulary(5, None)

    embed_size = 256
    encoder = Encoder(embed_size)
    decoder = Decoder(embed_size, 512, len(vocab), num_layers=3)

    model = ImageCaption(encoder, decoder).to(device)
    state = torch.load("./epoch_1.pth")
    model.load_state_dict(state)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    image = Image.open(
        "/home/dkdk/data/coco2014/coco2014/images/train2014/COCO_train2014_000000000081.jpg"
    ).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    features = model.encoder(image_tensor.to(device))
    caption = model.decoder.predict(features)[0]

    for i in caption:
        print(vocab.idx2word[i])

    """
    test_image = Image.open(
        "/home/dkdk/data/coco2014/coco2014/images/train2014/COCO_train2014_000000000081.jpg"
    ).convert("RGB")
    test_image_tensor = transform_validation(test_image).unsqueeze(0)
    test_features = model.encoder(test_image_tensor.to(device))

    data_loader.dataset.vocab.idx2word[i]
    for i in model.decoder.predict(test_features)[0]

    """
