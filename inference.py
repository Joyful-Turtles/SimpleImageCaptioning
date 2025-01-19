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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    image = Image.open(
        "/home/dkdk/data/coco2014/coco2014/images/val2014/COCO_val2014_000000000074.jpg"
    ).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    features = model.encoder(image_tensor.to(device))
    print(f"features: {features.shape}")
    caption = model.decoder.predict(features, vocab, device)
    print(caption)
