import torch
from torchvision import transforms
from model import Encoder, Decoder, ImageCaption
from vocabulary import Vocabulary
from PIL import Image


if __name__ == "__main__":
    vocab = Vocabulary(5, None)

    embed_size = 256
    encoder = Encoder(embed_size)
    decoder = Decoder(embed_size, 512, len(vocab))

    model = ImageCaption(encoder, decoder).to("mps")
    state = torch.load("./epoch_9.pth")
    model.load_state_dict(state)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    image = Image.open("/Users/dsparch/Workspace/Data/COCO/train2014/COCO_train2014_000000000009.jpg").convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    features = model.encoder(image_tensor.to("mps"))
    caption = model.decoder.predict(features, "<start>", "<end>", vocab, "mps")
    print(caption)

