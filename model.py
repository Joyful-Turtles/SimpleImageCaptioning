import torch
import torchvision


class Encoder(torch.nn.Module):
    def __init__(self, embed_size, backbone="resnet50"):
        super(Encoder, self).__init__()

        backbone = backbone.lower()
        if backbone == "resnet18":
            backbone = torchvision.models.resnet18(pretrained=True)
        elif backbone == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=True)
        else:
            backbone = torchvision.models.resnet18(pretrained=True)

        modules = list(backbone.children())[:-1]
        self.resnet = torch.nn.Sequential(*modules)
        self.linear = torch.nn.Linear(backbone.fc.in_features, embed_size)

    def forward(self, images):
        with torch.no_grad():
            x = self.resnet(images)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


class Decoder(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers

        self.embed = torch.nn.Embedding(vocab_size, embed_size)
        self.lstm = torch.nn.LSTM(
            embed_size, hidden_size, num_layers, batch_first=True, dropout=0.5
        )
        self.fc = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """forward
        :param features: image features from encoder
        :param captions: captions
        """
        batch_size = features.size(0)
        seq_length = captions.size(1)

        # batch x seq_length x embed_size
        captions_embed = self.embed(captions)

        hidden = torch.zeros((self.num_layers, batch_size, self.lstm.hidden_size)).to(
            features.device
        )
        cell = torch.zeros((self.num_layers, batch_size, self.lstm.hidden_size)).to(
            features.device
        )

        x = torch.zeros((batch_size, seq_length, self.fc.out_features)).to(
            features.device
        )

        # Teacher Forcing instead Autoregressive
        for t in range(captions.size(1)):
            if t == 0:
                inputs = features.unsqueeze(1)
            else:
                inputs = captions_embed[:, t, :].unsqueeze(1)

            outputs, (hidden, cell) = self.lstm(inputs, (hidden, cell))
            x[:, t, :] = self.fc(outputs.squeeze(1))

        return x

    def predict(self, features, max_len=20):
        with torch.no_grad():
            batch_size = features.size(0)
            hidden = torch.zeros(
                self.lstm.num_layers, batch_size, self.lstm.hidden_size
            ).to(features.device)
            cell = torch.zeros(
                self.lstm.num_layers, batch_size, self.lstm.hidden_size
            ).to(features.device)

            inputs = features.unsqueeze(1)
            generated_captions = (
                torch.zeros(batch_size, max_len).long().to(features.device)
            )

            for t in range(0, max_len):
                outputs, (hidden, cell) = self.lstm(inputs, (hidden, cell))
                logits = self.fc(outputs.squeeze(1))
                predicted = logits.argmax(dim=1)
                inputs = self.embed(predicted).unsqueeze(1)
                generated_captions[:, t] = predicted

        return generated_captions.cpu().tolist()


class ImageCaption(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaption, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        features = self.encoder(images)
        x = self.decoder(features, captions)
        return x
