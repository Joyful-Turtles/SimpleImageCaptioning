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

        hidden = torch.zeros(
            self.lstm.num_layers,
            batch_size,
            self.lstm.hidden_size,
        ).to(features.device)

        cell = torch.zeros(
            self.lstm.num_layers,
            batch_size,
            self.lstm.hidden_size,
        ).to(features.device)

        x = torch.zeros(batch_size, seq_length, self.fc.out_features).to(
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

    def predict(self, features, vocab, device, max_len=30):
        features = features.to(device)

        batch_size = features.size(0)

        hidden = torch.zeros(
            self.lstm.num_layers, batch_size, self.lstm.hidden_size
        ).to(device)
        cell = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(
            device
        )

        inputs = features.unsqueeze(1)

        generated_captions = [[] for _ in range(batch_size)]

        for t in range(max_len):
            outputs, (hidden, cell) = self.lstm(
                inputs, (hidden, cell)
            )  # outputs: (batch_size, 1, hidden_size)
            outputs = self.fc(outputs.squeeze(1))  # (batch_size, vocab_size)
            predicted_idx = torch.argmax(outputs, dim=1)  # (batch_size,)

            for i in range(batch_size):
                word_idx = predicted_idx[i].item()
                generated_captions[i].append(word_idx)

            # Prepare next inputs
            inputs = self.embed(predicted_idx).unsqueeze(
                1
            )  # (batch_size, 1, embed_size)

            # Check for <end> token to stop generation early
            end_mask = predicted_idx == vocab.word2idx.get("<end>", -1)
            if end_mask.any():
                for i in range(batch_size):
                    if end_mask[i] and len(generated_captions[i]) < max_len:
                        # Pad the rest of the caption with <end> tokens if desired
                        pass  # 또는 다른 로직을 추가할 수 있습니다.

        # Convert indices to words
        final_captions = []
        for caption in generated_captions:
            words = [vocab.idx2word.get(idx, "<unk>") for idx in caption]
            final_captions.append(words)

        return final_captions


class ImageCaption(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaption, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        features = self.encoder(images)
        x = self.decoder(features, captions)
        return x
