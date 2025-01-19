import torch
import torchvision


class Encoder(torch.nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)

        modules = list(resnet50.children())[:-1]
        self.resnet = torch.nn.Sequential(*modules)
        self.linear = torch.nn.Linear(resnet50.fc.in_features, embed_size)

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
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        batch_size = features.size(0)
        seq_length = captions.size(1)
        # batch x seq_length x embed_size
        captions_embed = self.embed(captions)

        hidden = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(features.device)
        cell = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(features.device)

        x = torch.zeros(batch_size, seq_length, self.fc.out_features).to(features.device)

        # Teacher Forcing instead Autoregressive
        for t in range(captions.size(1)):
            if t == 0:
                inputs = features.unsqueeze(1)
            else:
                inputs = captions_embed[:, t, :].unsqueeze(1)

            outputs, (hidden, cell) = self.lstm(inputs, (hidden, cell))
            x[:, t, :] = self.fc(outputs.squeeze(1))

        return x

    def predict(self, features, vocab, device, max_len=20):
        features = features.to(device)

        batch_size = features.size(0)
        hidden = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        cell = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)

        inputs = features.unsqueeze(1)
        generated_caption = []
        for t in range(max_len):
            outputs, (hidden, cell) = self.lstm(inputs, (hidden, cell))
            outputs = self.fc(outputs.squeeze(1))
            predicted_idx = torch.argmax(outputs, dim=1)
            generated_caption.append(predicted_idx.item())
            inputs = self.embed(predicted_idx).unsqueeze(1)

        return [vocab.idx2word[idx] for idx in generated_caption]


class ImageCaption(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaption, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        features = self.encoder(images)
        x = self.decoder(features, captions)
        return x
