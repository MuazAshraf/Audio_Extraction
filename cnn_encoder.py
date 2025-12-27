import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    """
    CNN Encoder to extract features from Mel Spectrograms.
    Input: (Batch, Channels, Freq, Time) -> e.g., (B, 1, 128, T)
    Output: (Batch, Time', Features)
    """
    def __init__(self, in_channels=1, base_channels=32):
        super(EncoderCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=(1, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(base_channels*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flat_feature_dim = 16 * (base_channels * 4)
        self.fc = nn.Linear(self.flat_feature_dim, 512)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.permute(0, 3, 1, 2)

        batch, time, ch, freq = x.size()
        x = x.reshape(batch, time, ch * freq)

        x = self.dropout(self.fc(x))
        return x


class Attention(nn.Module):
    """
    Bahdanau Attention mechanism
    """
    def __init__(self, hidden_dim, encoder_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (Batch, Hidden_Dim)
        # encoder_outputs: (Batch, Time, Encoder_Dim)

        time_len = encoder_outputs.size(1)

        hidden_expanded = hidden.unsqueeze(1).repeat(1, time_len, 1)

        combined = torch.cat((hidden_expanded, encoder_outputs), dim=2)
        energy = torch.tanh(self.attn(combined))
        attention = self.v(energy).squeeze(2)

        weights = F.softmax(attention, dim=1)

        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, weights


class DecoderGRU(nn.Module):
    """
    GRU Decoder with Attention
    """
    def __init__(self, vocab_size=29, embed_dim=256, hidden_dim=512, encoder_dim=512):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim + encoder_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_char, hidden, context):
        # input_char: (Batch,)
        # hidden: (1, Batch, 512)
        # context: (Batch, 512)

        embedded = self.embedding(input_char)

        rnn_input = torch.cat([embedded, context], dim=1)
        rnn_input = rnn_input.unsqueeze(1)

        output, hidden = self.gru(rnn_input, hidden)

        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden


class Seq2SeqASR(nn.Module):
    def __init__(self, vocab_size=29):
        super().__init__()
        self.encoder = EncoderCNN()
        self.attention = Attention(hidden_dim=512, encoder_dim=512)
        self.decoder = DecoderGRU(vocab_size, hidden_dim=512, encoder_dim=512)
        # Initialize decoder hidden from encoder
        self.init_hidden = nn.Linear(512, 512)

    def forward(self, spectrogram, target_text, teacher_forcing=0.5):
        # Encode
        encoder_output = self.encoder(spectrogram)  # (B, T, 512)

        # Initialize hidden from encoder (use mean of encoder outputs)
        batch_size = spectrogram.size(0)
        encoder_mean = encoder_output.mean(dim=1)  # (B, 512)
        hidden = torch.tanh(self.init_hidden(encoder_mean)).unsqueeze(0)  # (1, B, 512)
        input_char = target_text[:, 0]  # <sos>

        outputs = []
        for t in range(1, target_text.size(1)):
            
            # Attention
            context, weights = self.attention(hidden[-1], encoder_output)

            # Decode
            prediction, hidden = self.decoder(input_char, hidden, context)
            outputs.append(prediction)

            # Teacher forcing
            if torch.rand(1) < teacher_forcing:
                input_char = target_text[:, t]
            else:
                input_char = prediction.argmax(1)

        return torch.stack(outputs, dim=1)
