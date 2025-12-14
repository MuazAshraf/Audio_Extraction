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
        
        # Convolutional layers
        # Stride 2 in time reduces the time dimension size
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=(1, 1), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces Freq and Time by 2
        
        self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=(1, 1), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces Freq and Time by 2 again (Total /4)
        
        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=(1, 1), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces Freq and Time by 2 again (Total /8)
        
        # Calculate output feature size
        # If Mel Freq is 128: 128 -> 64 -> 32 -> 16
        # Final channels: base_channels*4
        # Flattened feature size = 16 * (base_channels*4)
        self.flat_feature_dim = 16 * (base_channels * 4)
        
        # Linear projection to RNN size or just keep as is
        self.fc = nn.Linear(self.flat_feature_dim, 512) 
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (Batch, 1, 128, Time)
        
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x))) 
        # Shape: (Batch, Channels, Freq', Time')
        
        # Rearrange to (Batch, Time', Channels, Freq') for processing as sequence
        x = x.permute(0, 3, 1, 2) 
        # Shape: (Batch, Time', Channels, Freq')
        
        batch, time, ch, freq = x.size()
        x = x.reshape(batch, time, ch * freq)
        # Shape: (Batch, Time', Flattened_Features)
        
        x = self.dropout(self.fc(x))
        return x

class Attention(nn.Module):
    """
    Bahdanau Attention (Additive) or Luong (Multiplicative).
    This implementation uses a simple combined mechanism.
    """
    def __init__(self, hidden_dim, encoder_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (Batch, Hidden_Dim) - from decoder
        # encoder_outputs: (Batch, Time, Encoder_Dim)
        
        batch_size = encoder_outputs.size(0)
        time_len = encoder_outputs.size(1)
        
        # Repeat hidden state for each time step
        # (Batch, Time, Hidden_Dim)
        hidden_expanded = hidden.unsqueeze(1).repeat(1, time_len, 1)
        
        # Calculate energy: Score(hidden, encoder_output)
        # (Batch, Time, Hidden + Encoder)
        combined = torch.cat((hidden_expanded, encoder_outputs), dim=2)
        
        # (Batch, Time, Hidden) -> (Batch, Time, 1)
        energy = torch.tanh(self.attn(combined))
        attention = self.v(energy).squeeze(2)
        
        # Weights over time
        weights = F.softmax(attention, dim=1)
        # (Batch, Time)
        
        # Context vector: Weighted sum of encoder outputs
        # (Batch, 1, Time) @ (Batch, Time, Encoder_Dim) -> (Batch, 1, Encoder_Dim)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        
        return context.squeeze(1), weights

class DecoderGRU(nn.Module):
    """
    GRU Decoder with Attention.
    """
    def __init__(self, output_dim, emb_dim, hidden_dim, encoder_dim, dropout=0.3):
        super(DecoderGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = Attention(hidden_dim, encoder_dim)
        
        self.gru = nn.GRU(emb_dim + encoder_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + hidden_dim + encoder_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_word, hidden, encoder_outputs):
        # input_word: (Batch) - index of previous word
        # hidden: (1, Batch, Hidden)
        # encoder_outputs: (Batch, Time, Encoder_Dim)
        
        input_word = input_word.unsqueeze(1) # (Batch, 1)
        embedded = self.dropout(self.embedding(input_word)) # (Batch, 1, Emb)
        
        # Calculate attention (Using last layer of hidden state for attention query)
        # hidden[-1] shape is (Batch, Hidden)
        context, weights = self.attention(hidden[-1], encoder_outputs)
        # Context: (Batch, Encoder_Dim)
        
        # Combine Embedding + Context for GRU Input
        # Note: We repeat context for the sequence length (which is 1 here)
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        # (Batch, 1, Emb + Encoder_Dim)
        
        # Pass through GRU
        output, hidden = self.gru(rnn_input, hidden)
        
        # Prediction: Combine Embedding, Output(Hidden), and Context
        # (Batch, Emb + Hidden + Encoder)
        prediction_input = torch.cat((embedded.squeeze(1), output.squeeze(1), context), dim=1)
        
        prediction = self.fc_out(prediction_input)
        # (Batch, Vocabulary_Size)
        
        return prediction, hidden, weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (Batch, 1, 128, Time)
        # trg: (Batch, Trg_Len) - Text indices
        
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encode audio features
        encoder_outputs = self.encoder(src)
        
        # Initial hidden state for decoder (can be zero or mapped)
        # Here we just use zero or learnable, but for simplicity let's initialize zero
        hidden = torch.zeros(1, batch_size, self.decoder.hidden_dim).to(self.device)
        
        # First input to the decoder is the <sos> token
        input_token = trg[:, 0]
        
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and encoder context
            # receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input_token, hidden, encoder_outputs)
            
            # place predictions in a tensor holding predictions for each token
            outputs[:, t, :] = output
            
            # decide if we are going to use teacher forcing or not
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input_token = trg[:, t] if teacher_force else top1
            
        return outputs
