import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_encoder(input_size, hidden_size, emb_size, layers=1, rnn_type='GRU', bidirectional=False):
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.embedding = nn.Embedding(input_size, emb_size)
            if rnn_type == 'GRU':
                self.rnn = nn.GRU(emb_size, hidden_size, layers, batch_first=True, dropout=0.0, bidirectional=bidirectional)
            elif rnn_type == 'LSTM':
                self.rnn = nn.LSTM(emb_size, hidden_size, layers, batch_first=True, dropout=0.0, bidirectional=bidirectional)
            self.rnn_type = rnn_type

        def forward(self, inputs):
            embedded = self.embedding(inputs)
            if self.rnn_type == 'GRU':
                outputs, hidden = self.rnn(embedded)
            elif self.rnn_type == 'LSTM':
                outputs, (hidden, cell) = self.rnn(embedded)
                hidden = (hidden, cell)
            return outputs, hidden
    
    return Encoder()

def create_decoder(output_size, hidden_size, emb_size, layers=1, rnn_type='GRU', bidirectional=False):
    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.embedding = nn.Embedding(output_size, emb_size)
            if rnn_type == 'GRU':
                self.rnn = nn.GRU(emb_size, hidden_size, layers, batch_first=True, dropout=0.0, bidirectional=bidirectional)
            elif rnn_type == 'LSTM':
                self.rnn = nn.LSTM(emb_size, hidden_size, layers, batch_first=True, dropout=0.0, bidirectional=bidirectional)
            self.fc = nn.Linear(hidden_size, output_size)
            self.rnn_type = rnn_type

        def forward(self, inputs, hidden):
            inputs = inputs.unsqueeze(1)
            embedded = self.embedding(inputs)
            if self.rnn_type == 'GRU':
                outputs, hidden = self.rnn(embedded, hidden)
            elif self.rnn_type == 'LSTM':
                outputs, hidden = self.rnn(embedded, hidden)
            output = self.fc(outputs.squeeze(1))
            return output, hidden
    
    return Decoder()

def create_seq2seq_model(encoder, decoder, device, teacher_forcing_ratio=0.5):
    class Seq2Seq(nn.Module):
        def __init__(self):
            super(Seq2Seq, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.device = device
            self.teacher_forcing_ratio = teacher_forcing_ratio

        def forward(self, src, trg, trg_vocab):
            src = src.to(self.device)
            trg = trg.to(self.device)
            batch_size = src.size(0)
            seq_length = trg.size(1)
            output_size = len(trg_vocab)
            outputs = torch.zeros(batch_size, seq_length, output_size).to(self.device)
            _, hidden = self.encoder(src)
            input = trg[:, 0]
            
            for t in range(1, seq_length):
                output, hidden = self.decoder(input, hidden)
                outputs[:, t, :] = output
                top1 = output.argmax(1)
                input = trg[:, t] if torch.rand(1).item() < self.teacher_forcing_ratio else top1
            
            return outputs
    
    return Seq2Seq()

def train_model(encoder, decoder, model, optimizer, criterion, xtrain, y_traindecoder, num_epochs=10, batch_size=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    losses = []
    
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        epoch_loss = 0
        batches = batchgen(xtrain, y_traindecoder, batch_size)
        for src, trg in batches:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg, tel_voc)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(batches)
        losses.append(avg_loss)
        print(f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}')
    
    return losses

def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = len(eng_voc)  
output_size = len(tel_voc)
hidden_size = 256
embedding_size = 256
num_layers = 2
rnn_type = 'LSTM'  # 'GRU' or 'LSTM'
bidirectional = False

encoder = create_encoder(input_size, hidden_size, embedding_size, num_layers, rnn_type, bidirectional)
decoder = create_decoder(output_size, hidden_size, embedding_size, num_layers, rnn_type, bidirectional)
model = create_seq2seq_model(encoder, decoder, device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=tel_voc['<pad>'])

losses = train_model(encoder, decoder, model, optimizer, criterion, xtrain, y_traindecoder, num_epochs=10, batch_size=50)
plot_loss(losses)