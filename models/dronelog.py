import torch
import torch.nn as nn

# Define the model Class
class DroneLog(nn.Module):
    def __init__(self, bert_model, encoder_type, n_heads, num_layers, freeze_embedding, bidirectional, lstm_hidden_size, pooling, num_classes_multiclass):
        super(DroneLog, self).__init__()
        self.bert_model = bert_model
        self.encoder_type = encoder_type
        self.pooling = pooling
        self.num_classes_multiclass = num_classes_multiclass
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=bert_model.config.hidden_size,
                nhead=n_heads,
                dim_feedforward=2048,
                dropout=0.1,
            ),
            num_layers=num_layers,
        )
        self.lstm = nn.LSTM(
            input_size=bert_model.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.gru = nn.GRU(
            input_size=bert_model.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc_multiclass = nn.Linear(bert_model.config.hidden_size, num_classes_multiclass)
        if freeze_embedding:
            for param in self.bert_model.parameters():
                param.requires_grad = False
            for param in self.sbert_model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        input_embedding = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.encoder_type == 'transformer':
            transformer_output = self.transformer_encoder(input_embedding.permute(
                1, 0, 2), src_key_padding_mask=(1 - attention_mask).bool())
            transformer_output = transformer_output.permute(
                1, 0, 2)  # Back to [batch_size, seq_len, hid_dim]
            if self.pooling == 'cls':
                # take the BERT's [CLS] representation (batch_size x hid_dim)
                pooled_output = transformer_output[:, 0, :]
            elif self.pooling == 'min':
                # min pooling the second logits to the end
                pooled_output = torch.min(
                    transformer_output[:, 1:, :], dim=1).values
            elif self.pooling == 'mean':
                # mean pooling the second logits to the end
                pooled_output = transformer_output[:, 1:, :].mean(dim=1)
            elif self.pooling == 'max':
                # max pooling the second logits to the end
                pooled_output = torch.max(
                    transformer_output[:, 1:, :], dim=1).values

        elif self.encoder_type == 'lstm':
            lstm_output, _ = self.lstm(input_embedding)
            # Extract the last hidden state from the LSTM
            pooled_output = lstm_output[:, -1, :]
        elif self.encoder_type == 'gru':
            gru_output, _ = self.gru(input_embedding)
            # Extract the last hidden state from the GRU
            pooled_output = gru_output[:, -1, :]
        else:
            # Take the representation from the BERT's last hidden state
            if self.pooling == 'cls':
                # take the BERT's [CLS] representation
                pooled_output = input_embedding[:, 0, :]
            elif self.pooling == 'min':
                # min pooling from the second logits to the end
                pooled_output = torch.min(
                    input_embedding[:, 1:, :], dim=1).values
            elif self.pooling == 'mean':
                # mean pooling from the second logits to the end
                pooled_output = input_embedding[:, 1:, :].mean(dim=1)
            elif self.pooling == 'max':
                # max pooling from the second logits to the end
                pooled_output = torch.max(
                    input_embedding[:, 1:, :], dim=1).values
        
        logits_multiclass = self.fc_multiclass(pooled_output)

        return logits_multiclass
