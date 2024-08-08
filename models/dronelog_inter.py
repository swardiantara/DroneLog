import torch
import torch.nn as nn

# Define the model Class
class DroneLogInter(nn.Module):
    def __init__(self, bert_model, encoder_type, n_heads, num_layers, freeze_embedding, bidirectional, lstm_hidden_size, pooling_type,  exclude_cls_before, exclude_cls_after, num_classes_1, num_classes_2, normalize_logits, loss_fc):
        super(DroneLogInter, self).__init__()
        self.bert_model = bert_model
        self.exclude_cls_before: bool = exclude_cls_before
        self.exclude_cls_after: bool = exclude_cls_after
        self.loss_fc = loss_fc
        self.normalize_logits = normalize_logits
        self.encoder_type = encoder_type
        self.bidirectional = bidirectional
        self.lstm_hidden_size = lstm_hidden_size
        self.encoder = self.get_encoder(lstm_hidden_size, num_layers, n_heads, bidirectional)
        self.pooling_type = pooling_type
        self.pooling = self.get_pooling_layer()
        self.num_classes_1 = num_classes_1
        self.num_classes_2 = num_classes_2
        self.fc_1 = nn.Linear(bert_model.config.hidden_size, num_classes_1)
        self.fc_2 = self.get_fc_2()
        self.activation = self.get_activation_fc()
        if freeze_embedding:
            for param in self.bert_model.parameters():
                param.requires_grad = False
    
    
    def get_activation_fc(self):
        if self.loss_fc != 'logloss':
            return nn.Softmax(dim=1)
        else:
            return nn.Sigmoid()
    
    
    def get_fc_2(self):
        if self.num_classes_2 is not None:
            return nn.Linear(self.bert_model.config.hidden_size, self.num_classes_2)
        else:
            return None
                
                
    def get_encoder(self, lstm_hidden_size, num_layers, n_heads, bidirectional):
        if self.encoder_type == 'lstm':
            return nn.LSTM(
            input_size=self.bert_model.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        elif self.encoder_type == 'gru':
            return nn.GRU(
            input_size=self.bert_model.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        elif self.encoder_type == 'transformer':
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.bert_model.config.hidden_size,
                    nhead=n_heads,
                    dim_feedforward=2048,
                    batch_first=True,
                    dropout=0.1,
                ),
                num_layers=num_layers,
            )
        elif self.encoder_type == 'none':
            return None
        else:
            raise ValueError("Invalid encoder_type. Use 'lstm', 'gru', 'transformer', or 'none'.")
    
    
    def get_pooling_layer(self):
        if self.pooling_type == 'avg':
            return nn.AdaptiveAvgPool1d(1)
        elif self.pooling_type == 'max':
            return nn.AdaptiveMaxPool1d(1)
        elif self.pooling_type == 'cls':
            return None  # No pooling, use [CLS] token representation directly
        elif self.pooling_type == 'last':
            return None  # No pooling, use last token's representation directly
        else:
            raise ValueError("Invalid pooling_type. Use 'avg', 'max', 'cls' or 'last'.")

    def forward(self, input_ids, attention_mask):
        # Get the input embedding from BERT
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        # input_embedding = bert_output.last_hidden_state
        last_hidden_state = bert_output.last_hidden_state
        
        # Check if the [CLS] is included to be fed to the Encoder
        if self.exclude_cls_before:
            last_hidden_state = last_hidden_state[:, 1:, :]
        
        # Apply encoder if specified
        if self.encoder is not None:
            if self.encoder_type == 'transformer':
                # (bsz, seq_len, d_dim)
                last_hidden_state = self.encoder(last_hidden_state)
            else: # LSTM or GRU
                last_hidden_state, _ = self.encoder(last_hidden_state)
            
        if self.exclude_cls_after:
            # Exclude the [CLS] token after passing through the Encoder Layer
            last_hidden_state = last_hidden_state[:, 1:, :]
        
        if self.pooling_type == 'cls':
            # Use [CLS] token representation directly
            pooled_output = last_hidden_state[:, 0, :]
        elif self.pooling_type == 'last':
            # Only for LSTM and GRU
            if self.bidirectional:
                # Concatenate the last hidden states from both directions
                pooled_output = torch.cat((last_hidden_state[:, -1, :self.lstm_hidden_size], last_hidden_state[:, 0, self.lstm_hidden_size:]), dim=1)
            else:
                # Extract the last hidden state from the forward LSTM
                pooled_output = last_hidden_state[:, -1, :]
        else:
            # Perform pooling based on the specified technique (avg or max)
            pooled_output = self.pooling(last_hidden_state.permute(0, 2, 1)).squeeze(2)

        logits_1 = self.fc_1(pooled_output)
        if self.normalize_logits:
            logits_1 = self.activation(logits_1)
            
        if self.fc_2 is not None:
            logits_2 = self.fc_2(pooled_output)
            # logits_2 = self.activation(logits_2)
        else:
            logits_2 = None

        return logits_1
