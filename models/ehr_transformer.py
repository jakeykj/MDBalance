import torch
from torch import nn
from .base_fuse import BaseFuseTrainer

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.rand(1, max_len, d_model))
        self.pe.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # x: (batch_size, seq_len, embedding_dim)
        return self.dropout(x)


class EHRTransformer(nn.Module):
    def __init__(self, input_size, num_classes,
                 d_model=256, n_head=8, n_layers=2,
                 dropout=0.3, max_len=350):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Input embedding
        self.emb = nn.Linear(input_size, d_model)
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Feature extraction layers
        feat_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            batch_first=True, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(feat_layer, num_layers=n_layers)
        self.feat_norm = nn.LayerNorm(d_model)
    
        
        # self.fc = nn.Linear(d_model, num_classes)
        # Enhanced MLP head for classification
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x, seq_lengths):
        # Create attention mask
        attn_mask = torch.stack([torch.cat([torch.zeros(len_, device=x.device),
                                float('-inf')*torch.ones(max(seq_lengths) - len_, device=x.device)])
                                for len_ in seq_lengths])
        
        # Input processing with normalization and dropout
        x = self.emb(x)
        # x = self.input_dropout(x)
        x = self.input_norm(x)
        x = self.pos_encoder(x)
        
        # Feature extraction
        feat = self.transformer_encoder(x, src_key_padding_mask=attn_mask)
        feat = self.feat_norm(feat)
        feat = self.dropout(feat)
        
        # Create padding mask for aggregation
        padding_mask = torch.ones_like(attn_mask).unsqueeze(2)
        padding_mask[attn_mask==float('-inf')] = 0
        
        ehr_representation = (padding_mask * feat).sum(dim=1) / padding_mask.sum(dim=1)
        output = self.fc(ehr_representation)
        return ehr_representation, output


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, num_classes,
                 d_model=256, n_head=8, n_layers=2,
                 dropout=0.3, max_len=350):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.emb = nn.Linear(input_size, d_model)
        # self.emb = nn.Embedding(num_tokens, d_model, padding_idx=num_tokens)
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=0, max_len=max_len)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, seq_lengths, output_prob=False):
        attn_mask = torch.stack([torch.cat([torch.zeros(len_, device=x.device),
                                 float('-inf')*torch.ones(max(seq_lengths)-len_, device=x.device)])
                                for len_ in seq_lengths])
        x = self.emb(x) # * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        feat = self.encoder(x, src_key_padding_mask=attn_mask)
        feat = self.encoder(feat, src_key_padding_mask=attn_mask)

        padding_mask = torch.ones_like(attn_mask).unsqueeze(2)
        padding_mask[attn_mask==float('-inf')] = 0
        feat = (padding_mask * feat).sum(dim=1) / padding_mask.sum(dim=1)

        prediction = self.fc(feat)

        if output_prob:
            prediction = prediction.sigmoid()

        return feat, prediction


class UniEHRTransformer(BaseFuseTrainer):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.num_classes = 25 if self.hparams.data.task == 'phenotype' else 1
        self.hidden_size = self.hparams['ehr_model']['hidden_size']
        self.n_head = self.hparams['ehr_model']['n_head']
        self.n_layers = self.hparams['ehr_model']['n_layers']
        self.dropout = self.hparams['ehr_model']['dropout']

        self.ehr_model = EHRTransformer(input_size=24, num_classes=self.num_classes,
                                        d_model=self.hidden_size, n_head=self.n_head,
                                        n_layers=self.n_layers, dropout=self.dropout)

        self.pred_criterion = nn.BCEWithLogitsLoss()

    def forward(self, data_dict):
        x = data_dict['ehr_ts']
        seq_lengths = data_dict['seq_len']
        feat, predictions = self.ehr_model(x, seq_lengths)
        predictions = predictions

        # get shared feature
        loss = self.pred_criterion(predictions, data_dict['labels'])
        #loss = sigmoid_focal_loss(predictions, data_dict['labels'], reduction='mean')
        outputs = {
            'loss': loss,
            'predictions': predictions.sigmoid()
        }
        return outputs
    
    def on_validation_epoch_end(self):
        scores = self._val_test_epoch_end(self.val_info, clear_cache=True)
        scores['step'] = float(self.current_epoch)
        self.log_dict({k: v for k, v in scores.items() if not isinstance(v, list)}, on_epoch=True, on_step=False)
        
        return scores
    
    def on_test_epoch_end(self):
        scores = self._val_test_epoch_end(self.test_info, clear_cache=True)
        self.test_results = {x: scores[x] for x in scores}