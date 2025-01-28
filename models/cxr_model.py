import torch
import torch.nn as nn
import lightning as L
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import average_precision_score, roc_auc_score

from .base_fuse import BaseFuseTrainer


class UniCXRModel(BaseFuseTrainer):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.num_classes = 25 if self.hparams.data.task == 'phenotype' else 1
        
        # Initialize ResNet50 backbone with pretrained weights
        self.cxr_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Replace the final classification layer
        cxr_hidden_size = self.hparams.cxr_model['hidden_size']
        self.cxr_model.fc = nn.Linear(in_features=2048, out_features=cxr_hidden_size)

        self.dropout = self.hparams.cxr_model.get('dropout', 0.2)
        self.task_specific_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cxr_hidden_size, cxr_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.LayerNorm(cxr_hidden_size)
            ) for _ in range(self.num_classes)
        ])
        
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                # nn.Linear(cxr_hidden_size, cxr_hidden_size),
                # nn.ReLU(),
                # nn.Dropout(self.dropout),
                # nn.LayerNorm(cxr_hidden_size),
                nn.Linear(cxr_hidden_size, 1)  # Binary output for each task
            ) for _ in range(self.num_classes)
        ])

        self.pred_criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, data_dict):

        img = data_dict['cxr_imgs']

        # global feature
        feat_cxr = self.cxr_model(img)

        task_representations = []
        task_outputs = []
        for task_idx in range(self.num_classes):
            task_repr = self.task_specific_layers[task_idx](feat_cxr)
            task_output = self.task_heads[task_idx](task_repr)
            
            task_representations.append(task_repr)
            task_outputs.append(task_output)
        task_outputs = torch.cat(task_outputs, dim=1)  # Shape: [batch_size, num_tasks]
        task_feats = torch.stack(task_representations, dim=1)  # Shape: [batch_size, num_tasks, hidden_size]

        # task-specific features

        outputs = {
            'feat_cxr': task_feats,
            'pred_cxr': task_outputs,
        }
        
        return outputs

    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch)
        pairs = batch['has_cxr']
        loss = self._compute_masked_pred_loss(out['pred_cxr'], batch['labels'], pairs)
        
        self.log_dict({'loss/train': loss.detach()},
                      on_epoch=True, on_step=True,
                      batch_size=batch['labels'].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        out = self._val_test_shared_step(batch, self.val_info)
        pairs = batch['has_cxr']
        loss = self._compute_masked_pred_loss(out['pred_cxr'], batch['labels'], pairs)

        self.log_dict({'loss/val': loss.detach()},
                      on_epoch=True, on_step=True,
                      batch_size=batch['labels'].shape[0])
        
        return loss

    def _compute_masked_pred_loss(self, input, target, mask):
        return (self.pred_criterion(input, target).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)
    
    def _val_test_shared_step(self, batch, cache):
        out = self._shared_step(batch)

        cache['predictions'].append(out['pred_cxr'].sigmoid())
        cache['labels'].append(batch['labels'])
        
        return out