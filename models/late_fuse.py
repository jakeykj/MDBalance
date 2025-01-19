import torch
from torch import nn
import lightning as L

from .ehr_transformer import EHRTransformer
from torchvision.models import resnet50, ResNet50_Weights

from .base_fuse import BaseFuseTrainer


class LateFuse(BaseFuseTrainer):
    def __init__(self,hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.pred_criterion = nn.BCEWithLogitsLoss(reduction='none')

        # set class number
        self.num_classes = hparams['num_classes']

        # EHR transformer
        ehr_input_size = 24 #if self.hparams['new_data'] else 76
        self.ehr_model = EHRTransformer(input_size=ehr_input_size, num_classes=self.num_classes,
                                                    d_model=self.hparams.hidden_size, n_head=self.hparams.ehr_n_head,
                                                    n_layers_feat=1, n_layers_shared=1,
                                                    n_layers_distinct=self.hparams.ehr_n_layers,
                                                    dropout=self.hparams.ehr_dropout, simple=True)

        # CXR Encoder
        self.cxr_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.cxr_model.fc = nn.Linear(in_features=2048, out_features=self.hparams.hidden_size)

        # EHR and CXR heads
        self.ehr_head = nn.Linear(self.hparams.hidden_size, self.num_classes)
        self.cxr_head = nn.Linear(self.hparams.hidden_size, self.num_classes)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2*self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_size, self.num_classes)
        )


    def forward(self, data_dict):

        # EHR and CXR data
        x = data_dict['ehr_ts']
        img = data_dict['cxr_imgs'] # 不存在的图片按0填充
        seq_lengths = data_dict['seq_len']
        pairs = data_dict['has_cxr'] # 确定有cxr数据

        # Encoders
        feat_ehr, _ = self.ehr_model(x, seq_lengths)
        feat_cxr = self.cxr_model(img)

        # EHR modality
        out_ehr = self.ehr_head(feat_ehr)
        pred_ehr = out_ehr
        

        # CXR modality
        cxr_idx = (pairs==1).nonzero().squeeze(1)
        
        if cxr_idx.shape[0] > 0: # cxr exists
            out_cxr_idx = self.cxr_head(feat_cxr)[cxr_idx]
            pred_cxr_idx = out_cxr_idx
            pred_cxr = torch.zeros_like(pred_ehr)
            pred_cxr[cxr_idx] = pred_cxr_idx
        else:
            pred_cxr = torch.zeros_like(pred_ehr)
        
        # Fuse
        final_pred = self.fusion(torch.cat((feat_ehr, feat_cxr), dim=1))
            
        # only return the distinct features
        outputs = {
            'feat_ehr': feat_ehr,
            'feat_cxr': feat_cxr,
            'predictions': final_pred,
            'pred_ehr':pred_ehr,
            'pred_cxr': pred_cxr,
        }
        
        return outputs
        
        
    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch) # model forward

        pairs = batch['has_cxr']
        ehr_mask = torch.ones_like(out['feat_ehr'][:, 0])

        loss_ehr = self._compute_masked_pred_loss(out['pred_ehr'], batch['labels'], ehr_mask)
        loss_cxr = self._compute_masked_pred_loss(out['pred_cxr'],batch['labels'], pairs)
        loss_fuse = self._compute_masked_pred_loss(out['predictions'], batch['labels'], pairs)

        loss_total = loss_ehr + loss_cxr + loss_fuse

        self.log_dict({'loss/train': loss_total.detach()},
                      on_epoch=True, on_step=True,
                      batch_size=batch['labels'].shape[0])

        return loss_total

    def validation_step(self,batch,batch_idx):
        out = self._val_test_shared_step(batch, self.val_info)
        pairs = batch['has_cxr']
        ehr_mask = torch.ones_like(out['feat_ehr'][:, 0])

        loss_ehr = self._compute_masked_pred_loss(out['pred_ehr'], batch['labels'], ehr_mask)
        loss_cxr = self._compute_masked_pred_loss(out['pred_cxr'],batch['labels'], pairs)
        loss_fuse = self._compute_masked_pred_loss(out['predictions'], batch['labels'], pairs)

        loss_total = loss_ehr + loss_cxr + loss_fuse

        self.log_dict({'loss/val': loss_total.detach()},
                      on_epoch=True, on_step=True,
                      batch_size=batch['labels'].shape[0])

        return  loss_total        
    
    def _compute_masked_pred_loss(self, input, target, mask):
        return (self.pred_criterion(input, target).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)


    def _val_test_shared_step(self, batch, cache):
        out = self._shared_step(batch)

        cache['predictions'].append(out['predictions'].sigmoid())
        cache['pred_ehr'].append(out['pred_ehr'].sigmoid())
        cache['pred_cxr'].append(out['pred_cxr'].sigmoid())
        cache['labels'].append(batch['labels'])
        
        return out


