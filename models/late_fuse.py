import torch
from torch import nn
import lightning as L

from .ehr_transformer import EHRTransformer, UniEHRTransformer
from .cxr_model import UniCXRModel
from torchvision.models import resnet50, ResNet50_Weights

from .base_fuse import BaseFuseTrainer


class LateFuse(BaseFuseTrainer):
    def __init__(self,hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.pred_criterion = nn.BCEWithLogitsLoss(reduction='none')

        # set class number
        self.num_classes = 25 if hparams['data']['task'] == 'phenotype' else 1

        # EHR transformer
        ehr_input_size = 24 #if self.hparams['new_data'] else 76
        ehr_hidden_size = hparams['ehr_model']['hidden_size']
        ehr_n_head = hparams['ehr_model']['n_head']
        ehr_n_layers = hparams['ehr_model']['n_layers']
        ehr_dropout = hparams['ehr_model']['dropout']

        self.ehr_model = EHRTransformer(input_size=ehr_input_size,
                                        num_classes=self.num_classes,
                                        d_model=ehr_hidden_size, 
                                        n_head=ehr_n_head,
                                        n_layers=ehr_n_layers,
                                        dropout=ehr_dropout)

        # CXR Encoder
        cxr_hidden_size = hparams['cxr_model']['hidden_size']
        self.cxr_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.cxr_model.fc = nn.Linear(in_features=2048, out_features=cxr_hidden_size)

        # EHR and CXR heads
        self.ehr_head = nn.Linear(ehr_hidden_size, self.num_classes)
        self.cxr_head = nn.Linear(cxr_hidden_size, self.num_classes)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(ehr_hidden_size+cxr_hidden_size, hparams['model']['hidden_size']),
            nn.ReLU(),
            nn.Dropout(hparams['model']['dropout']),
            nn.Linear(hparams['model']['hidden_size'], self.num_classes)
        )

        self.distill_criterion = nn.MSELoss(reduction='none')
        self.ehr_distill_loss_weight = hparams['model']['ehr_distill_loss_weight']
        self.cxr_distill_loss_weight = hparams['model']['cxr_distill_loss_weight']

        if hparams['model']['ehr_modal_distill']:
            self.ehr_teacher_model = UniEHRTransformer.load_from_checkpoint(hparams['ehr_model']['checkpoint_path'])
            self.ehr_teacher_model = self.ehr_teacher_model.ehr_model
            for param in self.ehr_teacher_model.parameters():
                param.requires_grad = False
        
        if hparams['model']['cxr_modal_distill']:
            self.cxr_teacher_model = UniCXRModel.load_from_checkpoint(hparams['cxr_model']['checkpoint_path'])
            self.cxr_teacher_model = self.cxr_teacher_model.cxr_model
            for param in self.cxr_teacher_model.parameters():
                param.requires_grad = False


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
            pred_cxr_idx = self.cxr_head(feat_cxr)[cxr_idx]
            pred_cxr = torch.zeros_like(pred_ehr)
            pred_cxr[cxr_idx] = pred_cxr_idx
        else:
            pred_cxr = torch.zeros_like(pred_ehr)
        
        # Fuse
        final_pred = self.fusion(torch.cat((feat_ehr, feat_cxr), dim=1))

        # Distill
        if self.hparams['model']['ehr_modal_distill']:
            feat_ehr_teacher, _ = self.ehr_teacher_model(x, seq_lengths)
        else:
            feat_ehr_teacher = None
        if self.hparams['model']['cxr_modal_distill']:
            feat_cxr_teacher = self.cxr_teacher_model(img)
        else:
            feat_cxr_teacher = None
        
            
        # only return the distinct features
        outputs = {
            'feat_ehr': feat_ehr,
            'feat_cxr': feat_cxr,
            'feat_ehr_teacher': feat_ehr_teacher,
            'feat_cxr_teacher': feat_cxr_teacher,
            'predictions': final_pred,
            'pred_ehr':pred_ehr,
            'pred_cxr': pred_cxr,
        }
        
        return outputs
        
        
    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch) # model forward

        pairs = batch['has_cxr']
        ehr_mask = torch.ones_like(out['feat_ehr'][:, 0])

        # loss_ehr = self._compute_masked_pred_loss(out['pred_ehr'], batch['labels'], ehr_mask)
        # loss_cxr = self._compute_masked_pred_loss(out['pred_cxr'],batch['labels'], pairs)
        loss_fuse = self._compute_masked_pred_loss(out['predictions'], batch['labels'], pairs)

        # loss_total = loss_ehr + loss_cxr + loss_fuse
        loss_total = loss_fuse

        # distill loss
        if self.hparams['model']['ehr_modal_distill']:  
            loss_ehr_distill = self._compute_masked_distill_loss(out['feat_ehr'], out['feat_ehr_teacher'].data, ehr_mask)
            loss_total = loss_total +  self.ehr_distill_loss_weight * loss_ehr_distill
        if self.hparams['model']['cxr_modal_distill']:
            loss_cxr_distill = self._compute_masked_distill_loss(out['feat_cxr'], out['feat_cxr_teacher'].data, pairs)
            loss_total = loss_total + self.cxr_distill_loss_weight * loss_cxr_distill


        self.log_dict({'loss/train': loss_total.detach(),
                       'loss/train_ehr_distill': loss_ehr_distill.detach(),
                       'loss/train_cxr_distill': loss_cxr_distill.detach()},
                      on_epoch=True, on_step=True,
                      batch_size=batch['labels'].shape[0])

        return loss_total

    def validation_step(self,batch,batch_idx):
        out = self._val_test_shared_step(batch, self.val_info)
        pairs = batch['has_cxr']
        ehr_mask = torch.ones_like(out['feat_ehr'][:, 0])

        # loss_ehr = self._compute_masked_pred_loss(out['pred_ehr'], batch['labels'], ehr_mask)
        # loss_cxr = self._compute_masked_pred_loss(out['pred_cxr'],batch['labels'], pairs)
        loss_fuse = self._compute_masked_pred_loss(out['predictions'], batch['labels'], pairs)

        # loss_total = loss_ehr + loss_cxr + loss_fuse
        loss_total = loss_fuse

        self.log_dict({'loss/val': loss_total.detach()},
                      on_epoch=True, on_step=True,
                      batch_size=batch['labels'].shape[0])

        return  loss_total        
    
    def _compute_masked_pred_loss(self, input, target, mask):
        return (self.pred_criterion(input, target).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)
    
    def _compute_masked_distill_loss(self, input, target, mask):
        return (self.distill_criterion(input, target).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)


    def _val_test_shared_step(self, batch, cache):
        out = self._shared_step(batch)

        cache['predictions'].append(out['predictions'].sigmoid())
        cache['pred_ehr'].append(out['pred_ehr'].sigmoid())
        cache['pred_cxr'].append(out['pred_cxr'].sigmoid())
        cache['labels'].append(batch['labels'])
        
        return out


