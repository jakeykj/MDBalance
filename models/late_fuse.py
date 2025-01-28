import torch
from torch import nn
import lightning as L
import numpy as np
import torch.nn.functional as F

from .ehr_transformer import EHRTransformer, UniEHRTransformer
from .cxr_model import UniCXRModel

from .base_fuse import BaseFuseTrainer


class LateFuse(BaseFuseTrainer):
    def __init__(self,hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.pred_criterion = nn.BCEWithLogitsLoss(reduction='none')

        # set class number
        self.class_names = hparams.get('class_names', None)
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
        self.cxr_model = UniCXRModel(hparams)

        # EHR and CXR heads
        # self.ehr_head = nn.Linear(ehr_hidden_size, self.num_classes)
        # self.cxr_head = nn.Linear(cxr_hidden_size, self.num_classes)

        # Fusion layer
        self.d_model = hparams['model']['hidden_size']
        self.dropout = hparams['model']['dropout']

        self.task_specific_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ehr_hidden_size+cxr_hidden_size, self.d_model),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.LayerNorm(self.d_model)
            ) for _ in range(self.num_classes)
        ])

        self.task_heads = nn.ModuleList([
            nn.Sequential(
                # nn.Linear(self.d_model, self.d_model),
                # nn.ReLU(),
                # nn.Dropout(self.dropout),
                # nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, 1)  # Binary output for each task
            ) for _ in range(self.num_classes)
        ])

        self.grad_clip_val = hparams['model'].get('grad_clip_val', 1.0)

        self.distill_criterion = nn.MSELoss(reduction='none')
        self.ada_loss_weight = hparams['model'].get('ada_loss_weight', False)

        self.ehr_teacher_model = UniEHRTransformer.load_from_checkpoint(hparams['ehr_chkp'])
        if self.ehr_teacher_model is not None:
            self.ehr_teacher_model = self.ehr_teacher_model.ehr_model
            for param in self.ehr_teacher_model.parameters():
                param.requires_grad = False
            self.ehr_teacher_model.eval()

        self.cxr_teacher_model = UniCXRModel.load_from_checkpoint(hparams['cxr_chkp'])
        if self.cxr_teacher_model is not None:
            for param in self.cxr_teacher_model.parameters():
                param.requires_grad = False
            self.cxr_teacher_model.eval()

        # Add tracking for loss weights
        self.register_buffer('loss_weights', torch.ones(3))  # [train_weight, distill_weight]

    def forward(self, data_dict):

        # EHR and CXR data
        x = data_dict['ehr_ts']
        img = data_dict['cxr_imgs'] # 不存在的图片按0填充
        seq_lengths = data_dict['seq_len']
        pairs = data_dict['has_cxr'] # 确定有cxr数据

        # Encoders
        feat_ehr, _ = self.ehr_model(x, seq_lengths)
        feat_ehr = torch.stack(feat_ehr, dim=1)  # Shape: [batch_size, num_tasks, hidden_size]
        
        out_cxr = self.cxr_model({'cxr_imgs': img})
        feat_cxr = out_cxr['feat_cxr']
        
        # Fuse
        fused_feats = []
        fused_preds = []
        for task_idx in range(self.num_classes):
            fused_label_feat = torch.cat((feat_ehr[:, task_idx], feat_cxr[:, task_idx]), dim=1)
            task_repr = self.task_specific_layers[task_idx](fused_label_feat)
            task_output = self.task_heads[task_idx](task_repr)
            fused_feats.append(task_repr)
            fused_preds.append(task_output)
        fused_preds = torch.cat(fused_preds, dim=1)  # Shape: [batch_size, num_tasks]
        fused_feats = torch.stack(fused_feats, dim=1)  # Shape: [batch_size, num_tasks, hidden_size]


        # Distill
        if self.ehr_teacher_model is not None:
            feat_ehr_teacher, pred_ehr_teacher = self.ehr_teacher_model(x, seq_lengths)
            feat_ehr_teacher = torch.stack(feat_ehr_teacher, dim=1)
        else:
            feat_ehr_teacher = None
            pred_ehr_teacher = None

        if self.cxr_teacher_model is not None:
            out_cxr_teacher = self.cxr_teacher_model({'cxr_imgs': img})
            feat_cxr_teacher = out_cxr_teacher['feat_cxr']
            pred_cxr_teacher = out_cxr_teacher['pred_cxr']
        else:
            feat_cxr_teacher = None
            pred_cxr_teacher = None
        
        if self.ehr_teacher_model is not None:
            preds_with_ehr_teachers = []
            with torch.no_grad():
                for i in range(self.num_classes):
                    fused_label_feat = torch.cat((feat_ehr_teacher[:, i], feat_cxr[:, i]), dim=1)
                    task_repr = self.task_specific_layers[i](fused_label_feat)
                    task_output = self.task_heads[i](task_repr)
                    preds_with_ehr_teachers.append(task_output)
                preds_with_ehr_teachers = torch.cat(preds_with_ehr_teachers, dim=1)  # Shape: [batch_size, num_tasks]
        else:
            preds_with_ehr_teachers = None
        
        if self.cxr_teacher_model is not None:
            preds_with_cxr_teachers = []
            with torch.no_grad():
                for i in range(self.num_classes):
                    fused_label_feat = torch.cat((feat_ehr[:, i], feat_cxr_teacher[:, i]), dim=1)
                    task_repr = self.task_specific_layers[i](fused_label_feat)
                    task_output = self.task_heads[i](task_repr)
                    preds_with_cxr_teachers.append(task_output)
                preds_with_cxr_teachers = torch.cat(preds_with_cxr_teachers, dim=1)  # Shape: [batch_size, num_tasks]
        else:
            preds_with_cxr_teachers = None

        # only return the distinct features
        outputs = {
            'feat_ehr': feat_ehr,
            'feat_cxr': feat_cxr,
            'feat_ehr_teacher': feat_ehr_teacher,
            'feat_cxr_teacher': feat_cxr_teacher,
            'pred_ehr_teacher': pred_ehr_teacher,
            'pred_cxr_teacher': pred_cxr_teacher,
            'predictions': fused_preds,
            'predictions_with_ehr_teachers': preds_with_ehr_teachers,
            'predictions_with_cxr_teachers': preds_with_cxr_teachers,
        }
        
        return outputs

    def compute_loss_weights(self, grads):
        # Before stacking, normalize each gradient and reshape to same dimension
        normalized_grads = []
        for g in grads:
            # Flatten the gradients first
            g_flat = g.view(-1)
            # Normalize
            g_norm = g_flat / (g_flat.norm() + 1e-8)
            normalized_grads.append(g_norm)
        
        # Pad shorter gradients to match the longest one
        max_length = max(g.numel() for g in normalized_grads)
        padded_grads = []
        for g in normalized_grads:
            if g.numel() < max_length:
                padding = torch.zeros(max_length - g.numel(), device=g.device)
                g_padded = torch.cat([g, padding])
            else:
                g_padded = g
            padded_grads.append(g_padded)
        
        # Now stack the padded gradients
        grads = torch.stack(padded_grads)
        
        # Compute Gram matrix
        G = torch.mm(grads, grads.t())
        
        # Solve min_a a^T G a s.t. a >= 0, sum(a) = 1
        n_tasks = len(grads)
        a = torch.ones(n_tasks, device=grads.device) / n_tasks
        
        for _ in range(20):  # Usually converges in a few iterations
            grad_f = torch.mv(G, a)
            min_idx = torch.argmin(grad_f)
            v = torch.zeros_like(a)
            v[min_idx] = 1.0
            gamma = 2.0 / (2.0 + _)
            a = (1.0 - gamma) * a + gamma * v
        
        # Rescale weights based on gradient magnitudes
        weights = a / grads.norm(dim=1)
        weights = weights / weights.sum()
        return weights
    
    def get_ada_kd_weight(self, gamma):
        # tanh
        if self.hparams['model']['ada_kd_weight_type'] == 'tanh':
            alpha = 1.0
            weight = torch.exp(-alpha * (gamma - 1).clamp(min=0))

        # sigmoid
        elif self.hparams['model']['ada_kd_weight_type'] == 'sigmoid':
            alpha, beta = 5.0, 0.5
            weight = beta + (1-beta) / (1 + torch.exp(alpha * (gamma - 1)))

        # power law
        elif self.hparams['model']['ada_kd_weight_type'] == 'power_law':
            alpha = self.hparams['model']['ada_kd_alpha']
            weight = 1 / (gamma ** alpha)

        # soft threshold
        elif self.hparams['model']['ada_kd_weight_type'] == 'soft_threshold':
            threshold = 1.2
            slope = 5.0
            weight = torch.relu(threshold - gamma) * slope + 0.1

        return weight

    def training_step(self, batch, batch_idx):

        # Enable gradient tracking for computing per-task gradients
        self.zero_grad()
        out = self._shared_step(batch)

        # Compute prediction loss (size: [batch_size, num_classes])
        loss_fuse = self.pred_criterion(out['predictions'], batch['labels']).mean(dim=1)

        # Compute distillation loss if enabled
        if self.ehr_teacher_model is not None:
            loss_ehr_distill = self.distill_criterion(out['feat_ehr'], out['feat_ehr_teacher'].data)
        else:
            loss_ehr_distill = torch.tensor(0.0, device=loss_fuse.device)

        if self.cxr_teacher_model is not None:
            loss_cxr_distill = self.distill_criterion(out['feat_cxr'], out['feat_cxr_teacher'].data)
        else:
            loss_cxr_distill = torch.tensor(0.0, device=loss_fuse.device)

        # Compute individual weight for fusion loss and distillation loss, for each individual, measure the BCE loss
        # of the prediction produced by the teacher model and the fusion model. If the fusion model is better, the weight
        # of the distillation loss should be lower since fusion is beneficial for this individual.
        # Get teacher model predictions
        if self.ada_loss_weight:
            # with torch.no_grad():
                # if self.hparams['model']['ehr_modal_distill']:
                #     ehr_teacher_loss = self.pred_criterion(out['pred_ehr_teacher'], batch['labels']).mean(dim=1)
                # else:
                #     ehr_teacher_loss = torch.zeros_like(loss_fuse)
                    
                # if self.hparams['model']['cxr_modal_distill']:
                #     cxr_teacher_loss = self.pred_criterion(out['pred_cxr_teacher'], batch['labels']).mean(dim=1)
                # else:
                #     cxr_teacher_loss = torch.zeros_like(loss_fuse)

            # Compare fusion model performance using scores
            with torch.no_grad():
                labels = batch['labels']  # [batch_size, num_classes]

                p_fuse = torch.sigmoid(out['predictions'])
                s_fuse = labels * p_fuse + (1 - labels) * (1 - p_fuse)
                
                
                # Compute scores: y*p + (1-y)*(1-p)
                # s_fuse = s_fuse.mean(dim=1)  # Average across classes
                
                if self.ehr_teacher_model is not None:
                    p_ehr = torch.sigmoid(out['predictions_with_ehr_teachers']) 
                    s_ehr = labels * p_ehr + (1 - labels) * (1 - p_ehr)

                    gamma_ehr = s_fuse / (s_ehr + 1e-8)  # Shape: [batch_size, num_classes]
                    confidence_ehr = (torch.abs(p_fuse - 0.5) / (torch.abs(p_ehr - 0.5) + 1e-8))
                    ehr_distill_weight = torch.where(
                        gamma_ehr <= 1,
                        torch.ones_like(gamma_ehr),
                        self.get_ada_kd_weight(gamma_ehr)  #*self.get_ada_kd_weight(confidence_ehr)
                    )
                else:
                    ehr_distill_weight = torch.ones_like(out['predictions'])

                if self.hparams['model']['cxr_modal_distill']:
                    p_cxr = torch.sigmoid(out['predictions_with_cxr_teachers'])
                    s_cxr = labels * p_cxr + (1 - labels) * (1 - p_cxr)
                    confidence_cxr = (torch.abs(p_fuse - 0.5) / (torch.abs(p_cxr - 0.5) + 1e-8))

                    gamma_cxr = s_fuse / (s_cxr + 1e-8)
                    cxr_distill_weight = torch.where(
                        gamma_cxr <= 1,
                        torch.ones_like(gamma_cxr),
                        self.get_ada_kd_weight(gamma_cxr)  #*self.get_ada_kd_weight(confidence_cxr)
                    )
                else:
                    cxr_distill_weight = torch.ones_like(out['predictions'])

        else:
            gamma_ehr = torch.zeros_like(out['predictions'])
            gamma_cxr = torch.zeros_like(out['predictions'])
            ehr_distill_weight = torch.ones_like(out['predictions'])
            cxr_distill_weight = torch.ones_like(out['predictions'])

        # Apply weights to distillation losses
        loss_ehr_distill = (loss_ehr_distill * ehr_distill_weight.unsqueeze(2)).sum() / (ehr_distill_weight.sum()) / loss_ehr_distill.shape[2]
        loss_cxr_distill = (loss_cxr_distill * cxr_distill_weight.unsqueeze(2)).sum() / (cxr_distill_weight.sum()) / loss_cxr_distill.shape[2]
        loss_fuse = loss_fuse.mean()


        # Perform MGDA: find optimal loss weights using the Frank-Wolfe algorithm
        if self.ehr_teacher_model is not None and self.cxr_teacher_model is not None:
            losses = [loss_fuse]
            if self.ehr_teacher_model is not None:
                losses.append(loss_ehr_distill)
            if self.cxr_teacher_model is not None:
                losses.append(loss_cxr_distill)
            grads = []
            for loss in losses:
                if torch.is_tensor(loss) and loss.requires_grad:
                    self.zero_grad()
                    loss.backward(retain_graph=True)
                    grad = []
                    for param in self.parameters():
                        if param.grad is not None:
                            grad.append(param.grad.view(-1))
                    grad = torch.cat(grad)
                    grads.append(grad)
                else:
                    grads.append(torch.zeros_like(grads[0]) if grads else torch.zeros(1))

            # Compute optimal loss weights
            self.loss_weights = self.compute_loss_weights(grads)
            
            # Compute weighted total loss
            loss_total = (self.loss_weights[0] * loss_fuse + 
                         self.loss_weights[1] * (loss_ehr_distill) +
                         self.loss_weights[2] * (loss_cxr_distill))
        else:
            loss_total = loss_fuse

        # Log metrics
        log_dict = {
            'loss/train': loss_total.detach(),
            'loss/train_weight': self.loss_weights[0],
            'loss/distill_weight': self.loss_weights[1],
            'ada_kd_weight/gamma_ehr': gamma_ehr.float().mean(),
            'ada_kd_weight/gamma_cxr': gamma_cxr.float().mean(),
            'ada_kd_weight/ratio_fusion_better_than_ehr': (gamma_ehr>1).float().mean(),
            'ada_kd_weight/ratio_fusion_better_than_cxr': (gamma_cxr>1).float().mean(),
        }
        for i, class_name in enumerate(self.class_names):
            log_dict[f'EHR_dynamic_dist_weight/{class_name}'] = ehr_distill_weight[:, i].mean()
            log_dict[f'CXR_dynamic_dist_weight/{class_name}'] = cxr_distill_weight[:, i].mean()
        self.log_dict(log_dict, on_epoch=True, on_step=True, batch_size=batch['labels'].shape[0])
        
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
    
    def _compute_masked_pred_loss(self, input, target, mask, sample_weight=None):
        loss = self.pred_criterion(input, target).mean(dim=1)
        if sample_weight is not None:
            loss = loss * sample_weight
        return (loss * mask).sum() / max(mask.sum(), 1e-6)
    
    def _compute_masked_distill_loss(self, input, target, mask):
        return (self.distill_criterion(input, target).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)


    def _val_test_shared_step(self, batch, cache):
        out = self._shared_step(batch)

        cache['predictions'].append(out['predictions'].sigmoid())
        # cache['pred_ehr'].append(out['pred_ehr'].sigmoid())
        # cache['pred_cxr'].append(out['pred_cxr'].sigmoid())
        cache['labels'].append(batch['labels'])
        
        return out


