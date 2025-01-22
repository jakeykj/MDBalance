import torch
from torch import nn
import lightning as L
import numpy as np
import torch.nn.functional as F

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

        self.mixup_alpha = hparams['model'].get('mixup_alpha', 0.2)
        self.cutmix_alpha = hparams['model'].get('cutmix_alpha', 1.0)
        self.mix_prob = hparams['model'].get('mix_prob', 0.5)
        self.lambda_min = hparams['model'].get('lambda_min', 0.3)

        self.grad_clip_val = hparams['model'].get('grad_clip_val', 1.0)

        self.distill_criterion = nn.MSELoss(reduction='none')
        self.ehr_distill_loss_weight = hparams['model']['ehr_distill_loss_weight']
        self.cxr_distill_loss_weight = hparams['model']['cxr_distill_loss_weight']
        self.ada_loss_weight = hparams['model'].get('ada_loss_weight', False)

        if hparams['model']['ehr_modal_distill']:
            checkpoint_path = hparams['ehr_model']['checkpoint_path'][hparams['data']['task']]
            self.ehr_teacher_model = UniEHRTransformer.load_from_checkpoint(checkpoint_path, map_location='cpu')
            self.ehr_teacher_model = self.ehr_teacher_model.ehr_model
            
            for param in self.ehr_teacher_model.parameters():
                param.requires_grad = False
            self.ehr_teacher_model.eval()
        
        if hparams['model']['cxr_modal_distill']:
            checkpoint_path = hparams['cxr_model']['checkpoint_path'][hparams['data']['task']]
            cxr_teacher_model = UniCXRModel.load_from_checkpoint(checkpoint_path, map_location='cpu')
            
            self.cxr_teacher_model = cxr_teacher_model.cxr_model
            for param in self.cxr_teacher_model.parameters():
                param.requires_grad = False
            self.cxr_teacher_model.eval()

            self.cxr_teacher_head = cxr_teacher_model.cxr_head
            for param in self.cxr_teacher_head.parameters():
                param.requires_grad = False
            self.cxr_teacher_head.eval()

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
            feat_ehr_teacher, pred_ehr_teacher = self.ehr_teacher_model(x, seq_lengths)
        else:
            feat_ehr_teacher = None
        if self.hparams['model']['cxr_modal_distill']:
            feat_cxr_teacher = self.cxr_teacher_model(img)
            pred_cxr_teacher = self.cxr_teacher_head(feat_cxr_teacher)
        else:
            feat_cxr_teacher = None
        
            
        # only return the distinct features
        outputs = {
            'feat_ehr': feat_ehr,
            'feat_cxr': feat_cxr,
            'feat_ehr_teacher': feat_ehr_teacher,
            'feat_cxr_teacher': feat_cxr_teacher,
            'pred_ehr_teacher': pred_ehr_teacher,
            'pred_cxr_teacher': pred_cxr_teacher,
            'predictions': final_pred,
            'pred_ehr':pred_ehr,
            'pred_cxr': pred_cxr,
        }
        
        return outputs
        
    def _mix_data(self, data_dict, mixing_type='mixup'):
        """
        Apply mixup or cutmix to both EHR and CXR data
        Returns mixed data and mixing information
        """
        batch_size = data_dict['ehr_ts'].size(0)
        
        # Generate random mixing ratio
        if mixing_type == 'mixup':
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:  # cutmix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        lam = max(lam, self.lambda_min)
        index = torch.randperm(batch_size, device=data_dict['ehr_ts'].device)
        
        mixed_data = {}
        
        # Mix EHR data
        x_ehr = data_dict['ehr_ts']
        if mixing_type == 'mixup':
            mixed_data['ehr_ts'] = lam * x_ehr + (1 - lam) * x_ehr[index]
        else:  # For EHR, we'll do a temporal cutmix
            temp_cut_point = int(x_ehr.size(1) * lam)
            mixed_data['ehr_ts'] = x_ehr.clone()
            mixed_data['ehr_ts'][:, temp_cut_point:] = x_ehr[index, temp_cut_point:]
        
        # Mix CXR images
        x_cxr = data_dict['cxr_imgs']
        if mixing_type == 'mixup':
            mixed_data['cxr_imgs'] = lam * x_cxr + (1 - lam) * x_cxr[index]
        else:  # cutmix
            # Generate random box
            W, H = x_cxr.size(2), x_cxr.size(3)
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
            
            mixed_data['cxr_imgs'] = x_cxr.clone()
            mixed_data['cxr_imgs'][:, :, bbx1:bbx2, bby1:bby2] = \
                x_cxr[index, :, bbx1:bbx2, bby1:bby2]
            
            # Adjust lambda based on actual box size
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        # Handle other necessary data
        mixed_data['seq_len'] = data_dict['seq_len']  # Keep original sequence lengths
        mixed_data['has_cxr'] = data_dict['has_cxr']  # Keep original CXR availability flags
        
        # Mix labels - important for training!
        mixed_data['labels'] = data_dict['labels']  # Original labels needed for loss computation
        mixed_data['mixed_labels'] = data_dict['labels'][index]  # Mixed labels needed for loss computation
        mixed_data['lambda'] = lam  # Save mixing ratio for loss computation
        
        return mixed_data

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
            alpha = 2.0
            weight = 1 / (gamma ** alpha)

        # soft threshold
        elif self.hparams['model']['ada_kd_weight_type'] == 'soft_threshold':
            threshold = 1.2
            slope = 5.0
            weight = torch.relu(threshold - gamma) * slope + 0.1

        return weight

    def training_step(self, batch, batch_idx):
        # Decide whether to apply mixing and which type
        do_mixing = self.hparams['model']['enable_mixing'] and np.random.random() < self.mix_prob
        if do_mixing:
            mixing_type = np.random.choice(['mixup', 'cutmix'])
            mixed_batch = self._mix_data(batch, mixing_type)
            lam = mixed_batch['lambda']
        else:
            lam = 1.0

        # Enable gradient tracking for computing per-task gradients
        self.zero_grad()
        out = self._shared_step(batch)
        pairs = batch['has_cxr']

        # Compute prediction loss (size: [batch_size, num_classes])
        loss_fuse = self.pred_criterion(out['predictions'], batch['labels']).mean(dim=1)
        if do_mixing:
            loss_fuse_mixed = self.pred_criterion(out['predictions'], mixed_batch['mixed_labels']).mean(dim=1)
            loss_fuse = lam * loss_fuse + (1 - lam) * loss_fuse_mixed

        # Compute distillation loss if enabled
        if self.hparams['model']['ehr_modal_distill']:
            loss_ehr_distill = self.distill_criterion(out['feat_ehr'], out['feat_ehr_teacher'].data).mean(dim=1)
        else:
            loss_ehr_distill = torch.tensor(0.0, device=loss_fuse.device)

        if self.hparams['model']['cxr_modal_distill']:
            loss_cxr_distill = self.distill_criterion(out['feat_cxr'], out['feat_cxr_teacher'].data).mean(dim=1)
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
                # Convert logits to probabilities
                p_fuse = torch.sigmoid(out['predictions'])
                p_ehr = torch.sigmoid(out['pred_ehr_teacher']) if self.hparams['model']['ehr_modal_distill'] else None
                p_cxr = torch.sigmoid(out['pred_cxr_teacher']) if self.hparams['model']['cxr_modal_distill'] else None
                
                labels = batch['labels']
                
                # Compute scores: y*p + (1-y)*(1-p)
                s_fuse = labels * p_fuse + (1 - labels) * (1 - p_fuse)
                s_fuse = s_fuse.mean(dim=1)  # Average across classes
                
                if self.hparams['model']['ehr_modal_distill']:
                    s_ehr = labels * p_ehr + (1 - labels) * (1 - p_ehr)
                    s_ehr = s_ehr.mean(dim=1)
                    # Compute gamma and weights
                    gamma_ehr = s_fuse / (s_ehr + 1e-8)  # Add epsilon to prevent division by zero
                    ehr_distill_weight = torch.where(
                        gamma_ehr <= 1,
                        torch.ones_like(gamma_ehr),
                        self.get_ada_kd_weight(gamma_ehr)
                    )
                else:
                    ehr_distill_weight = torch.ones_like(loss_ehr_distill)

                if self.hparams['model']['cxr_modal_distill']:
                    s_cxr = labels * p_cxr + (1 - labels) * (1 - p_cxr)
                    s_cxr = s_cxr.mean(dim=1)
                    # Compute gamma and weights
                    gamma_cxr = s_fuse / (s_cxr + 1e-8)
                    cxr_distill_weight = torch.where(
                        gamma_cxr <= 1,
                        torch.ones_like(gamma_cxr),
                        self.get_ada_kd_weight(gamma_cxr)
                    )
                else:
                    cxr_distill_weight = torch.ones_like(loss_cxr_distill)

        else:
            gamma_ehr = torch.zeros_like(loss_ehr_distill)
            gamma_cxr = torch.zeros_like(loss_cxr_distill)
            ehr_distill_weight = torch.ones_like(loss_ehr_distill)
            cxr_distill_weight = torch.ones_like(loss_cxr_distill)

        # Apply weights to distillation losses
        loss_ehr_distill = (loss_ehr_distill * ehr_distill_weight).sum() / (ehr_distill_weight.sum())
        loss_cxr_distill = (loss_cxr_distill * cxr_distill_weight).sum() / (cxr_distill_weight.sum())
        loss_fuse = loss_fuse.mean()


        # Perform MGDA: find optimal loss weights using the Frank-Wolfe algorithm
        if self.hparams['model']['ehr_modal_distill'] or self.hparams['model']['cxr_modal_distill']:
            losses = [loss_fuse, loss_ehr_distill, loss_cxr_distill]
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
        self.log_dict({
            'loss/train': loss_total.detach(),
            'loss/train_weight': self.loss_weights[0],
            'loss/distill_weight': self.loss_weights[1],
            'ada_kd_weight/gamma_ehr': gamma_ehr.float().mean(),
            'ada_kd_weight/gamma_cxr': gamma_cxr.float().mean(),
            'ada_kd_weight/ratio_fusion_better_than_ehr': (gamma_ehr>1).float().mean(),
            'ada_kd_weight/ratio_fusion_better_than_cxr': (gamma_cxr>1).float().mean(),
            'mix_lambda': torch.tensor(lam),
            'mix_type': torch.tensor(1.0 if mixing_type == 'mixup' else 2.0) if do_mixing else torch.tensor(0.0),
        }, on_epoch=True, on_step=True, batch_size=batch['labels'].shape[0])
        
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
        cache['pred_ehr'].append(out['pred_ehr'].sigmoid())
        cache['pred_cxr'].append(out['pred_cxr'].sigmoid())
        cache['labels'].append(batch['labels'])
        
        return out


