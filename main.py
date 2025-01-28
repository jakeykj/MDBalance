import os
import pickle
import shutil
import yaml
import time
import argparse
from copy import deepcopy
from argparse import Namespace
import numpy as np
from torch import nn
import random
import torch
import lightning as L
from sklearn.metrics import average_precision_score, roc_auc_score

import hydra
from omegaconf import DictConfig, OmegaConf


from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from models.late_fuse import LateFuse
from models.ehr_transformer import UniEHRTransformer
from models.cxr_model import UniCXRModel

from utils import create_data_loaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_num_threads(5)
    L.seed_everything(seed, workers=True)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, cfg, log_dir, train_loader, val_loader, stage):
    set_seed(cfg.seed)

    ############# Configure training and logging #############
    callback_metric = 'overall/PRAUC'
    early_stop_callback = EarlyStopping(monitor=callback_metric,
                                        min_delta=0.00,
                                        patience=cfg.training.patience,
                                        verbose=False,
                                        mode="max")

    # file name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, version=stage, name='')
    csv_logger = pl_loggers.CSVLogger(save_dir=log_dir, version=stage, name='')

    # save the best model in the valid
    checkpoint_callback = ModelCheckpoint(
        auto_insert_metric_name=False,
        monitor=callback_metric,
        mode='max',
        save_top_k=1,
        verbose=True,
        filename='epoch{epoch:02d}-PRAUC={overall/PRAUC:.2f}'
    )

    trainer = L.Trainer(enable_checkpointing=True,
                        accelerator='gpu',
                        devices=[cfg.gpu],
                        fast_dev_run=20 if cfg.dev_run else False,
                        logger=[tb_logger, csv_logger],
                        num_sanity_val_steps=0,
                        max_epochs=cfg.training.num_epochs,
                        log_every_n_steps=1,
                        callbacks=[early_stop_callback, checkpoint_callback])
    
    ############# Train model #############

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # after training, restore the best model
    best_model_path = checkpoint_callback.best_model_path
    model = model.__class__.load_from_checkpoint(best_model_path, strict=False)

    out = {
        'trainer': trainer,
        'model': model,
        'best_model_path': best_model_path,
        'log_dir': csv_logger.log_dir,
    }

    return out


@hydra.main(config_path='configs', config_name='default', version_base=None)
def train_fusion_model(cfg: DictConfig):
    set_seed(cfg.seed)
    
    ############# Prepare data loaders #############
    train_loader, val_loader, test_loader = create_data_loaders(ehr_data_dir=cfg.data.ehr_root,
                                                                cxr_data_dir=None,
                                                                task=cfg.data.task,
                                                                replication=cfg.data.fold,
                                                                batch_size=cfg.training.batch_size,
                                                                num_workers=cfg.training.num_workers,
                                                                matched_subset=cfg.data.matched,
                                                                seed=cfg.seed,
                                                                pkl_dir=cfg.data.pkl_dir,
                                                                resized_base_path=cfg.data.resized_cxr_root,
                                                                image_meta_path=cfg.data.image_meta_path)

    log_dir = f'./experiments/{cfg.data.task}/{cfg.name}-seed{cfg.seed}-{time.strftime("%Y%m%d%H%M%S")}'

    
    ############# Stage 1(a): Uni-EHR training, skip if distillation is not enabled #############
    if cfg.model.ehr_modal_distill:
        if not cfg.retrain_teachers and os.path.exists(cfg.ehr_chkp):
            uniehr_teacher_model = UniEHRTransformer.load_from_checkpoint(cfg.ehr_chkp, map_location='cpu')
            print(f"Loaded UniEHR model from {cfg.ehr_chkp}")
        else:
            print('Training UniEHR model from scratch. If the checkpoint already exists, it will be overwritten.')
            uniehr_teacher_model = UniEHRTransformer(cfg)
            
            out = train_model(model=uniehr_teacher_model, cfg=cfg, log_dir=log_dir,
                              train_loader=train_loader, val_loader=val_loader, stage='uniehr')
            uniehr_teacher_model = out['model']
            shutil.copy(out['best_model_path'], cfg.ehr_chkp)
                
    else:
        uniehr_teacher_model = None
    
    ############# Stage 1(b): Uni-CXR training, skip if distillation is not enabled #############
    if cfg.model.cxr_modal_distill:
        if not cfg.retrain_teachers and os.path.exists(cfg.cxr_chkp):
            unicxr_teacher_model = UniCXRModel.load_from_checkpoint(cfg.cxr_chkp, map_location='cpu')
            print(f"Loaded UniCXR model from {cfg.cxr_chkp}")
        else:
            print('Training UniCXR model from scratch. If the checkpoint already exists, it will be overwritten.')
            unicxr_teacher_model = UniCXRModel(cfg)
            out = train_model(model=unicxr_teacher_model, cfg=cfg, log_dir=log_dir,
                              train_loader=train_loader, val_loader=val_loader, stage='unicxr')
            unicxr_teacher_model = out['model']
            shutil.copy(out['best_model_path'], cfg.cxr_chkp)
    else:
        unicxr_teacher_model = None
    
    ############# Stage 2: Fusion model training #############
    fuse_model = LateFuse(cfg)
    fuse_model.class_names = train_loader.dataset.CLASSES
    out = train_model(model=fuse_model, cfg=cfg, log_dir=log_dir,
                      train_loader=train_loader, val_loader=val_loader, stage='fuse')
    fuse_model = out['model']
    best_model_path = out['best_model_path']
    trainer = out['trainer']
    
    ############# Testing #############
    if not cfg.dev_run:
        best_model = LateFuse.load_from_checkpoint(best_model_path, map_location='cpu')
        best_model.class_names = train_loader.dataset.CLASSES
        trainer.test(model=best_model, dataloaders=test_loader)
        with open(os.path.join(out['log_dir'], 'test_set_results.yaml'), 'w') as f:
            yaml.dump(best_model.test_results, f)
        print(best_model.test_results)

    print(f'The best model is saved in {best_model_path}')
    
    return best_model.test_results


if __name__ == '__main__':
    test_results = train_fusion_model()
