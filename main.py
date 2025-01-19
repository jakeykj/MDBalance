import os
import pickle
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

@hydra.main(config_path='configs', config_name='default', version_base=None)
def run_model(cfg: DictConfig):
    set_seed(cfg.seed)
    
    ############# Create model #############
    if cfg.stage == 'uniehr':
        model_class = UniEHRTransformer

    elif cfg.stage == 'unicxr':
        model_class = UniCXRModel
        cfg.data.matched = True  # use only matched data for unicxr model

    elif cfg.stage == 'fuse':
        model_class = LateFuse
    
    else:
        raise ValueError(f'Unknown stage `{cfg.stage}`')
    
    model = model_class(cfg)

    ############# Create data loaders #############
    train_loader, val_loader, test_loader = create_data_loaders(cfg.data.ehr_root, None, cfg.data.task,
                                                                cfg.data.fold, cfg.training.batch_size, cfg.training.num_workers,
                                                                matched_subset=cfg.data.matched, index=None, seed=cfg.seed, one_hot=False,
                                                                resized_base_path=cfg.data.resized_cxr_root,
                                                                image_meta_path=cfg.data.image_meta_path)

    
    ############# Configure training and logging #############
    callback_metric = 'overall/PRAUC'
    early_stop_callback = EarlyStopping(monitor=callback_metric,
                                        min_delta=0.00,
                                        patience=cfg.training.patience,
                                        verbose=False,
                                        mode="max")

    # file name
    log_dir = f'./experiments/{cfg.data.task}_{"matched" if cfg.data.matched else "fulldata"}/{cfg.stage}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    ver_name = (f'{cfg.name or "run"}-seed{cfg.seed}-{time.strftime("%Y%m%d%H%M%S")}')

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, version=ver_name)
    csv_logger = pl_loggers.CSVLogger(save_dir=log_dir, version=ver_name)

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

    ############# Test model #############
    best_model_path = checkpoint_callback.best_model_path
    print(f"best_model_path: {best_model_path}")

    best_model = model_class.load_from_checkpoint(best_model_path, strict=False)

    if not cfg.dev_run:
        trainer.test(model=best_model, dataloaders=test_loader)
        with open(os.path.join(csv_logger.log_dir, 'test_set_results.yaml'), 'w') as f:
            yaml.dump(best_model.test_results, f)
        print(best_model.test_results)
    
    return best_model.test_results


if __name__ == '__main__':
    test_results = run_model()
