import os
import pickle
from pathlib import Path
import ast
import re

import yaml
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image
import random

import torch
import lightning as L
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


# set dataloader seet
def set_seed(_seed):
    global seed
    seed = _seed

    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU


class MultiModalMIMIC(Dataset):
    def __init__(self, data_root, fold, partition, cxr_img_root, task,
                 time_limit=48, normalization='robust_scale', ehr_time_step=1,
                 matched_subset=True, imagenet_normalization=True,
                 preload_images=False, pkl_dir=None, attribution_cols=None,index = None,one_hot = None,
                 resized_base_path='/research/mimic_cxr_resized',
                 image_meta_path="/hdd/datasets/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv"):
        self.task = task
        self.normalization = normalization
        self.ehr_time_step = ehr_time_step
        self.time_limit = time_limit
        # self.cxr_img_root = Path(cxr_img_root)
        self.matched_subset = matched_subset
        self.index = index
        self.one_hot = one_hot
        self.preload_images = {}
        self.resized_base_path = resized_base_path

        if attribution_cols is None:
            self.attribution_cols = ['first_careunit', 'age', 'gender', 'admission_type', 'admission_location',
                                     'insurance', 'marital_status', 'race']

        self.data_root = Path(data_root)
        if pkl_dir is not None:
            ehr_pkl_fpath = Path(pkl_dir) / f'{task}_fold{fold}_{partition}_timestep{ehr_time_step}_{normalization}_{"matched" if matched_subset else "full"}_ts.pkl'
            cxr_pkl_fpath = Path(pkl_dir) / f'{task}_fold{fold}_{partition}_timestep{ehr_time_step}_{normalization}_{"matched" if matched_subset else "full"}_cxr.pkl'
        else:
            ehr_pkl_fpath = None
            cxr_pkl_fpath = None


        # EHR data prepare------------

        # load EHR metadata and feature statistics (full data)
        meta_files_root = self.data_root/'splits'/f'fold{fold}'
        self.ehr_meta = pd.read_csv(meta_files_root/f'stays_{partition}.csv') 
        with open(meta_files_root/'train_stats.yaml', 'r') as f:
            self.train_stats = yaml.safe_load(f)


        # get ehr cols name
        _stay_sample = pd.read_csv(self.data_root/'time_series'/f'{self.ehr_meta.stay_id.loc[0]}.csv')
        self.features = [x for x in _stay_sample.columns if x in self.train_stats]
        self.mask_name = [x for x in _stay_sample.columns if 'mask' in x]

        # get attribute of each col
        self.features_stats = {
            stat: np.array([self.train_stats[feat][stat] for feat in self.features]).astype(float)
            for stat in ['iqr','max','mean','median','min','std']
        }
        self.features_no_normalization = [feat for feat in self.features if not self.train_stats[feat]['normalize']]

        #  set imputation value 
        self.default_imputation = {feat: self.train_stats[feat]['median'] for feat in self.features}

        if self.matched_subset :
            self.ehr_meta = self.ehr_meta[(self.ehr_meta['valid_cxrs'] != '[]') & (self.ehr_meta['valid_cxrs'].notna())]
            self.stay_ids = self.ehr_meta['stay_id'].tolist()

        # load EHR time series, process ehr data
        self.stay_ids = self.ehr_meta['stay_id'].tolist()
        if ehr_pkl_fpath and ehr_pkl_fpath.exists():
            with open(ehr_pkl_fpath, 'rb') as f:
                self.normalized_data, self.missing_masks = pickle.load(f)
            print('Time series data loaded from pkl file.')
        else:
            self.normalized_data, self.missing_masks = self.load_and_normalize_time_series()
            if ehr_pkl_fpath:
                with open(ehr_pkl_fpath, 'wb') as f:
                    pickle.dump([self.normalized_data, self.missing_masks], f)


        # CXR data prepare-----------------------

        # load CXR metadata
        # image_meta_path = "/hdd/datasets/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv"
        self.cxr_meta = pd.read_csv(image_meta_path)

        # define transformation for CXR
        cxr_transform = [transforms.Resize(256)]
        if partition == 'train':
            cxr_transform += [
                # transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)),
            ]
        cxr_transform += [
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]
        if imagenet_normalization:
            cxr_transform += [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.cxr_transform = transforms.Compose(cxr_transform)

        # preload CXR image
        if cxr_pkl_fpath is not None and cxr_pkl_fpath.exists():    
            with open(cxr_pkl_fpath, 'rb') as f:
                self.preload_images = pickle.load(f)
            print('CXR images loaded from pkl file.')
        else:
            for stay_id in tqdm(self.stay_ids,desc="Preloading CXR images"):
                self.preload_images[stay_id] = self._get_last_cxr_image_by_stay_id(stay_id, self.resized_base_path)
            if cxr_pkl_fpath:
                with open(cxr_pkl_fpath, 'wb') as f:
                    pickle.dump(self.preload_images, f) 


        # create labels for prediction task
        if task == 'mortality':
            self.CLASSES = ['Mortality']
            self.targets = self.ehr_meta['icu_mortality'].values # icu_mortality or  hadm_mortality
        elif task == 'phenotype':
            self.CLASSES = self.ehr_meta.columns[-26:-1].tolist() # 25 labels
            self.targets = self.ehr_meta[self.CLASSES].values
        else:
            raise ValueError(f'Unknown task `{task}`')

        
        self.meta_attr = self.ehr_meta.set_index('stay_id')
        self.meta_attr = self.meta_attr[self.attribution_cols]

    def __getitem__(self, idx):
        stay_id = self.stay_ids[idx]
        data = torch.FloatTensor(self.normalized_data[stay_id][:self.time_limit]) # [time_step,features]
        masks = torch.FloatTensor(self.missing_masks[stay_id][:self.time_limit])
        labels = torch.FloatTensor(np.atleast_1d(self.targets[idx])) # [25]

        if self.index is not None:
            index = self.index
            labels = labels[index].unsqueeze(0)

        # get image 
        cxr_img = self.cxr_transform(self.preload_images[stay_id]) if self.preload_images[stay_id] is not None else None
        has_cxr = False if cxr_img == None else True
        groups = None
        meta_attrs = self.meta_attr.loc[stay_id]

        if self.one_hot and self.task == "mortality":
            num_classes = 2 
            labels_one_hot = torch.zeros(num_classes)  
            labels_one_hot.scatter_(0, labels.long(), 1)  
            return stay_id, data, masks, cxr_img, has_cxr, labels_one_hot, meta_attrs, torch.LongTensor([idx])
        
        return stay_id, data, masks, cxr_img, has_cxr, labels, meta_attrs, torch.LongTensor([idx])

 
        

    def __len__(self):
        return len(self.stay_ids)

    def __load_time_series_by_stay_id(self, stay_id):
        stay_data_origin = pd.read_csv(self.data_root/'time_series'/f'{stay_id}.csv').sort_values(by='time_step')
        stay_data =  stay_data_origin[['time_step'] + self.features] # 提取需要的特征
        stay_data_mask = stay_data_origin[self.mask_name] # mask data

        # aggregate data into time steps
        # time_step = 3600 * self.ehr_time_step # 一小时为单位
        # stay_data['time_step'] = (stay_data['time_step']//time_step).astype(int)
        # stay_data = stay_data.groupby('time_step').mean() # 计算平均值
        # stay_data = stay_data.reindex(range(stay_data.index.max()))

        # get missing mask
        missing_mask = stay_data[self.features].isna().astype(float).values # [time_steps,feature] 0 means missing

        # apply imputation 对缺失值进行填充
        data_imputed = stay_data[self.features].ffill().fillna(self.default_imputation)
       
        # robust normalization
        data_normalized = (data_imputed - self.features_stats['median']) / self.features_stats['iqr']
        data_normalized[self.features_no_normalization] = data_imputed[self.features_no_normalization]

        # min max normalization
        #data_normalized = (data_imputed - self.features_stats['min']) / (self.features_stats['max'] - self.features_stats['min'])

        # Z score normalization
        #data_normalized = (data_imputed - self.features_stats['mean']) / self.features_stats['std']

        #data_normalized[self.do_not_scales] = data_imputed[self.do_not_scales]
        #data_normalized = data_normalized.values # 转化成numpy数组 [time_step,features]
        concatenated_data = np.concatenate((data_normalized, stay_data_mask), axis=1)  # 按列拼接，shape = [time_step, features + mask]
        #print(f"shape of concatenated_data is {concatenated_data.shape}, shape of data_mask is {stay_data_mask.shape}")


        return stay_id, concatenated_data, missing_mask

    def load_and_normalize_time_series(self):
        normalized_data = {}
        missing_masks = {}

        for stay_id in tqdm(self.stay_ids, desc='Loading and pre-processing raw time series'):
            _, data, masks = self.__load_time_series_by_stay_id(stay_id)
            normalized_data[stay_id] = data
            missing_masks[stay_id] = masks

        return normalized_data, missing_masks

    def _get_last_cxr_image_by_stay_id(self, stay_id, resized_base_path='/research/mimic_cxr_resized'):
        valid_cxrs = self.ehr_meta.loc[self.ehr_meta['stay_id'] == stay_id, 'valid_cxrs'].values[0]
        
        # not cxr data
        if pd.isna(valid_cxrs) or valid_cxrs == "[]":
            return None


        # re parse dicom_id
        valid_cxrs_clean = re.sub(r"Timestamp\('([^']+)'\)", r"'\1'", valid_cxrs)
        valid_cxrs_clean_parse = ast.literal_eval(valid_cxrs_clean)
        dicom_id = valid_cxrs_clean_parse[-1][0]

        subject_id = self.ehr_meta.loc[self.ehr_meta['stay_id'] == stay_id, 'subject_id'].values[0]
        img_path = self.get_image_path(dicom_id, subject_id, resized_base_path=resized_base_path)
        if img_path is None:
            return None

        try:
            cxr_img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"{img_path} not exists!!!!")
            return None

        return cxr_img


    def get_image_path(self, dicom_id, subject_id, resized_base_path='/research/mimic_cxr_resized'):

        # matching_rows = self.cxr_meta.loc[self.cxr_meta['dicom_id'] == dicom_id]
        # # print(f"dicom is {dicom_id}")
        # # /research/mimic_cxr_resized
        
        # # 检查是否有匹配的行
        # if matching_rows.empty:
        #     #print(f"dicom_id {dicom_id} not found in cxr_meta")
        #     return None

        # # 提取 study_id
        # study_id = matching_rows['study_id'].values[0]
        # s = f"s{study_id}"
        # p_head = f"p{str(subject_id)[:2]}"
        # p_full = f"p{subject_id}"
        # image_path = f"/hdd/datasets/mimic-cxr-jpg/2.0.0/files/{p_head}/{p_full}/{s}/{dicom_id}.jpg"

        image_path = f"{resized_base_path}/{dicom_id}.jpg"

        return image_path


def pad_temporal_data(batch):
    # 每个Batch下进行的操作
    # 返回的value将被输入到model中
    #stay_ids, data, masks, cxr_offsets, cxr_imgs, has_cxr, labels, groups, meta_attrs,idx = zip(*batch)
    stay_ids, data, masks, cxr_imgs, has_cxr, labels, meta_attrs,idx = zip(*batch)
    seq_len = [x.shape[0] for x in data] # 获取时间步
    max_len = max(seq_len)
    # 填充每个样本到 [max_len, features] mask也是
    # stack重新回归到[batch,]
    data_padded = torch.stack([torch.cat([x, torch.zeros(max_len-x.shape[0], x.shape[1])], dim=0)
                               for x in data], dim=0)
    masks_padded = torch.stack([torch.cat([x, torch.zeros(max_len-x.shape[0], x.shape[1])], dim=0)
                               for x in masks], dim=0)

    # 创建一个空列表，用于存储处理后的结果
    processed_cxr_imgs = []

    # 遍历 cxr_imgs 中的每个元素
    for x in cxr_imgs:
        if x is None:
            processed_cxr_imgs.append(torch.zeros(3, 224, 224))
        else:
            if isinstance(x, tuple):
                processed_cxr_imgs.append(torch.tensor(x))
            else:
                processed_cxr_imgs.append(x)

    cxr_imgs = torch.stack(processed_cxr_imgs)


    has_cxr = torch.FloatTensor(has_cxr)
    labels = torch.stack(labels, dim=0)

    idx = torch.stack(idx, dim=0)
    meta_attrs = pd.DataFrame(meta_attrs)
    #print(f"labels is {labels.shape}")
    batch_data = {
        'stay_ids': list(stay_ids),
        'seq_len': seq_len,
        'ehr_ts': data_padded,
        'ehr_masks': masks_padded,
        'cxr_imgs': cxr_imgs,
        'has_cxr': has_cxr,
        'labels': labels,
        'meta_attrs': meta_attrs,
        'idx':idx
    }
    return batch_data

def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def create_data_loaders(ehr_data_dir, cxr_data_dir, task, replication, batch_size,
                        num_workers, time_limit=None, matched_subset=True,index = None,seed = None, one_hot=False,
                        pkl_dir='./data_pkls/', resized_base_path='/research/mimic_cxr_resized',
                        image_meta_path="/hdd/datasets/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv"):
    set_seed(seed)
    time_limit = 48

    data_loaders = []
    for split in ['train', 'val', 'test']: #['train', 'val', 'test']
        is_train = (split == 'train')
        ds = MultiModalMIMIC(ehr_data_dir, replication, split,
                             cxr_data_dir, task, time_limit=time_limit,matched_subset = matched_subset,index = index, one_hot = one_hot,
                             pkl_dir=pkl_dir, resized_base_path=resized_base_path,
                             image_meta_path=image_meta_path)
        dl = DataLoader(ds, pin_memory=True,
                        shuffle=is_train, drop_last=is_train,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        collate_fn=pad_temporal_data,
                        worker_init_fn=seed_worker)
        data_loaders.append(dl)
    return data_loaders
