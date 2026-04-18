import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from medmnist import INFO, dataset as medmnist_dataset
import DA.data_augmentation_albumentations as data_augmentation_albumentations

class MEDMNISTAlbumentations(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None, download=True, data_flag='breastmnist', indices=None):
        info = INFO[data_flag]
        DataClass = getattr(medmnist_dataset, info['python_class'])
        self.dataset = DataClass(root=root, split=split, download=download)
        self.transform = transform
        self.indices = indices
        
    def __len__(self):
        return len(self.dataset) if self.indices is None else len(self.indices)

    def __getitem__(self, index):
        actual_index = self.indices[index] if self.indices is not None else index
        img, target = self.dataset[actual_index]
        
        img = np.array(img)

        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=2)
        elif img.shape[-1] == 1:
            img = img.repeat(3, axis=-1)

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed['image']

        return {
            "images": img, 
            "label": torch.tensor(target).long().squeeze()
        }


def load_medmnist_datasets(config):
    root = config.get('cache_folder', './tmp')
    data_flag = config.get('dataset', 'breastmnist').lower()
    
    if not os.path.exists(root):
        os.makedirs(root)
    
    return root, data_flag, None, None 

def create_medmnist_loaders(config, train_data, val_data, test_data, tab_preproc, transform):
    root = train_data    
    data_flag = val_data 
    
    resize = [A.Resize(28, 28)]
    
    base_transforms = [
        #A.Resize(28, 28),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ]

    sl_augs = data_augmentation_albumentations.map_augments(transform, config)
    print(f"SL Augmentations for current individual: {sl_augs}")
    
    train_transform = A.Compose(resize + sl_augs + base_transforms)
    eval_transform = A.Compose(resize + base_transforms)

    train_set = MEDMNISTAlbumentations(root, 'train', transform=train_transform, data_flag=data_flag)
    val_set   = MEDMNISTAlbumentations(root, 'val',   transform=eval_transform,  data_flag=data_flag)
    test_set  = MEDMNISTAlbumentations(root, 'test',  transform=eval_transform,  data_flag=data_flag)

    batch_size = config.get('batch_size', 32)
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader

def create_medmnist_loaders_incremental(config, train_data, val_data, test_data, tab_preproc, transform):
    root = train_data    
    data_flag = val_data 
    
    resize = [A.Resize(28, 28)]
    
    base_transforms = [
        A.Resize(28, 28),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ]

    sl_augs = data_augmentation_albumentations.map_augments(transform, config)
    
    train_transform = A.Compose(resize + sl_augs + base_transforms)
    eval_transform = A.Compose(resize + base_transforms)
    
    full_train_set = MEDMNISTAlbumentations(root, 'train', data_flag=data_flag)
    total_len = len(full_train_set.dataset)
    
    split_idx = int(total_len * config.get('online_split', 0.8))
    indices = list(range(total_len))
    
    warmup_indices = indices[:split_idx]
    support_indices = indices[split_idx:]
    
    #For DAOP dynamic training, we will use the same dataset for warmup and sup, but with different transforms. The warmup_loader will use the eval_transform (without DAOP augmentations) and the sup_loader will use the train_transform  (with DAOP augmentations), so that the EA can apply the augmentations on the sup_loader data.
    warmup_set = MEDMNISTAlbumentations(root, 'train', transform=eval_transform, data_flag=data_flag, indices=warmup_indices)
    sup_set = MEDMNISTAlbumentations(root, 'train', transform=train_transform, data_flag=data_flag, indices=support_indices)
    val_set   = MEDMNISTAlbumentations(root, 'val',   transform=eval_transform,  data_flag=data_flag)
    test_set  = MEDMNISTAlbumentations(root, 'test',  transform=eval_transform,  data_flag=data_flag)
    
    print(f"[DATA] Total: {total_len} | Warmup: {len(warmup_set)} | Sup: {len(sup_set)} | Val: {len(val_set)} | Test: {len(test_set)} | Seed: {config['seed']}")
    
    batch_size = config.get('batch_size', 32)
    
    warmup_loader = torch.utils.data.DataLoader(
        warmup_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    sup_loader = torch.utils.data.DataLoader(
        sup_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )       
    
    return warmup_loader, sup_loader, val_loader, test_loader
    
    
    