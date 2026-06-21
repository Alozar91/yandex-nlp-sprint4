import os
import numpy as np
import pandas as pd
from PIL import Image
import timm
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from transformers import AutoTokenizer
import torch

def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose(
            [
                # ЗАМЕНИТЬ SmallestMaxSize и RandomCrop на Resize:
                A.Resize(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42
        )
    else:
        transforms = A.Compose(
            [
                # ЗАМЕНИТЬ SmallestMaxSize и CenterCrop на Resize:
                A.Resize(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42
        )
    
    return transforms


class FoodDataset(Dataset):
    def __init__(self, config, df, image_dir, tokenizer, max_length=128, split='train', is_train=True):
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.df['normalized_calories'] = self.df['total_calories'] / 1000  # Нормализация
        self.df['norm_mass'] = self.df['total_mass'] / 1000.0
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Новый флаг для управления аугментациями
        self.is_train = is_train
        # Вызов get_transforms с соответствующим типом
        self.augment = get_transforms(config=config, ds_type='train' if is_train else 'val')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dish_id = row['dish_id']
        # calories = row['total_calories']
        calories = row['normalized_calories']  # Использовать нормализованные значения
        mass = torch.tensor(row['norm_mass'], dtype=torch.float32)

        # Загрузка изображения
        img_path = os.path.join(self.image_dir, str(dish_id), "rgb.png")
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        image = self.augment(image=image)['image']

        # Токенизация текста (список ингредиентов)
        # ingredients_text = ', '.join(row['ingredient_names'])
        ingredients_text = row['ingredient_names'].replace(';', ', ')
        text_encoded = self.tokenizer(
            ingredients_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        text_input_ids = text_encoded['input_ids'].squeeze()
        text_attention_mask = text_encoded['attention_mask'].squeeze()

        return {
            'image': image,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'mass': mass,
            'calories': torch.tensor(calories, dtype=torch.float32)
        }

def get_dataloaders(config, df, image_dir, tokenizer, batch_size=4):
    # Отделяем train-блюда
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    # Перемешиваем и делим
    indices = torch.randperm(len(train_df)).tolist()
    train_size = int(0.8 * len(indices))
    
    train_dataset = FoodDataset(config, train_df.iloc[indices[:train_size]], image_dir, tokenizer, split='train', is_train=True)
    val_dataset = FoodDataset(config, train_df.iloc[indices[train_size:]], image_dir, tokenizer, split='train', is_train=False)
    test_dataset = FoodDataset(config, df[df['split'] == 'test'].reset_index(drop=True), image_dir, tokenizer, split='test', is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader