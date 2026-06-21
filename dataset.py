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

class FoodDataset(Dataset):
    def __init__(self, df, image_dir, tokenizer, max_length=128, split='train', augment=True):
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        # Аугментации для train, нормализация для test
        if self.augment:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dish_id = row['dish_id']
        calories = row['total_calories']

        # Загрузка изображения
        img_path = os.path.join(self.image_dir, str(dish_id), "rgb.png")
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        image = self.transform(image=image)['image']

        # Токенизация текста (список ингредиентов)
        ingredients_text = ', '.join(row['ingredient_names'])
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
            'calories': torch.tensor(calories, dtype=torch.float32)
        }

def get_dataloaders(df, image_dir, tokenizer, batch_size=4):
    """Создаёт dataloaders для train и test"""
    # Тренировочный датасет — с аугментациями
    train_dataset = FoodDataset(df, image_dir, tokenizer, split='train', augment=True)

    # Валидационный датасет — БЕЗ аугментаций
    val_dataset = FoodDataset(df, image_dir, tokenizer, split='train', augment=False)

    # Разделяем индексы для train/val
    total_size = len(train_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    # Создаём подмножества по индексам (используем те же индексы для обоих датасетов)
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Подмножества для train и val
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    # Тестовый датасет
    test_dataset = FoodDataset(df, image_dir, tokenizer, split='test', augment=False)

    # DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
