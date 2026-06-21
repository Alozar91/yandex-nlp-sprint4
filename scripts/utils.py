import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchmetrics
import timm
from transformers import AutoModel, AutoTokenizer
from functools import partial

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

class FoodCaloriePredictor(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', image_model_name='tf_efficientnet_b0', hidden_dim=768):
        super().__init__()
        # Энкодер для изображений
        self.image_encoder = timm.create_model(image_model_name, pretrained=True, num_classes=0)
        self.image_feat_dim = self.image_encoder.num_features

        # Энкодер для текста
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_feat_dim = self.text_encoder.config.hidden_size

        # Проекционные слои
        self.image_projection = nn.Linear(self.image_feat_dim, hidden_dim)
        self.text_projection = nn.Linear(self.text_feat_dim, hidden_dim)

        # Финальный регрессионный слой
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, image, text_input_ids, text_attention_mask, mass):
        # Обработка изображения
        image_features = self.image_encoder(image)
        image_proj = self.image_projection(image_features)

        # Обработка текста
        text_outputs = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_proj = self.text_projection(text_features) 

        # Конкатенация признаков
        # combined = torch.cat([image_proj, text_proj], dim=1)
        # combined = image_proj * text_proj
        # combined = torch.cat([image_proj, text_proj], dim=1)
        combined = torch.cat([image_proj, text_proj, mass.unsqueeze(1)], dim=1)

        # Предсказание калорийности
        output = self.classifier(combined)  # [B, 1]
        return output.squeeze()


def train(cfg, model, train_loader, val_loader, device):
    """Функция обучения модели"""
    model.to(device)
    
    stop_training = False
    TARGET_MAE = 0.05

    # Оптимизатор с разными LR для разных частей модели
    optimizer = AdamW([
        {"params": model.text_encoder.parameters(), "lr": cfg.TEXT_LR, "weight_decay": cfg.WEIGHT_DECAY, "name": "text"},
        {"params": model.image_encoder.parameters(), "lr": cfg.IMAGE_LR, "weight_decay": cfg.WEIGHT_DECAY, "name": "image"},
        {"params": model.image_projection.parameters(), "lr": cfg.CLASSIFIER_LR, "weight_decay": cfg.WEIGHT_DECAY, "name": "img_proj"},
        {"params": model.text_projection.parameters(), "lr": cfg.CLASSIFIER_LR, "weight_decay": cfg.WEIGHT_DECAY, "name": "txt_proj"},
        {"params": model.classifier.parameters(), "lr": cfg.CLASSIFIER_LR, "weight_decay": cfg.WEIGHT_DECAY, "name": "clf"},
    ])

    criterion = nn.L1Loss()  # MAE — целевая метрика
    mae_metric = torchmetrics.MeanAbsoluteError().to(device)
    best_mae = float('inf')  # Для сохранения лучшей модели

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    for param in model.image_encoder.parameters():
        param.requires_grad = False
    for param in model.text_encoder.parameters():
        param.requires_grad = False

    for epoch in range(cfg.NUM_EPOCHS):
        # Обучение
        if stop_training:
            break        
        model.train()
        if epoch == 3:
            # Размораживаем энкодеры
            for param in model.image_encoder.parameters():
                param.requires_grad = True
            for param in model.text_encoder.parameters():
                param.requires_grad = True
                
            # НОВОЕ: Понижаем Learning Rate для энкодеров в 10 раз
            for g in optimizer.param_groups:
                if g['name'] == 'image':
                    g['lr'] = cfg.IMAGE_LR / 10
                elif g['name'] == 'text':
                    g['lr'] = cfg.TEXT_LR / 10
        train_loss = 0.0
        mae_metric.reset()
        if epoch == 3:
            for param in model.image_encoder.parameters():
                param.requires_grad = True
            for param in model.text_encoder.parameters():
                param.requires_grad = True
        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            mass = batch['mass'].to(device)
            targets = batch['calories'].to(device)

            optimizer.zero_grad()
            outputs = model(images, text_input_ids, text_attention_mask, mass)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            mae_metric(outputs, targets)

            # Логирование по батчам (опционально)
            if i % cfg.LOG_INTERVAL == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

        train_mae = mae_metric.compute().item()
        avg_train_loss = train_loss / len(train_loader)

        # Валидация
        model.eval()
        val_loss = 0.0
        mae_metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                text_input_ids = batch['text_input_ids'].to(device)
                text_attention_mask = batch['text_attention_mask'].to(device)
                mass = batch['mass'].to(device)                # ← добавить
                targets = batch['calories'].to(device)

                outputs = model(images, text_input_ids, text_attention_mask, mass)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                mae_metric(outputs, targets)

        val_mae = mae_metric.compute().item()
        avg_val_loss = val_loss / len(val_loader)

        # Обновление learning rate scheduler
        scheduler.step(val_mae)

        # Сохранение лучшей модели
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), cfg.SAVE_PATH)
            print(f"Модель сохранена с MAE: {val_mae:.4f}")

        if (epoch + 1) % cfg.PRINT_EVERY == 0:
            print(
                f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train MAE: {train_mae:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val MAE: {val_mae:.4f}"                
            )

        # --- Проверка условия остановки ---
        if val_mae <= TARGET_MAE:
            print(f"\nЦелевое MAE {TARGET_MAE*100:.0f}% достигнуто на эпохе {epoch+1}!")
            stop_training = True
            model.to('cpu')          
            torch.save(model.state_dict(), cfg.SAVE_PATH)
            print(f"Модель сохранена в {cfg.SAVE_PATH}")
            continue

    print(f"\nОбучение завершено. Лучшая MAE на валидации: {best_mae:.4f}")
    return best_mae