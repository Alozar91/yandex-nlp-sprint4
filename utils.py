import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchmetrics
import timm
from transformers import AutoModel, AutoTokenizer
from functools import partial

class FoodCaloriePredictor(nn.Module):
    def __init__(self, text_model_name='distilbert-base-uncased', image_model_name='efficientnet_b3', hidden_dim=512):
        super().__init__()
        # Энкодер для изображений
        self.image_encoder = timm.create_model(image_model_name, pretrained=True, num_classes=0)
        image_feat_dim = self.image_encoder.num_features

        # Энкодер для текста
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_feat_dim = self.text_encoder.config.dim

        # Проекционные слои
        self.image_projection = nn.Linear(image_feat_dim, hidden_dim)
        self.text_projection = nn.Linear(text_feat_dim, hidden_dim)

        # Финальный регрессионный слой
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image, text_input_ids, text_attention_mask):
        # Обработка изображения
        image_features = self.image_encoder(image)  # [B, image_feat_dim]
        image_proj = self.image_projection(image_features)  # [B, hidden_dim]

        # Обработка текста
        text_outputs = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS token
        text_proj = self.text_projection(text_features)  # [B, hidden_dim]

        # Конкатенация признаков
        combined = torch.cat([image_proj, text_proj], dim=1)  # [B, 2*hidden_dim]

        # Предсказание калорийности
        output = self.classifier(combined)  # [B, 1]
        return output.squeeze()


def train(cfg, model, train_loader, val_loader, device):
    """Функция обучения модели"""
    model.to(device)

    # Оптимизатор с разными LR для разных частей модели
    optimizer = AdamW([
        {"params": model.text_encoder.parameters(), "lr": cfg.TEXT_LR},
        {"params": model.image_encoder.parameters(), "lr": cfg.IMAGE_LR},
        {"params": model.image_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
        {"params": model.text_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
        {"params": model.classifier.parameters(), "lr": cfg.CLASSIFIER_LR},
    ])

    criterion = nn.L1Loss()  # MAE — целевая метрика
    mae_metric = torchmetrics.MeanAbsoluteError().to(device)
    best_mae = float('inf')  # Для сохранения лучшей модели

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    for epoch in range(cfg.NUM_EPOCHS):
        # Обучение
        model.train()
        train_loss = 0.0
        mae_metric.reset()

        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            targets = batch['calories'].to(device)

            optimizer.zero_grad()
            outputs = model(images, text_input_ids, text_attention_mask)
            loss = criterion(outputs, targets)
            loss.backward()
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
                targets = batch['calories'].to(device)

                outputs = model(images, text_input_ids, text_attention_mask)
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
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_mae': train_mae,
                'val_mae': val_mae,
            }, cfg.SAVE_PATH)
            print(f"Модель сохранена с MAE: {val_mae:.4f}")

        if (epoch + 1) % cfg.PRINT_EVERY == 0:
            print(
                f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train MAE: {train_mae:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val MAE: {val_mae:.4f}"
            )

    print(f"\nОбучение завершено. Лучшая MAE на валидации: {best_mae:.4f}")
    return best_mae

def validate(model, val_loader, device, f1_metric):
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device)
            }
            labels = batch['label'].to(device)

            logits = model(**inputs)
            predicted = logits.argmax(dim=1)
            _ = f1_metric(preds=predicted, target=labels)

    return f1_metric.compute().cpu().numpy()