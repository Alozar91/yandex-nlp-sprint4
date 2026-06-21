import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from PIL import Image

def evaluate_model(cfg, model, test_loader, device):
    """Оценка модели на тестовой выборке"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_mass = []  # Сохраняем массу для анализа

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            mass = batch['mass'].to(device)  # ← ДОБАВИТЬ массу
            targets = batch['calories'].cpu().numpy()

            # Передаём массу в модель
            outputs = model(images, text_input_ids, text_attention_mask, mass)
            predictions = outputs.cpu().numpy()

            all_predictions.extend(predictions)
            all_targets.extend(targets)
            all_mass.extend(batch['mass'].cpu().numpy())

    # Денормализация (умножаем на 1000, так как делили на 1000 при подготовке)
    predictions_denorm = np.array(all_predictions) * 1000
    targets_denorm = np.array(all_targets) * 1000

    # Расчёт метрик на денормализованных данных
    mae = mean_absolute_error(targets_denorm, predictions_denorm)
    mse = mean_squared_error(targets_denorm, predictions_denorm)
    rmse = np.sqrt(mse)

    results = {
        'predictions': predictions_denorm.tolist(),  # Уже в ккал
        'targets': targets_denorm.tolist(),          # Уже в ккал
        'mass': all_mass,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }

    print(f"="*50)
    print(f"ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ:")
    print(f"="*50)
    print(f"Тестовая MAE: {mae:.2f} ккал")
    print(f"Тестовая RMSE: {rmse:.2f} ккал")
    print(f"Целевая метрика: MAE < 50 ккал")
    if mae < 50:
        print(f"✅ ЦЕЛЬ ДОСТИГНУТА!")
    else:
        print(f"⚠️  Цель не достигнута, отклонение: {mae - 50:.2f} ккал")
    print(f"="*50)

    return results

def plot_predictions_vs_targets(results, save_path="results/predictions_vs_targets.png"):
    """Построение графика предсказаний vs истинных значений"""
    plt.figure(figsize=(10, 8))
    plt.scatter(results['targets'], results['predictions'], alpha=0.6, c='blue')
    
    # Линия идеального предсказания
    min_val = min(min(results['targets']), min(results['predictions']))
    max_val = max(max(results['targets']), max(results['predictions']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальное предсказание')
    
    plt.xlabel('Истинная калорийность (ккал)', fontsize=12)
    plt.ylabel('Предсказанная калорийность (ккал)', fontsize=12)
    plt.title(f'Предсказания vs Истинные значения\nMAE: {results["mae"]:.2f} ккал', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_errors(df, results, top_k=5):
    """Анализ худших и лучших предсказаний"""
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    errors_df = pd.DataFrame({
        'dish_id': test_df['dish_id'].values,
        'true_calories': results['targets'],
        'pred_calories': results['predictions'],
        'error': np.abs(np.array(results['predictions']) - np.array(results['targets'])),
        'error_signed': np.array(results['predictions']) - np.array(results['targets']),
        'ingredients': test_df['ingredient_names'].values,
        'total_mass': test_df['total_mass'].values
    })

    # Топ-5 худших предсказаний (наибольшая ошибка)
    worst_errors = errors_df.nlargest(top_k, 'error')
    # Топ-5 лучших предсказаний (наименьшая ошибка)
    smallest_errors = errors_df.nsmallest(top_k, 'error')

    print("\n" + "="*50)
    print("ТОП-5 ХУДШИХ ПРЕДСКАЗАНИЙ:")
    print("="*50)
    for idx, row in worst_errors.iterrows():
        print(f"\nБлюдо ID: {row['dish_id']}")
        print(f"  Реальная калорийность: {row['true_calories']:.1f} ккал")
        print(f"  Предсказанная:         {row['pred_calories']:.1f} ккал")
        print(f"  Ошибка:                {row['error']:.1f} ккал ({row['error_signed']:+.1f})")
        print(f"  Масса порции:          {row['total_mass']:.0f} г")

    print("\n" + "="*50)
    print("ТОП-5 ЛУЧШИХ ПРЕДСКАЗАНИЙ:")
    print("="*50)
    for idx, row in smallest_errors.iterrows():
        print(f"\nБлюдо ID: {row['dish_id']}")
        print(f"  Реальная калорийность: {row['true_calories']:.1f} ккал")
        print(f"  Предсказанная:         {row['pred_calories']:.1f} ккал")
        print(f"  Ошибка:                {row['error']:.1f} ккал")

    return worst_errors, smallest_errors

def plot_error_distribution(results, save_path="results/error_distribution.png"):
    """Построение распределения ошибок"""
    errors = np.array(results['predictions']) - np.array(results['targets'])

    plt.figure(figsize=(12, 6))
    
    # Гистограмма
    plt.subplot(1, 2, 1)
    sns.histplot(errors, bins=40, kde=True, alpha=0.7, color='skyblue')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Нулевая ошибка')
    plt.xlabel('Ошибка предсказания (ккал)', fontsize=11)
    plt.ylabel('Количество блюд', fontsize=11)
    plt.title('Распределение ошибок', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(errors, vert=True, patch_artist=True, labels=['Ошибки'])
    plt.ylabel('Ошибка (ккал)', fontsize=11)
    plt.title('Box plot ошибок', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Статистика ошибок
    print(f"\nСтатистика ошибок:")
    print(f"  Среднее:     {np.mean(errors):.2f} ккал")
    print(f"  Медиана:     {np.median(errors):.2f} ккал")
    print(f"  Std:         {np.std(errors):.2f} ккал")
    print(f"  Мин:         {np.min(errors):.2f} ккал")
    print(f"  Макс:        {np.max(errors):.2f} ккал")

def categorize_by_calorie_range(df, results):
    """Анализ ошибок по диапазонам калорийности"""
    test_df = df[df['split'] == 'test'].copy().reset_index(drop=True)
    test_df['pred_calories'] = results['predictions']
    test_df['error'] = np.abs(test_df['pred_calories'] - test_df['total_calories'])

    # Создаём диапазоны калорийности
    bins = [0, 150, 300, 500, 800, np.inf]
    labels = ['0–150', '150–300', '300–500', '500–800', '800+']
    test_df['calorie_range'] = pd.cut(test_df['total_calories'], bins=bins, labels=labels)

    # Средняя ошибка по диапазонам
    error_by_range = test_df.groupby('calorie_range', observed=False).agg({
        'error': ['mean', 'std', 'count'],
        'total_calories': 'mean'
    }).round(2)

    print("\n" + "="*60)
    print("АНАЛИЗ ОШИБОК ПО ДИАПАЗОНАМ КАЛОРИЙНОСТИ:")
    print("="*60)
    print(error_by_range)
    print("="*60)

    # Визуализация
    plt.figure(figsize=(10, 6))
    mean_errors = test_df.groupby('calorie_range')['error'].mean()
    plt.bar(mean_errors.index, mean_errors.values, alpha=0.7, color='coral')
    plt.axhline(y=50, color='red', linestyle='--', label='Целевая MAE (50 ккал)')
    plt.xlabel('Диапазон калорийности (ккал)', fontsize=11)
    plt.ylabel('Средняя абсолютная ошибка (ккал)', fontsize=11)
    plt.title('Средняя ошибка по диапазонам калорийности', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    return error_by_range

def visualize_worst_predictions(worst_errors, dish_df, image_dir, save_path="results/worst_predictions_analysis.png"):
    """
    Визуализация топ-5 худших предсказаний с фотографиями, 
    ингредиентами и анализом ошибок
    """
    fig, axes = plt.subplots(5, 2, figsize=(16, 25))
    fig.suptitle("Анализ топ-5 блюд с наибольшей ошибкой предсказания", 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for i, (idx, row) in enumerate(worst_errors.iterrows()):
        dish_id = row['dish_id']
        
        # Получаем информацию о блюде из датасета
        dish_info = dish_df[dish_df['dish_id'] == dish_id].iloc[0]
        ingredients = dish_info['ingredient_names']
        total_mass = dish_info['total_mass']
        
        # === ЛЕВАЯ КОЛОНКА: Фотография ===
        img_path = os.path.join(image_dir, str(dish_id), "rgb.png")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Блюдо ID: {dish_id}", fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')
        else:
            axes[i, 0].text(0.5, 0.5, "Изображение не найдено", 
                           ha='center', va='center', fontsize=12)
            axes[i, 0].axis('off')
        
        # === ПРАВАЯ КОЛОНКА: Информация ===
        # Определяем направление ошибки
        error_direction = "ЗАВЫШЕНА" if row['error_signed'] > 0 else "ЗАНИЖЕНА"
        error_color = 'red' if abs(row['error']) > 100 else 'orange'
        
        text = (
            f"📊 ИНФОРМАЦИЯ О БЛЮДЕ\n"
            f"{'='*40}\n\n"
            f"🍽️  ИНГРЕДИЕНТЫ:\n"
            f"{ingredients}\n\n"
            f"⚖️  Масса порции: {total_mass:.0f} г\n\n"
            f"📈 КАЛОРИЙНОСТЬ:\n"
            f"  Реальная:          {row['true_calories']:.1f} ккал\n"
            f"  Предсказанная:     {row['pred_calories']:.1f} ккал\n"
            f"  Абсолютная ошибка: {row['error']:.1f} ккал\n"
            f"  Направление:       {error_direction} на {abs(row['error_signed']):.1f} ккал\n\n"
            f"💡 ВОЗМОЖНЫЕ ПРИЧИНЫ ОШИБКИ:\n"
        )
        
        # Добавляем гипотезы в зависимости от типа ошибки
        hypotheses = []
        
        # Если модель завысила калорийность
        if row['error_signed'] > 50:
            hypotheses.append("• Модель могла переоценить размер порции")
            hypotheses.append("• Высококалорийные ингредиенты (масло, соусы) не указаны явно")
            if total_mass < 200:
                hypotheses.append("• Маленькая порция, но модель предсказала большую")
        
        # Если модель занизила калорийность
        elif row['error_signed'] < -50:
            hypotheses.append("• Модель недооценила калорийность ингредиентов")
            hypotheses.append("• Возможно, не учтены скрытые калории (масло для жарки, заправки)")
            if 'cheese' in ingredients.lower() or 'sauce' in ingredients.lower():
                hypotheses.append("• Присутствуют высококалорийные компоненты (сыр, соус)")
        
        # Общие причины
        if total_mass > 500:
            hypotheses.append("• Очень большая порция — сложно точно оценить")
        if len(ingredients.split(',')) > 10:
            hypotheses.append("• Много ингредиентов — модель могла что-то упустить")
        
        if not hypotheses:
            hypotheses.append("• Нетипичное сочетание ингредиентов")
            hypotheses.append("• Редкое блюдо, мало примеров в обучающей выборке")
        
        text += "\n".join(hypotheses)
        
        axes[i, 1].text(0.05, 0.95, text, transform=axes[i, 1].transAxes,
                       fontsize=10, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[i, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Визуализация сохранена в: {save_path}")
    plt.show()

def print_final_analysis(worst_errors, results):
    """
    Печатает итоговый текстовый анализ ошибок модели
    """
    print("\n" + "="*70)
    print("📝 ИТОГОВЫЙ АНАЛИЗ ОШИБОК МОДЕЛИ")
    print("="*70)
    
    print("\n1️⃣  ОБЩАЯ СТАТИСТИКА:")
    print(f"   • Средняя ошибка (MAE): {results['mae']:.2f} ккал")
    print(f"   • Целевая метрика:      MAE < 50 ккал")
    if results['mae'] < 50:
        print(f"   • ✅ СТАТУС: Цель достигнута!")
    else:
        print(f"   • ⚠️  СТАТУС: Требуется доработка")
    
    print("\n2️⃣  ТИПИЧНЫЕ ПРИЧИНЫ ОШИБОК:")
    print("   • Размер порции: модель не всегда точно оценивает массу по фото")
    print("   • Скрытые ингредиенты: масло для жарки, заправки, специи")
    print("   • Способ приготовления: жарка vs запекание vs варка")
    print("   • Редкие блюда: мало примеров в обучающей выборке")
    print("   • Сложные блюда: много компонентов, трудно учесть все")
    
    print("\n3️⃣  РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:")
    print("   • Добавить признак массы порции как отдельный вход (уже сделано)")
    print("   • Увеличить датасет, особенно редких блюд")
    print("   • Добавить информацию о способе приготовления")
    print("   • Использовать более мощные энкодеры (EfficientNet-B3/B4)")
    print("   • Попробовать ансамбль моделей")
    
    print("\n4️⃣  НАИБОЛЕЕ ПРОБЛЕМНЫЕ КАТЕГОРИИ:")
    # Анализируем, какие блюда чаще ошибаются
    worst_categories = []
    for idx, row in worst_errors.iterrows():
        ingredients = row['ingredients'].lower()
        if 'sauce' in ingredients or 'dressing' in ingredients:
            worst_categories.append("Блюда с соусами и заправками")
        elif 'cheese' in ingredients or 'cream' in ingredients:
            worst_categories.append("Блюда с молочными продуктами")
        elif 'fried' in ingredients or 'roasted' in ingredients:
            worst_categories.append("Жареные/запечённые блюда")
        else:
            worst_categories.append("Разное")
    
    from collections import Counter
    category_counts = Counter(worst_categories)
    for cat, count in category_counts.most_common():
        print(f"   • {cat}: {count} случаев")
    
    print("\n" + "="*70)