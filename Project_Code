import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns

# --- 1. ГЕНЕРАЦИЯ / ЗАГРУЗКА ДАННЫХ ---

def generate_synthetic_data(n_years=10, n_regions=3):
    """
    Генерирует синтетические данные для демонстрации.
    Структура: Регион | Год | Месяц | Температура | Осадки | NDVI | EVI | Урожайность
    """
    data_list = []
    regions = [f"Region_{i}" for i in range(n_regions)]
    
    np.random.seed(42)
    
    for region in regions:
        base_yield = np.random.uniform(30, 50) # Базовая урожайность для региона
        for year in range(2010, 2010 + n_years):
            # Сезонные факторы
            for month in range(1, 13):
                # Имитация сезонности (синусоиды)
                temp = 15 + 10 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 2)
                precip = max(0, 50 + 30 * np.cos(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 10))
                
                # NDVI и EVI коррелируют с температурой и осадками, но с задержкой
                ndvi = max(0, 0.3 + 0.4 * np.sin(2 * np.pi * (month - 3) / 12) + np.random.normal(0, 0.05))
                evi = ndvi * 0.9
                
                # Урожайность - это агрегированный показатель за ГОД.
                # Для обучения временных рядов мы часто предсказываем конечное значение,
                # имея данные по месяцам. В таблице продублируем годовой yield для всех месяцев,
                # чтобы модель могла "учиться" смотреть на динамику в течение года.
               
                year_yield = base_yield + (year - 2010) * 0.5 + np.random.normal(0, 2)
                
                data_list.append({
                    "Region": region,
                    "Year": year,
                    "Month": month,
                    "Temp": temp,
                    "Precip": precip,
                    "NDVI": ndvi,
                    "EVI": evi,
                    "Yield": year_yield # Целевая переменная
                })
                
    return pd.DataFrame(data_list)

df = generate_synthetic_data(n_years=12, n_regions=5)
print("Пример данных (первые 5 строк):")
print(df.head())

# --- 2. ПРЕДОБРАБОТКА ДАННЫХ ДЛЯ LSTM ---

# Выбираем признаки
feature_cols = ['Temp', 'Precip', 'NDVI', 'EVI']
target_col = 'Yield'

# Нормализация данных
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Масштабируем признаки
df_scaled = df.copy()
df_scaled[feature_cols] = scaler_features.fit_transform(df[feature_cols])

# Масштабируем целевую переменную
df_scaled[target_col] = scaler_target.fit_transform(df[[target_col]])

# Функция для создания временных окон 
# Мы берем данные за 12 месяцев и предсказываем урожайность (которая одинакова для всего года, 
# или можно предсказывать Yield последнего месяца в окне)
def create_sequences(data, region_col, feature_cols, target_col, sequence_length=12):
    sequences_X = []
    sequences_y = []
    region_info = [] # Чтобы помнить, к какому региону относится пример
    
    for region in data[region_col].unique():
        region_data = data[data[region_col] == region].sort_values(['Year', 'Month'])
        
        # Если данных меньше, чем длина окна, пропускаем
        if len(region_data) < sequence_length:
            continue
            
        values = region_data[feature_cols].values
        targets = region_data[target_col].values
        
        # Скользящее окно
        for i in range(len(region_data) - sequence_length + 1):
            # Берем окно длиной sequence_length (например, 12 месяцев)
            seq_x = values[i : i + sequence_length]
            # Цель - урожайность года, к которому относится окно
            seq_y = targets[i + sequence_length - 1] 
            
            sequences_X.append(seq_x)
            sequences_y.append(seq_y)
            region_info.append(region)
            
    return np.array(sequences_X), np.array(sequences_y), np.array(region_info)

# Создаем последовательности (по 12 месяцев)
X, y, regions = create_sequences(df_scaled, 'Region', feature_cols, target_col, sequence_length=12)

print(f"\nРазмерность входных данных X: {X.shape}") # (Samples, Timesteps, Features)
print(f"Размерность целевой переменной y: {y.shape}")

# Разделение на Train/Test (по времени не разбиваем, так как это разные годы и регионы, 
# просто перемешаем или возьмем последние пары год-регион в тест. Для простоты - случайное разбиение)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, reg_train, reg_test = train_test_split(
    X, y, regions, test_size=0.2, random_state=42
)

# --- 3. ПОСТРОЕНИЕ МОДЕЛИ (LSTM) ---

model = Sequential([
    # 1-й слой LSTM
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    
    # 2-й слой LSTM
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    
    # Выходной слой (Регрессия)
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# --- 4. ОБУЧЕНИЕ ---

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# --- 5. ОЦЕНКА МОДЕЛИ И ВИЗУАЛИЗАЦИЯ ---

# Предсказание
y_pred_scaled = model.predict(X_test)

# Обратное масштабирование (Denormalization) из [0,1] обратно в ц/га
y_test_real = scaler_target.inverse_transform(y_test.reshape(-1, 1))
y_pred_real = scaler_target.inverse_transform(y_pred_scaled)

# Метрики
mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

print(f"\n=== РЕЗУЛЬТАТЫ ===")
print(f"MAE (Средняя абсолютная ошибка): {mae:.2f} ц/га")
print(f"RMSE (Корень из среднеквадратичной ошибки): {rmse:.2f} ц/га")

# График 1: Факт vs Прогноз (Точечный график)
plt.figure(figsize=(10, 6))
plt.scatter(y_test_real, y_pred_real, color='blue', alpha=0.6)
plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--', lw=2)
plt.xlabel('Фактическая урожайность (ц/га)')
plt.ylabel('Предсказанная урожайность (ц/га)')
plt.title('Факт vs Прогноз (все регионы)')
plt.grid(True)
plt.show()

# График 2: Визуализация по регионам (Временной ряд)
# Соберем DataFrame для удобства отрисовки
results_df = pd.DataFrame({
    'Region': reg_test,
    'Actual': y_test_real.flatten(),
    'Predicted': y_pred_real.flatten()
})

plt.figure(figsize=(14, 7))
# Отсортируем для красоты (хотя индекс здесь условный)
# Для каждого региона нарисуем точки
for region in results_df['Region'].unique():
    region_data = results_df[results_df['Region'] == region]
    # Чтобы линии не прыгали хаотично, отсортируем по фактическому значению (как proxy для времени)
    region_data = region_data.sort_values('Actual') 
    
    plt.plot(region_data['Actual'].values, label=f'{region} (Факт)', marker='o', linestyle='-')
    plt.plot(region_data['Predicted'].values, label=f'{region} (Прогноз)', marker='x', linestyle='--')

plt.title('Сравнение Факта и Прогноза по регионам')
plt.ylabel('Урожайность (ц/га)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# График 3: Потери при обучении
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('График функции потерь (MSE)')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.show()
