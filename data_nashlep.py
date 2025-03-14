import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from scipy import stats

# Загружаем данные
data = pd.read_csv("path_to_your_dataset.csv")  # Путь к датасету

# 1. Проверка на пропущенные значения
print(data.isnull().sum())

# 2. Обработка пропущенных значений
# Применяем медиану для числовых столбцов, т.к. это не будет искажать данные
imputer = SimpleImputer(strategy="median")
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 3. Выявление и удаление выбросов (например, с помощью Z-оценки)
z_scores = np.abs(stats.zscore(data_imputed.select_dtypes(include=np.number)))
data_no_outliers = data_imputed[(z_scores < 3).all(axis=1)]  # Оставляем только строки без выбросов

# 4. Кодирование категориальных признаков (если есть такие)
# Предположим, что целевая переменная "class" является категориальной
label_encoder = LabelEncoder()
data_no_outliers['class'] = label_encoder.fit_transform(data_no_outliers['class'])

# 5. Применение Isolation Forest для обнаружения аномалий и их удаления
iso_forest = IsolationForest(contamination=0.05)  # Указываем уровень аномалий
anomalies = iso_forest.fit_predict(data_no_outliers.drop('class', axis=1))

# Удаляем аномальные записи
data_cleaned = data_no_outliers[anomalies == 1]  # Оставляем только нормальные данные
print(f"После очистки осталось строк: {data_cleaned.shape[0]}")


# 1. Преобразуем целевую переменную для бинарной классификации
data_cleaned['class_binary'] = data_cleaned['class'].apply(lambda x: 0 if x == 0 else 1)  # 0 - benign, 1 - malicious

# 2. Разделение на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split
X = data_cleaned.drop(['class', 'class_binary'], axis=1)  # Признаки
y_binary = data_cleaned['class_binary']  # Целевая переменная
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# 3. Обучение модели XGBoost
import xgboost as xgb
model_xgb = xgb.XGBClassifier(scale_pos_weight=1)  # Для балансировки классов
model_xgb.fit(X_train, y_train)

# 4. Оценка модели
from sklearn.metrics import accuracy_score, classification_report
y_pred = model_xgb.predict(X_test)
print("Точность модели XGBoost для бинарной классификации:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# 1. Преобразуем целевую переменную для мультиклассовой классификации
y_multiclass = data_cleaned['class']  # Это переменная с 11 классами

# 2. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y_multiclass, test_size=0.2, random_state=42)

# 3. Обучение модели XGBoost для мультиклассовой классификации
model_xgb_multiclass = xgb.XGBClassifier(objective='multi:softmax', num_class=11)
model_xgb_multiclass.fit(X_train, y_train)

# 4. Оценка модели
y_pred_multiclass = model_xgb_multiclass.predict(X_test)
print("Точность модели XGBoost для мультиклассовой классификации:", accuracy_score(y_test, y_pred_multiclass))
print(classification_report(y_test, y_pred_multiclass))



# 1. Используем данные только для "benign" (нормальные данные)
benign_data = data_cleaned[data_cleaned['class_binary'] == 0].drop(['class', 'class_binary'], axis=1)

# 2. Обучение модели Isolation Forest
iso_forest_anomalies = IsolationForest(contamination=0.05)
anomalies_benign = iso_forest_anomalies.fit_predict(benign_data)

# 3. Определим аномалии на тестовом наборе (все данные)
anomalies_all = iso_forest_anomalies.predict(X_test)

# 4. Оценка
print("Обнаружены аномалии в тестовом наборе:", np.sum(anomalies_all == -1))
