"""
RANDOM FOREST PARA PREDECIR UN NÚMERO (paso 03)

Usa los dos archivos que crea 02_dividir_datos.py:
  - datos_entrenamiento.csv : con ellos el modelo aprende
  - datos_prueba.csv        : con ellos se examina (no se tocan antes)

Sigue el patrón del capítulo 2 de handson-ml2:
  Pipeline (preparación + RandomForestRegressor) -> entrenar -> evaluar con RMSE

IMPORTANTE: tus datos de CONAGUA no traen rendimiento de cultivos.
Para probar el código el objetivo es la lluvia total del mes.
Cuando tengas la base de cosechas, cambias OBJETIVO (y EXCLUIR).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------
# CONFIGURACIÓN (OBJETIVO debe ser igual que en 02_dividir_datos.py)
# ----------------------------------------------------------------------
ARCHIVO_ENTRENAMIENTO = "datos_entrenamiento.csv"
ARCHIVO_PRUEBA = "datos_prueba.csv"
OBJETIVO = "lluvia_total"          # el número que queremos predecir
EXCLUIR = ["lluvia_max_24h"]       # columnas que NO usamos para predecir
# (lluvia_max_24h es parte de la lluvia total del mes: si la dejáramos,
#  el modelo "haría trampa" porque ya conocería casi la respuesta)

# ----------------------------------------------------------------------
# 1. LEER LOS DATOS YA SEPARADOS POR 02_dividir_datos.py
# ----------------------------------------------------------------------
train_set = pd.read_csv(ARCHIVO_ENTRENAMIENTO)
test_set = pd.read_csv(ARCHIVO_PRUEBA)
print("Filas de entrenamiento:", len(train_set))
print("Filas de prueba:       ", len(test_set))

# Separamos, en cada grupo, los datos para predecir (X) y la respuesta (y)
X_train = train_set.drop(columns=[OBJETIVO] + EXCLUIR)
y_train = train_set[OBJETIVO].copy()
X_test = test_set.drop(columns=[OBJETIVO] + EXCLUIR)
y_test = test_set[OBJETIVO].copy()

# ----------------------------------------------------------------------
# 2. PIPELINE: preparación + Random Forest
# ----------------------------------------------------------------------
num_attribs = []
cat_attribs = []
for columna in X_train.columns:
    if columna == "mes":
        cat_attribs.append(columna)
    else:
        num_attribs.append(columna)

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler()),
])

preparacion = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_attribs),
])

# Un solo objeto que prepara los datos Y predice (como full_pipeline_with_predictor
# del capítulo 2). n_estimators=100 = un bosque de 100 árboles.
modelo = Pipeline([
    ("preparacion", preparacion),
    ("bosque", RandomForestRegressor(n_estimators=100, random_state=42)),
])

# fit = el modelo aprende. Solo ve los datos de ENTRENAMIENTO.
modelo.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 3. EVALUAR
# ----------------------------------------------------------------------
# RMSE = error típico del modelo, en las mismas unidades que el objetivo (mm).

# a) Con los mismos datos con los que aprendió (casi siempre sale muy bajo)
pred_train = modelo.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))

# b) Validación cruzada: examina al modelo con partes del entrenamiento
#    que no vio en cada vuelta (cv=10, como en el capítulo 2)
scores = cross_val_score(modelo, X_train, y_train,
                         scoring="neg_mean_squared_error", cv=10)
rmse_cv = np.sqrt(-scores)

# c) Con los datos de PRUEBA, que no se usaron para aprender (el resultado más honesto)
pred_test = modelo.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))

# d) Punto de comparación: ¿qué error tendría quien SIEMPRE respondiera
#    el promedio? Si el bosque no mejora esto, no está aprendiendo nada útil.
pred_promedio = np.full(len(y_test), y_train.mean())
rmse_promedio = np.sqrt(mean_squared_error(y_test, pred_promedio))

print("\nPredecimos:", OBJETIVO, "(promedio real:", round(y_train.mean(), 1), ")")
print("Columnas usadas para predecir:", list(X_train.columns))
print("\nRMSE en entrenamiento:          ", round(rmse_train, 1))
print("RMSE validación cruzada (media):", round(rmse_cv.mean(), 1))
print("RMSE en prueba:                 ", round(rmse_test, 1))
print("RMSE de siempre responder el promedio:", round(rmse_promedio, 1))
print("R² en prueba:", round(modelo.score(X_test, y_test), 3))

# ----------------------------------------------------------------------
# 4. EJEMPLO: comparar algunas predicciones con la realidad
# ----------------------------------------------------------------------
ejemplo = X_test.head(8).copy()
ejemplo["real"] = y_test.head(8)
ejemplo["prediccion"] = pred_test[:8].round(1)
print("\nAlgunos casos de prueba:")
print(ejemplo[["anio", "mes", "real", "prediccion"]].to_string(index=False))