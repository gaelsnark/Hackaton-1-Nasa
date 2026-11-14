"""
RANDOM FOREST PARA PREDECIR UN NÚMERO (versión de prueba)

Usa el archivo datos_limpios.csv que crea limpieza_datos_simple.py.
Sigue el patrón del capítulo 2 de handson-ml2:
  separar datos -> Pipeline (preparación + RandomForestRegressor) -> evaluar con RMSE

IMPORTANTE: tus datos de CONAGUA no traen rendimiento de cultivos.
Para probar el código usamos como objetivo la lluvia total del mes.
Cuando tengas la base de cosechas, solo cambias OBJETIVO (y la lista de EXCLUIR).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------
ARCHIVO = "datos_limpios.csv"
OBJETIVO = "lluvia_total"          # el número que queremos predecir
EXCLUIR = ["lluvia_max_24h"]       # columnas que NO usamos para predecir
# (lluvia_max_24h es parte de la lluvia total del mes: si la dejáramos,
#  el modelo "haría trampa" porque ya conocería casi la respuesta)

# ----------------------------------------------------------------------
# 1. LEER DATOS
# ----------------------------------------------------------------------
tabla = pd.read_csv(ARCHIVO)

# Al objetivo no se le pueden rellenar vacíos (sería inventar la respuesta),
# así que quitamos las filas donde falta.
tabla = tabla.dropna(subset=[OBJETIVO])
print("Filas disponibles:", len(tabla))

# ----------------------------------------------------------------------
# 2. SEPARAR ENTRENAMIENTO Y PRUEBA (train_test_split, capítulo 2)
# ----------------------------------------------------------------------
# El modelo aprende con el 80 % y se examina con el 20 % que nunca vio.
train_set, test_set = train_test_split(tabla, test_size=0.2, random_state=42)

X_train = train_set.drop(columns=[OBJETIVO] + EXCLUIR)   # datos para predecir
y_train = train_set[OBJETIVO].copy()                      # la respuesta correcta
X_test = test_set.drop(columns=[OBJETIVO] + EXCLUIR)
y_test = test_set[OBJETIVO].copy()

# ----------------------------------------------------------------------
# 3. PIPELINE: preparación + Random Forest
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

# Un solo objeto que prepara los datos Y predice (así lo hace el repo en
# full_pipeline_with_predictor). n_estimators=100 = un bosque de 100 árboles.
modelo = Pipeline([
    ("preparacion", preparacion),
    ("bosque", RandomForestRegressor(n_estimators=100, random_state=42)),
])

modelo.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 4. EVALUAR
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

# c) Con el 20 % de prueba, que no se tocó hasta ahora (el resultado más honesto)
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
print("RMSE en prueba (20 % separado): ", round(rmse_test, 1))
print("RMSE de siempre responder el promedio:", round(rmse_promedio, 1))
print("R² en prueba:", round(modelo.score(X_test, y_test), 3))

# ----------------------------------------------------------------------
# 5. EJEMPLO: comparar algunas predicciones con la realidad
# ----------------------------------------------------------------------
ejemplo = X_test.head(8).copy()
ejemplo["real"] = y_test.head(8)
ejemplo["prediccion"] = pred_test[:8].round(1)
print("\nAlgunos casos de prueba:")
print(ejemplo[["anio", "mes", "real", "prediccion"]].to_string(index=False))