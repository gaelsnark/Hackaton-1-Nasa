"""
05 - AJUSTAR EL MODELO (GridSearchCV y RandomizedSearchCV, capítulo 2)

Un Random Forest tiene "perillas" que tú eliges ANTES de entrenarlo (se llaman
hiperparámetros): cuántos árboles, cuántas columnas ve cada árbol, qué tan
chicos pueden ser los grupos finales... Este script prueba combinaciones de
perillas y se queda con la mejor.

Cada combinación se evalúa con validación cruzada usando SOLO los datos de
ENTRENAMIENTO. Al final, el modelo ajustado se examina con los datos de PRUEBA
(que no se usaron para elegir nada) y se compara con el modelo sin ajustar.

Qué modelo ajusta (cambia TIPO):
  "regresion"     -> el de 03 (predice un número)   lee los CSV de 02_dividir_datos.py
  "clasificacion" -> el de 04 (bueno / malo)        lee los CSV de 04_clasificador_forest.py

Cómo busca (cambia METODO):
  "grid"      -> GridSearchCV: prueba TODAS las combinaciones de unas listas
                 (más completo, más lento)
  "aleatorio" -> RandomizedSearchCV: prueba combinaciones al azar dentro de unos
                 rangos (más rápido; en el repo se usa cuando hay muchas opciones)

Funciona con cualquier estación: no tiene ningún dato de una región escrito.
Puede tardar uno o dos minutos: entrena el bosque muchas veces.
"""
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------
TIPO = "regresion"          # "regresion" o "clasificacion"
METODO = "grid"             # "grid" o "aleatorio"
OBJETIVO = "lluvia_total"   # solo para "regresion": igual que en 02 y 03
EXCLUIR = ["lluvia_max_24h"]  # solo para "regresion": igual que en 03
VALIDACIONES = 5            # en cuántas partes se divide el entrenamiento (cv=5, como en el repo)
PRUEBAS_ALEATORIO = 20      # cuántas combinaciones prueba "aleatorio"
MINIMO_FILAS = 60           # con menos filas de entrenamiento no tiene sentido ajustar
CARPETA_PROYECTO = Path(__file__).resolve().parent

if TIPO not in ("regresion", "clasificacion"):
    raise SystemExit('TIPO debe ser "regresion" o "clasificacion".')
if METODO not in ("grid", "aleatorio"):
    raise SystemExit('METODO debe ser "grid" o "aleatorio".')

# ----------------------------------------------------------------------
# 1. LEER LOS DATOS (ya separados en entrenamiento y prueba)
# ----------------------------------------------------------------------
if TIPO == "regresion":
    archivo_train = CARPETA_PROYECTO / "datos_entrenamiento.csv"
    archivo_test = CARPETA_PROYECTO / "datos_prueba.csv"
    paso_previo = "02_dividir_datos.py"
else:
    archivo_train = CARPETA_PROYECTO / "datos_entrenamiento_clasificacion.csv"
    archivo_test = CARPETA_PROYECTO / "datos_prueba_clasificacion.csv"
    paso_previo = "04_clasificador_forest.py"

for archivo in [archivo_train, archivo_test]:
    if not archivo.is_file():
        raise SystemExit(f"No encuentro {archivo}. Primero corre {paso_previo} en Proyecto_agri.")

train_set = pd.read_csv(archivo_train)
test_set = pd.read_csv(archivo_test)
print("Filas de entrenamiento:", len(train_set))
print("Filas de prueba:       ", len(test_set))

if len(train_set) < MINIMO_FILAS:
    raise SystemExit("Hay muy pocas filas de entrenamiento (" + str(len(train_set)) +
                     "). Prueba con una estación que tenga más años de historial.")

# Separamos los datos para predecir (X) y la respuesta (y)
if TIPO == "regresion":
    X_train = train_set.drop(columns=[OBJETIVO] + EXCLUIR, errors="ignore")
    y_train = train_set[OBJETIVO].copy()
    X_test = test_set.drop(columns=[OBJETIVO] + EXCLUIR, errors="ignore")
    y_test = test_set[OBJETIVO].copy()
else:
    X_train = train_set.drop(columns=["categoria"])
    y_train = train_set["categoria"].copy()
    X_test = test_set.drop(columns=["categoria"])
    y_test = test_set["categoria"].copy()

# ----------------------------------------------------------------------
# 2. EL PIPELINE (igual que en 03 y 04): preparación + Random Forest
# ----------------------------------------------------------------------
num_attribs = []
cat_attribs = []
for columna in X_train.columns:
    if columna == "mes":
        cat_attribs.append(columna)
    else:
        num_attribs.append(columna)


def crear_modelo(bosque):
    """Arma el Pipeline completo: preparación de los datos + el bosque que le pases."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("std_scaler", StandardScaler()),
    ])
    preparacion = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_attribs),
    ])
    return Pipeline([("preparacion", preparacion), ("bosque", bosque)])


if TIPO == "regresion":
    modelo = crear_modelo(RandomForestRegressor(random_state=42))
    puntuacion = "neg_mean_squared_error"   # como en el capítulo 2 (después se saca la raíz = RMSE)
else:
    modelo = crear_modelo(RandomForestClassifier(random_state=42))
    puntuacion = "f1_macro"                 # promedio del F1 de "bueno" y de "malo"

# ----------------------------------------------------------------------
# 3. LAS PERILLAS QUE VAMOS A PROBAR
# El nombre lleva "bosque__" porque son perillas del paso "bosque" del Pipeline
# (así se hace también en el capítulo 2: "preparation__num__imputer__strategy").
# ----------------------------------------------------------------------
# Para GridSearchCV: listas de valores. Prueba TODAS las combinaciones.
param_grid = {
    "bosque__n_estimators": [50, 100, 200],          # cuántos árboles tiene el bosque
    "bosque__max_features": [0.2, 0.4, 0.7, 1.0],    # fracción de columnas que ve cada árbol
    "bosque__min_samples_leaf": [1, 3, 5],           # mínimo de datos por hoja (más alto = árboles más simples)
}

# Para RandomizedSearchCV: rangos. Saca valores al azar de cada rango.
param_distribs = {
    "bosque__n_estimators": randint(low=50, high=300),       # enteros de 50 a 299
    "bosque__max_features": uniform(loc=0.2, scale=0.8),     # decimales entre 0.2 y 1.0
    "bosque__min_samples_leaf": randint(low=1, high=11),     # enteros de 1 a 10
}

# ----------------------------------------------------------------------
# 4. BUSCAR LA MEJOR COMBINACIÓN
# ----------------------------------------------------------------------
if METODO == "grid":
    busqueda = GridSearchCV(modelo, param_grid, cv=VALIDACIONES, scoring=puntuacion)
    combinaciones = 1
    for nombre in param_grid:
        combinaciones = combinaciones * len(param_grid[nombre])
    print("\nMétodo: GridSearchCV. Valores que se prueban:")
    for nombre in param_grid:
        print("  ", nombre.replace("bosque__", ""), ":", param_grid[nombre])
else:
    busqueda = RandomizedSearchCV(modelo, param_distribs, n_iter=PRUEBAS_ALEATORIO,
                                  cv=VALIDACIONES, scoring=puntuacion, random_state=42)
    combinaciones = PRUEBAS_ALEATORIO
    print("\nMétodo: RandomizedSearchCV. Rangos de los que se sortea:")
    print("   n_estimators: enteros de 50 a 299")
    print("   max_features: decimales entre 0.2 y 1.0")
    print("   min_samples_leaf: enteros de 1 a 10")

print("Combinaciones:", combinaciones, "x", VALIDACIONES, "validaciones =",
      combinaciones * VALIDACIONES, "entrenamientos del bosque. Espera un momento...")

inicio = time.time()
busqueda.fit(X_train, y_train)     # prueba todo y, al final, re-entrena la mejor con TODO el entrenamiento
print("Tardó", round(time.time() - inicio), "segundos.")


def texto_params(params):
    """Convierte {'bosque__n_estimators': 100, ...} en 'n_estimators=100, ...'"""
    partes = []
    for nombre in params:
        valor = params[nombre]
        if isinstance(valor, float):
            valor = round(valor, 2)
        partes.append(nombre.replace("bosque__", "") + "=" + str(valor))
    return ", ".join(partes)


def legible(puntaje):
    """El puntaje de regresión viene en negativo y al cuadrado: lo pasamos a RMSE."""
    if TIPO == "regresion":
        return np.sqrt(-puntaje)
    return puntaje


nombre_medida = "RMSE" if TIPO == "regresion" else "F1"
print("\nMejores valores encontrados:", texto_params(busqueda.best_params_))
print(nombre_medida, "con validación cruzada (solo entrenamiento):",
      round(legible(busqueda.best_score_), 3))

resultados = busqueda.cv_results_
orden = np.argsort(resultados["rank_test_score"])      # de la mejor a la peor
print("\nLas 5 mejores combinaciones (" + nombre_medida + "; en regresión menos es mejor):")
for i in orden[:5]:
    print("  ", round(legible(resultados["mean_test_score"][i]), 3), "|", texto_params(resultados["params"][i]))

# ----------------------------------------------------------------------
# 5. EXAMEN FINAL CON LOS DATOS DE PRUEBA
# Se compara el bosque sin ajustar (valores por defecto, como en 03 y 04)
# con el ajustado. Los datos de prueba no se usaron para elegir nada.
# ----------------------------------------------------------------------
if TIPO == "regresion":
    modelo_base = crear_modelo(RandomForestRegressor(n_estimators=100, random_state=42))
else:
    modelo_base = crear_modelo(RandomForestClassifier(n_estimators=100, random_state=42))
modelo_base.fit(X_train, y_train)

modelo_ajustado = busqueda.best_estimator_

pred_base = modelo_base.predict(X_test)
pred_ajustado = modelo_ajustado.predict(X_test)

print("\nResultado en PRUEBA (datos que no se usaron para elegir nada):")
if TIPO == "regresion":
    rmse_base = np.sqrt(mean_squared_error(y_test, pred_base))
    rmse_ajustado = np.sqrt(mean_squared_error(y_test, pred_ajustado))
    rmse_promedio = np.sqrt(mean_squared_error(y_test, np.full(len(y_test), y_train.mean())))
    print("  Modelo sin ajustar: RMSE =", round(rmse_base, 1), "| R² =", round(modelo_base.score(X_test, y_test), 3))
    print("  Modelo ajustado:    RMSE =", round(rmse_ajustado, 1), "| R² =", round(modelo_ajustado.score(X_test, y_test), 3))
    print("  Siempre responder el promedio: RMSE =", round(rmse_promedio, 1))
    mejora = (rmse_base - rmse_ajustado) / rmse_base * 100     # % que bajó el error
    diferencia_texto = str(round(mejora, 1)) + " % menos error"
    pequena = abs(mejora) < 3
    peor = mejora <= -3
else:
    exac_base = accuracy_score(y_test, pred_base)
    exac_ajustado = accuracy_score(y_test, pred_ajustado)
    exac_adivinar = max(np.mean(y_test == "bueno"), 1 - np.mean(y_test == "bueno"))
    print("  Modelo sin ajustar: exactitud =", round(exac_base * 100, 1), "% | F1 =",
          round(f1_score(y_test, pred_base, average="macro") * 100, 1), "%")
    print("  Modelo ajustado:    exactitud =", round(exac_ajustado * 100, 1), "% | F1 =",
          round(f1_score(y_test, pred_ajustado, average="macro") * 100, 1), "%")
    print("  Siempre decir lo más común: exactitud =", round(exac_adivinar * 100, 1), "%")
    mejora = (exac_ajustado - exac_base) * 100                 # puntos porcentuales
    diferencia_texto = str(round(mejora, 1)) + " puntos de exactitud"
    pequena = abs(mejora) < 2
    peor = mejora <= -2

if peor:
    print("\nConclusión: en la prueba el modelo ajustado salió peor (" + diferencia_texto + "). Puede pasar:")
    print("la validación cruzada elige lo mejor para el entrenamiento, pero no garantiza ganar en la prueba.")
    print("Con pocos datos, usa el modelo sin ajustar o amplía las listas de valores.")
elif pequena:
    print("\nConclusión: casi no hay diferencia (" + diferencia_texto + "). Con pocos datos de prueba eso puede ser")
    print("ruido. Es normal: los valores por defecto del Random Forest ya funcionan bien.")
else:
    print("\nConclusión: el ajuste mejoró el resultado en la prueba (" + diferencia_texto + ").")

# ----------------------------------------------------------------------
# 6. GUARDAR EL MODELO AJUSTADO (como al final del capítulo 2 con joblib)
# Para usarlo después sin volver a entrenar:  modelo = joblib.load("archivo.pkl")
# ----------------------------------------------------------------------
nombre_archivo = "modelo_ajustado_" + TIPO + ".pkl"
ruta_modelo = CARPETA_PROYECTO / nombre_archivo
joblib.dump(modelo_ajustado, ruta_modelo)
print("\nModelo ajustado guardado en", ruta_modelo)
