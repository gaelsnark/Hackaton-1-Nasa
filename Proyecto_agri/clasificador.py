"""
04 - RANDOM FOREST PARA PREDECIR UNA CATEGORÍA (bueno / malo)

Usa los archivos que crea dividir_datos.py:
  datos_entrenamiento.csv  y  datos_prueba.csv

Qué hace:
  1. Convierte un número en categoría ("bueno" o "malo") con una regla
  2. Entrena un RandomForestClassifier (capítulos 3 y 7 de handson-ml2)
  3. Lo evalúa con datos que no vio y lo compara contra "adivinar"
  4. Guarda los datos con su categoría para el paso 05 (ajuste del modelo)

IMPORTANTE: los datos de CONAGUA no traen rendimiento de cultivos, así que
la regla de "bueno/malo" es de PRUEBA y se basa en la lluvia del mes.
Cuando tengas la base de cosechas, cambia COLUMNA_BASE (por ejemplo a
"rendimiento") y usa CRITERIO = "mediana" o "fijo".

Funciona con cualquier estación: no tiene ningún dato de Amealco escrito
(los umbrales se calculan con los datos de cada estación).
"""
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Trabajar siempre en la carpeta donde está ESTE archivo: así lee y guarda ahí
# aunque lo ejecutes desde otra carpeta (por ejemplo desde la raíz del proyecto).
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------
ARCHIVO_ENTRENAMIENTO = "datos_entrenamiento.csv"
ARCHIVO_PRUEBA = "datos_prueba.csv"

COLUMNA_BASE = "lluvia_total"     # el número con el que se decide si es bueno o malo
EXCLUIR = ["lluvia_max_24h"]      # columnas que NO se usan para predecir
# (lluvia_max_24h es parte de la lluvia total: si la dejáramos, el modelo "haría trampa")

# Cómo se decide "bueno":
#   "por_mes"  -> bueno si la lluvia del mes SUPERA la mediana de ese mismo mes
#                 (así no basta con saber que en verano llueve más)
#   "mediana"  -> bueno si supera la mediana de todos los datos de entrenamiento
#   "fijo"     -> bueno si llega al menos a UMBRAL_FIJO
CRITERIO = "por_mes"
UMBRAL_FIJO = 50                  # solo se usa si CRITERIO = "fijo" (mismas unidades que COLUMNA_BASE)

MIN_DATOS_POR_MES = 5             # si un mes tiene menos datos que esto, se usa la mediana general
MINIMO_FILAS = 60                 # con menos filas de entrenamiento no tiene sentido entrenar

MESES = ["ENE", "FEB", "MAR", "ABR", "MAY", "JUN",
         "JUL", "AGO", "SEP", "OCT", "NOV", "DIC"]

# ----------------------------------------------------------------------
# 1. LEER LOS DATOS YA SEPARADOS POR 02
# ----------------------------------------------------------------------
train_set = pd.read_csv(ARCHIVO_ENTRENAMIENTO)
test_set = pd.read_csv(ARCHIVO_PRUEBA)

# Si a una fila le falta el número con el que se decide, no se le puede poner categoría
train_set = train_set.dropna(subset=[COLUMNA_BASE]).copy()
test_set = test_set.dropna(subset=[COLUMNA_BASE]).copy()
print("Filas de entrenamiento:", len(train_set))
print("Filas de prueba:       ", len(test_set))

if len(train_set) < MINIMO_FILAS:
    raise SystemExit("Hay muy pocos datos de entrenamiento (" + str(len(train_set)) +
                     "). Prueba con una estación que tenga más años de historial.")
if CRITERIO == "por_mes" and "mes" not in train_set.columns:
    raise SystemExit('Tu tabla no tiene columna "mes". Usa CRITERIO = "mediana" o "fijo".')

# ----------------------------------------------------------------------
# 2. PONER LA CATEGORÍA (bueno / malo)
# Los umbrales se calculan SOLO con los datos de entrenamiento, para que el
# examen (prueba) no influya en cómo se define lo que es "bueno".
# ----------------------------------------------------------------------
mediana_general = train_set[COLUMNA_BASE].median()

umbral_mes = {}
for mes in MESES:
    valores = train_set[train_set["mes"] == mes][COLUMNA_BASE] if "mes" in train_set.columns else []
    if len(valores) >= MIN_DATOS_POR_MES:
        umbral_mes[mes] = valores.median()
    else:
        umbral_mes[mes] = mediana_general

for tabla in [train_set, test_set]:
    categorias = []
    for i in range(len(tabla)):
        valor = tabla[COLUMNA_BASE].iloc[i]
        if CRITERIO == "por_mes":
            bueno = valor > umbral_mes[tabla["mes"].iloc[i]]   # empatar con la mediana cuenta como malo
        elif CRITERIO == "mediana":
            bueno = valor > mediana_general
        else:
            bueno = valor >= UMBRAL_FIJO                       # "al menos" el umbral
        if bueno:
            categorias.append("bueno")
        else:
            categorias.append("malo")
    tabla["categoria"] = categorias

print("\nCriterio:", CRITERIO, "sobre", COLUMNA_BASE)
if CRITERIO == "por_mes":
    print("Umbral de cada mes (lo que hay que superar para ser 'bueno'):")
    for mes in MESES:
        print("  ", mes, ":", round(umbral_mes[mes], 1))
elif CRITERIO == "mediana":
    print("Umbral:", round(mediana_general, 1))
else:
    print("Umbral:", UMBRAL_FIJO)

# Revisar que haya de las dos categorías (si no, no hay nada que aprender)
n_buenos = int((train_set["categoria"] == "bueno").sum())
n_malos = int((train_set["categoria"] == "malo").sum())
print("\nEn entrenamiento: bueno =", n_buenos, "| malo =", n_malos)
if min(n_buenos, n_malos) < 10:
    raise SystemExit("Casi todo cae en una sola categoría; cambia CRITERIO o el umbral.")
if min(n_buenos, n_malos) / len(train_set) < 0.2:
    print("AVISO: una categoría es muy poco frecuente (<20 %). Mira con cuidado precisión y recall.")

# ----------------------------------------------------------------------
# 3. SEPARAR DATOS PARA PREDECIR (X) Y RESPUESTA (y)
# ----------------------------------------------------------------------
# errors="ignore": si alguna columna de EXCLUIR no existe en esta estación, no pasa nada
X_train = train_set.drop(columns=[COLUMNA_BASE, "categoria"] + EXCLUIR, errors="ignore")
y_train = train_set["categoria"].copy()
X_test = test_set.drop(columns=[COLUMNA_BASE, "categoria"] + EXCLUIR, errors="ignore")
y_test = test_set["categoria"].copy()

# ----------------------------------------------------------------------
# 4. PIPELINE: preparación + Random Forest (capítulo 2 + capítulo 3)
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

modelo = Pipeline([
    ("preparacion", preparacion),
    ("bosque", RandomForestClassifier(n_estimators=100, random_state=42)),
])

modelo.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 5. EVALUAR
# ----------------------------------------------------------------------
# a) Con los datos con los que aprendió (casi siempre sale altísimo: no te fíes)
pred_train = modelo.predict(X_train)
exactitud_train = np.mean(pred_train == y_train)

# b) Validación cruzada sobre el entrenamiento (cross_val_predict, capítulo 3)
pred_cv = cross_val_predict(modelo, X_train, y_train, cv=5)
exactitud_cv = np.mean(pred_cv == y_train)

# c) Con los datos de PRUEBA (el resultado más honesto)
pred_test = modelo.predict(X_test)
exactitud_test = np.mean(pred_test == y_test)
precision = precision_score(y_test, pred_test, pos_label="bueno")
recall = recall_score(y_test, pred_test, pos_label="bueno")
f1 = f1_score(y_test, pred_test, pos_label="bueno")

# d) Punto de comparación: ¿y si SIEMPRE se respondiera la categoría más frecuente?
proporcion_buenos = np.mean(y_test == "bueno")
exactitud_adivinar = max(proporcion_buenos, 1 - proporcion_buenos)

print("\nExactitud = de cada 100 casos, cuántos clasifica bien.")
print("Exactitud en entrenamiento:          ", round(exactitud_train * 100, 1), "%")
print("Exactitud validación cruzada:        ", round(exactitud_cv * 100, 1), "%")
print("Exactitud en prueba:                 ", round(exactitud_test * 100, 1), "%")
print("Exactitud de siempre decir lo más común:", round(exactitud_adivinar * 100, 1), "%")

print("\nSobre la categoría 'bueno' (en prueba):")
print("  Precisión:", round(precision * 100, 1), "% (cuando dice 'bueno', acierta tanto)")
print("  Recall:   ", round(recall * 100, 1), "% (de los casos realmente buenos, encuentra tantos)")
print("  F1:       ", round(f1 * 100, 1), "% (resumen de los dos)")

matriz = confusion_matrix(y_test, pred_test, labels=["malo", "bueno"])
print("\nMatriz de confusión (filas = lo real, columnas = lo que dijo el modelo):")
print(pd.DataFrame(matriz, index=["real malo", "real bueno"],
                   columns=["dijo malo", "dijo bueno"]))

# ----------------------------------------------------------------------
# 6. EJEMPLO: casos de prueba con la probabilidad de "bueno"
# ----------------------------------------------------------------------
probabilidades = modelo.predict_proba(X_test)
columna_bueno = list(modelo.classes_).index("bueno")

ejemplo = X_test.head(8).copy()
ejemplo["real"] = y_test.head(8)
ejemplo["prediccion"] = pred_test[:8]
ejemplo["prob_bueno_%"] = (probabilidades[:8, columna_bueno] * 100).round(0)
columnas_ejemplo = []
for columna in ["anio", "mes", "real", "prediccion", "prob_bueno_%"]:
    if columna in ejemplo.columns:
        columnas_ejemplo.append(columna)
print("\nAlgunos casos de prueba:")
print(ejemplo[columnas_ejemplo].to_string(index=False))

# ----------------------------------------------------------------------
# 7. GUARDAR los datos con su categoría (los usa ajustar_model.py)
# ----------------------------------------------------------------------
entrenamiento_clasif = X_train.copy()
entrenamiento_clasif["categoria"] = y_train
prueba_clasif = X_test.copy()
prueba_clasif["categoria"] = y_test
entrenamiento_clasif.to_csv("datos_entrenamiento_clasificacion.csv", index=False)
prueba_clasif.to_csv("datos_prueba_clasificacion.csv", index=False)
print("\nGuardados: datos_entrenamiento_clasificacion.csv y datos_prueba_clasificacion.csv")
