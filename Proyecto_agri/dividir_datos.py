"""
SEPARAR DATOS EN ENTRENAMIENTO Y PRUEBA (train_test_split, capítulo 2)

Lee datos_limpios.csv (lo crea limpieza_datos_simple.py) y lo divide en:
  - datos_entrenamiento.csv : 80 % de las filas, con ellas el modelo aprende
  - datos_prueba.csv        : 20 % de las filas, se guardan sin tocar para
                              examinar al modelo al final

Los siguientes archivos (el del Random Forest, etc.) leerán estos dos
archivos, así todos trabajan con la misma división.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------
CARPETA_PROYECTO = Path(__file__).resolve().parent
ARCHIVO = CARPETA_PROYECTO / "datos_limpios.csv"
OBJETIVO = "lluvia_total"   # lo que queremos predecir (luego será el rendimiento)
TAMANO_PRUEBA = 0.2         # 0.2 = 20 % de las filas van a prueba
MINIMO_FILAS = 75           # con menos meses con datos no tiene sentido entrenar (75 meses = 6 años)

# ----------------------------------------------------------------------
# 1. LEER DATOS
# ----------------------------------------------------------------------
tabla = pd.read_csv(ARCHIVO)
print("Filas totales:", len(tabla))

# Si esta estación no trae la columna que queremos predecir, no se puede seguir
if OBJETIVO not in tabla.columns:
  raise SystemExit(
    f"No existe la columna {OBJETIVO} en {ARCHIVO}. Columnas que sí hay: "
    f"{', '.join(tabla.columns)}. Prueba con otra estación o cambia OBJETIVO."
  )

# Si a una fila le falta el valor que queremos predecir, no sirve ni para
# aprender ni para examinar, así que se quita (no se puede rellenar: sería
# inventar la respuesta).
filas_antes = len(tabla)
tabla = tabla.dropna(subset=[OBJETIVO])
print("Filas sin", OBJETIVO, "eliminadas:", filas_antes - len(tabla))

# Con muy pocos meses, el modelo no tiene de dónde aprender ni con qué examinarse
if len(tabla) < MINIMO_FILAS:
    raise SystemExit("Solo hay " + str(len(tabla)) + " meses con " + OBJETIVO + " (se necesitan al menos " +
                     str(MINIMO_FILAS) + "). Con tan poco historial el modelo no puede aprender ni "
                     "examinarse con sentido. Prueba con una estación con más años de datos.")

# ----------------------------------------------------------------------
# 2. SEPARAR (igual que en el capítulo 2 del repo)
# ----------------------------------------------------------------------
# train_test_split revuelve las filas al azar y corta: el 80 % es
# entrenamiento y el 20 % es prueba.
# random_state=42 fija el "azar": cada vez que lo corras, la división sale igual.
train_set, test_set = train_test_split(tabla, test_size=TAMANO_PRUEBA, random_state=42)

# ----------------------------------------------------------------------
# 3. REVISAR QUE LA DIVISIÓN TENGA SENTIDO
# ----------------------------------------------------------------------
print("\nFilas de entrenamiento:", len(train_set))
print("Filas de prueba:       ", len(test_set))

print("\nPromedio de", OBJETIVO, "(deberían ser parecidos):")
print("  entrenamiento:", round(train_set[OBJETIVO].mean(), 1))
print("  prueba:       ", round(test_set[OBJETIVO].mean(), 1))

print("\nRango de años:")
print("  entrenamiento:", train_set["anio"].min(), "a", train_set["anio"].max())
print("  prueba:       ", test_set["anio"].min(), "a", test_set["anio"].max())

# ----------------------------------------------------------------------
# 4. GUARDAR
# ----------------------------------------------------------------------
train_set.to_csv(CARPETA_PROYECTO / "datos_entrenamiento.csv", index=False)
test_set.to_csv(CARPETA_PROYECTO / "datos_prueba.csv", index=False)
print("\nGuardados: datos_entrenamiento.csv y datos_prueba.csv")
