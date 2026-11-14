"""
LIMPIEZA Y PREPARACIÓN DE DATOS (versión sencilla)

Lee mes22001.txt (CONAGUA), lo limpia y lo prepara con
Pipeline + ColumnTransformer (patrón del capítulo 2 de handson-ml2).

Uso: pon este archivo en la misma carpeta que mes22001.txt y ejecútalo.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------
ARCHIVO = Path(__file__).with_name("mes22001.txt")
MAX_FALTANTES = 60   # si a una columna le falta más de este porcentaje, se elimina

MESES = ["ENE", "FEB", "MAR", "ABR", "MAY", "JUN",
         "JUL", "AGO", "SEP", "OCT", "NOV", "DIC"]

# Título de cada tabla dentro del .txt  ->  nombre de columna que usaremos
TABLAS = {
    "LLUVIA MÁXIMA 24 H.":         "lluvia_max_24h",
    "LLUVIA TOTAL MENSUAL":        "lluvia_total",
    "EVAPORACIÓN MENSUAL":         "evaporacion",
    "TEMPERATURA MÁXIMA PROMEDIO": "temp_max_prom",
    "TEMPERATURA MÁXIMA EXTREMA":  "temp_max_ext",
    "TEMPERATURA MÍNIMA PROMEDIO": "temp_min_prom",
    "TEMPERATURA MÍNIMA EXTREMA":  "temp_min_ext",
    "TEMPERATURA MEDIA MENSUAL":   "temp_media",
}
nombres_columnas = list(TABLAS.values())

# ----------------------------------------------------------------------
# 1. LEER EL .TXT
# ----------------------------------------------------------------------
# open() abre el archivo y readlines() lo convierte en una lista de textos,
# uno por cada línea. (Si te sale error de acentos, cambia "utf-8" por "latin-1")
with open(ARCHIVO, encoding="utf-8") as archivo:
    lineas = archivo.readlines()

datos = {}       # guardará: (año, número de mes) -> {nombre de variable: valor}
variable = ""    # nombre de la tabla que estamos leyendo en este momento

for linea in lineas:
    texto = linea.strip()           # la línea sin espacios ni salto de línea
    partes = linea.split("\t")      # corta la línea en cada tabulador
    primera = partes[0].strip()     # lo que hay en la primera columna

    if texto in TABLAS:
        # Esta línea es el título de una tabla nueva
        variable = TABLAS[texto]

    elif variable != "" and primera.isdigit() and len(primera) == 4:
        # Esta línea empieza con un año de 4 dígitos (ej. 1944): es una fila de datos.
        # Las filas MÍNIMA, MÁXIMA, MEDIA y DESV.ST no pasan esta prueba.
        while len(partes) < 13:     # por si la línea viene corta, la completamos
            partes.append("")
        anio = int(primera)

        for i in range(12):         # los 12 meses
            celda = partes[i + 1].strip()
            if celda == "":
                valor = np.nan      # nan = celda vacía
            else:
                valor = float(celda)

            clave = (anio, i)
            if clave not in datos:
                datos[clave] = {}
            datos[clave][variable] = valor

# ----------------------------------------------------------------------
# 2. CONVERTIR A UNA TABLA (una fila por año-mes, una columna por variable)
# ----------------------------------------------------------------------
filas = []
for clave in sorted(datos):         # sorted() los ordena por año y luego por mes
    anio, num_mes = clave
    fila = [anio, MESES[num_mes]]
    for nombre in nombres_columnas:
        # .get() busca la variable; si no existe para ese mes, usa nan
        fila.append(datos[clave].get(nombre, np.nan))
    filas.append(fila)

# DataFrame = tabla de pandas (como una hoja de Excel). Lo usamos porque
# ColumnTransformer necesita elegir las columnas por su nombre.
tabla = pd.DataFrame(filas, columns=["anio", "mes"] + nombres_columnas)
print("Tabla leída:", tabla.shape[0], "filas x", tabla.shape[1], "columnas")

# ----------------------------------------------------------------------
# 3. LIMPIEZA
# ----------------------------------------------------------------------
# a) Quitar las filas donde NINGUNA variable tiene dato
filas_antes = len(tabla)
tabla = tabla.dropna(subset=nombres_columnas, how="all")
tabla = tabla.reset_index(drop=True)
print("Filas sin ninguna medición eliminadas:", filas_antes - len(tabla))

# b) Quitar las columnas a las que les falta demasiado
columnas_quitar = []
print("Porcentaje de datos que faltan en cada columna:")
for columna in nombres_columnas:
    porcentaje = tabla[columna].isna().sum() / len(tabla) * 100
    print("  ", columna, ":", round(porcentaje, 1), "%")
    if porcentaje > MAX_FALTANTES:
        columnas_quitar.append(columna)
 
tabla = tabla.drop(columns=columnas_quitar)
print("Columnas eliminadas por faltar demasiados datos:", columnas_quitar)

# ----------------------------------------------------------------------
# 4. PREPARACIÓN: Pipeline + ColumnTransformer (capítulo 2 del repo)
# ----------------------------------------------------------------------
# Columnas numéricas: el año y las mediciones que se quedaron
num_attribs = ["anio"]
for columna in nombres_columnas:
    if columna not in columnas_quitar:
        num_attribs.append(columna)

# Columnas de texto: el mes
cat_attribs = ["mes"]

# Pipeline = pasos en orden. Para los números:
#   1) rellenar vacíos con la mediana   2) poner todo en la misma escala
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler()),
])

# ColumnTransformer = a cada grupo de columnas le aplica su tratamiento:
#   números -> num_pipeline        mes (texto) -> OneHotEncoder
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_attribs),
])

matriz = full_pipeline.fit_transform(tabla)
preparado = pd.DataFrame(matriz, columns=full_pipeline.get_feature_names_out())

# ----------------------------------------------------------------------
# 5. GUARDAR
# ----------------------------------------------------------------------
tabla.to_csv("datos_limpios.csv", index=False)
preparado.to_csv("datos_preparados.csv", index=False)

print("Datos limpios:    ", tabla.shape, "-> datos_limpios.csv")
print("Datos preparados: ", preparado.shape, "-> datos_preparados.csv")
print("Vacíos que quedan tras el pipeline:", int(preparado.isna().sum().sum()))