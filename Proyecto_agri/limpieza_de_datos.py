"""
LIMPIEZA Y PREPARACIÓN DE DATOS (versión sencilla)

Lee el archivo MENSUAL de una estación de CONAGUA (por ejemplo mes22001.txt),
lo limpia y lo prepara con Pipeline + ColumnTransformer (patrón del capítulo 2
de handson-ml2).

Uso: pon este archivo en la misma carpeta que el .txt de la estación y ejecútalo.
Para otra región: descarga el archivo MENSUAL de esa estación (su nombre empieza
con "mes", por ejemplo mes22016.txt), ponlo en la misma carpeta y cambia ARCHIVO.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------
CARPETA_PROYECTO = Path(__file__).resolve().parent
ARCHIVO = CARPETA_PROYECTO / "mes22001.txt"  # archivo MENSUAL de la estación (cámbialo para usar otra región)
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
# Primero revisamos que el archivo de datos exista junto al script
if not ARCHIVO.is_file():
    raise SystemExit(f"No encuentro el archivo de datos: {ARCHIVO}")

# open() abre el archivo y readlines() lo convierte en una lista de textos,
# uno por cada línea. Los acentos pueden venir en dos formatos: probamos "utf-8"
# y, si falla, "latin-1". (try = "intenta esto"; si da ese error, hace lo de except)
try:
    with open(ARCHIVO, encoding="utf-8") as archivo:
        lineas = archivo.readlines()
except UnicodeDecodeError:
    with open(ARCHIVO, encoding="latin-1") as archivo:
        lineas = archivo.readlines()

datos = {}                # guardará: (año, número de mes) -> {nombre de variable: valor}
variable = ""             # nombre de la tabla que estamos leyendo en este momento
tablas_encontradas = []   # tablas (lluvia, temperaturas...) que sí aparecen en el archivo
celdas_raras = 0          # celdas que traen texto en lugar de un número

for linea in lineas:
    texto = linea.strip()           # la línea sin espacios ni salto de línea
    partes = linea.split("\t")      # corta la línea en cada tabulador
    primera = partes[0].strip()     # lo que hay en la primera columna

    if texto in TABLAS:
        # Esta línea es el título de una tabla nueva
        variable = TABLAS[texto]
        tablas_encontradas.append(variable)

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
                try:
                    valor = float(celda)
                except ValueError:
                    valor = np.nan  # texto raro (por ejemplo "S/D"): se toma como vacío
                    celdas_raras = celdas_raras + 1

            clave = (anio, i)
            if clave not in datos:
                datos[clave] = {}
            datos[clave][variable] = valor

# Revisar qué se encontró: así notas enseguida si el archivo no es del tipo correcto
if len(datos) == 0:
    raise SystemExit(
        f"No encontré ninguna tabla de datos en {ARCHIVO}. ¿Es el archivo "
        "MENSUAL ('Estadística mensual', su nombre empieza con mes)? Los archivos de "
        "'Normales climatológicas' o 'Diario' tienen otro formato y este programa no los lee."
    )

print("Tablas encontradas en el archivo:", len(tablas_encontradas), "de", len(TABLAS))
faltan = []
for nombre in nombres_columnas:
    if nombre not in tablas_encontradas:
        faltan.append(nombre)
if len(faltan) > 0:
    print("Tablas que NO aparecen en el archivo (esas columnas quedarán vacías):", faltan)
if celdas_raras > 0:
    print("Celdas con texto en lugar de número (se tomaron como vacías):", celdas_raras)

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
if len(tabla) == 0:
    raise SystemExit("El archivo no trae ninguna medición. Prueba con otra estación.")

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

# Avisos para saber si esta estación sirve para los pasos siguientes
if "lluvia_total" in columnas_quitar:
    print("AVISO: a esta estación le faltan demasiados datos de lluvia_total. Los pasos 02 a 05 "
          "usan lluvia_total como objetivo, así que no podrán correr con ella.")
if len(tabla) < 120:
    print("AVISO: solo hay", len(tabla), "meses con datos (10 años = 120). Con poco historial "
          "el modelo aprende poco; los pasos siguientes necesitan al menos 75 meses.")

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
RUTA_LIMPIOS = CARPETA_PROYECTO / "datos_limpios.csv"
RUTA_PREPARADOS = CARPETA_PROYECTO / "datos_preparados.csv"
tabla.to_csv(RUTA_LIMPIOS, index=False)
preparado.to_csv(RUTA_PREPARADOS, index=False)

print("Datos limpios:    ", tabla.shape, "->", RUTA_LIMPIOS)
print("Datos preparados: ", preparado.shape, "->", RUTA_PREPARADOS)
print("Vacíos que quedan tras el pipeline:", int(preparado.isna().sum().sum()))