"""
AGRUPAR AÑOS PARECIDOS (KMeans, capítulo 9 de handson-ml2)

Lee datos_limpios.csv (lo crea limpieza_de_datos.py) y junta los AÑOS que se
parecen entre sí, por ejemplo: "años secos y calurosos", "años lluviosos y frescos".
Sirve para ver qué tipos de año ha tenido la región y cuáles son los mejores
o peores casos.

Qué hace:
  1. Resume cada año en pocos números (lluvia del año, temperaturas promedio...)
  2. Los pone en la misma escala (Pipeline: SimpleImputer + StandardScaler, capítulo 2)
  3. Prueba KMeans con 2, 3, 4... grupos y mide cuál separa mejor (inertia y silhouette)
  4. Entrena el KMeans final y describe cada grupo con palabras
  5. Guarda cada año con su grupo en anios_agrupados.csv

Funciona con cualquier estación: usa las columnas que tenga el archivo.
NO necesita los otros pasos (dividir, modelo, ajustar): solo limpieza_de_datos.py.
"""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Trabajar siempre en la carpeta donde está ESTE archivo: así lee y guarda ahí
# aunque lo ejecutes desde otra carpeta (por ejemplo desde la raíz del proyecto).
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------
ARCHIVO = "datos_limpios.csv"
GRUPOS = "auto"          # "auto" = el programa elige cuántos grupos; o escribe un número (por ejemplo 3)
MAXIMO_GRUPOS = 6        # hasta cuántos grupos prueba cuando GRUPOS = "auto"
MESES_MINIMOS = 10       # un año necesita al menos tantos meses con datos para usarse
MINIMO_ANIOS = 20        # con menos años no tiene sentido agrupar

# Cómo se resume cada columna mensual en un número por año:
#   "lluvia_total"   -> lluvia de un año completo (promedio mensual x 12)
#   "lluvia_max_24h" -> la lluvia más fuerte de ese año en un día
#   temperaturas     -> promedio del año
NOMBRES = {
    "lluvia_total": "lluvia_anual",
    "lluvia_max_24h": "lluvia_max_dia",
    "temp_media": "temp_media",
    "temp_max_prom": "temp_max_prom",
    "temp_min_prom": "temp_min_prom",
}

# ----------------------------------------------------------------------
# 1. LEER Y RESUMIR CADA AÑO
# ----------------------------------------------------------------------
if not os.path.exists(ARCHIVO):
    raise SystemExit("No encuentro " + ARCHIVO + " en la carpeta " + os.getcwd() +
                     ". Primero corre limpieza_de_datos.py.")

tabla = pd.read_csv(ARCHIVO)
print("Meses en el archivo:", len(tabla))

# Solo usamos las columnas de NOMBRES que sí existan en esta estación
columnas = []
for columna in NOMBRES:
    if columna in tabla.columns:
        columnas.append(columna)

if len(columnas) < 2:
    raise SystemExit("Esta estación tiene muy pocas variables (" + ", ".join(columnas) +
                     "). Para agrupar se necesitan al menos 2. Prueba con otra estación.")

filas = []
for anio in sorted(set(tabla["anio"])):
    del_anio = tabla[tabla["anio"] == anio]       # los meses de ese año
    fila = {"anio": int(anio)}
    meses_con_datos = 0
    for columna in columnas:
        valores = del_anio[columna].dropna()      # los meses que sí tienen dato
        if len(valores) < MESES_MINIMOS:
            fila[NOMBRES[columna]] = np.nan       # muy pocos meses: no confiamos en el resumen
        elif columna == "lluvia_total":
            fila[NOMBRES[columna]] = valores.mean() * 12
            meses_con_datos = meses_con_datos + 1
        elif columna == "lluvia_max_24h":
            fila[NOMBRES[columna]] = valores.max()
            meses_con_datos = meses_con_datos + 1
        else:
            fila[NOMBRES[columna]] = valores.mean()
            meses_con_datos = meses_con_datos + 1
    if meses_con_datos > 0:                        # el año aporta al menos una variable
        filas.append(fila)

anual = pd.DataFrame(filas)
print("Años con datos suficientes (al menos", MESES_MINIMOS, "meses):", len(anual))

# Quitar las columnas a las que les falta demasiado (más de 30 % de los años)
variables = []
for columna in columnas:
    nombre = NOMBRES[columna]
    if anual[nombre].isna().mean() <= 0.30:
        variables.append(nombre)
    else:
        print("Se descarta", nombre, "porque le faltan datos en muchos años.")

if len(variables) < 2:
    raise SystemExit("Quedaron menos de 2 variables con datos suficientes. Prueba con otra estación.")

# Quitar los años a los que les falta TODO (no se pueden agrupar)
anual = anual.dropna(subset=variables, how="all").reset_index(drop=True)
if len(anual) < MINIMO_ANIOS:
    raise SystemExit("Solo hay " + str(len(anual)) + " años utilizables (se necesitan al menos " +
                     str(MINIMO_ANIOS) + "). Prueba con una estación con más historial.")

print("Variables para agrupar:", variables)

# ----------------------------------------------------------------------
# 2. PREPARAR: mismo Pipeline del capítulo 2
# ----------------------------------------------------------------------
# KMeans mide DISTANCIAS entre años. Si la lluvia está en cientos (mm) y la
# temperatura en decenas (°C), la lluvia "pesaría" mucho más. StandardScaler
# pone todas las variables en la misma escala para que cuenten por igual.
# (En el Random Forest esto no importaba; en KMeans sí.)
preparacion = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # rellena huecos con la mediana
    ("std_scaler", StandardScaler()),
])
X = preparacion.fit_transform(anual[variables])

# ----------------------------------------------------------------------
# 3. ¿CUÁNTOS GRUPOS? (capítulo 9: inertia y silhouette)
# ----------------------------------------------------------------------
# inertia   = qué tan lejos quedan los años del centro de su grupo (más baja = grupos más compactos).
#             Siempre baja al agregar grupos, así que sola no basta.
# silhouette = de -1 a 1: qué tan bien separados están los grupos (más alta = mejor).
tope = min(MAXIMO_GRUPOS, len(anual) - 1)
resultados = {}
print("\nProbando cuántos grupos conviene:")
for k in range(2, tope + 1):
    prueba = KMeans(n_clusters=k, n_init=10, random_state=42)
    etiquetas = prueba.fit_predict(X)
    resultados[k] = silhouette_score(X, etiquetas)
    print("  k =", k, "| inertia =", round(prueba.inertia_, 1), "| silhouette =", round(resultados[k], 3))

if GRUPOS == "auto":
    k = max(resultados, key=resultados.get)          # el k con mejor silhouette
    print("\nSe eligió k =", k, "(el de mejor silhouette)")
else:
    k = int(GRUPOS)
    if k < 2 or k > len(anual) - 1:
        raise SystemExit("GRUPOS debe ser entre 2 y " + str(len(anual) - 1) + ".")
    print("\nUsando k =", k, "(el que pusiste en GRUPOS)")

# ----------------------------------------------------------------------
# 4. KMEANS FINAL
# ----------------------------------------------------------------------
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
anual["grupo"] = kmeans.fit_predict(X) + 1           # +1 para llamarlos grupo 1, 2, 3...
calidad = silhouette_score(X, anual["grupo"])

print("Silhouette final:", round(calidad, 3))
if calidad >= 0.5:
    print("  -> grupos bien separados.")
elif calidad >= 0.25:
    print("  -> grupos razonables, pero con años que quedan en la frontera.")
else:
    print("  -> grupos poco separados: el clima cambia de forma gradual, no en tipos claros.")
    print("     Tómalos como una tendencia, no como categorías exactas.")

# ----------------------------------------------------------------------
# 5. DESCRIBIR CADA GRUPO
# ----------------------------------------------------------------------
# kmeans.cluster_centers_ son los centros en la escala estandarizada:
# 0 = igual al promedio de todos los años, +1 = una desviación por encima, etc.
# Con eso escribimos "alta", "normal" o "baja" para cada variable.
promedios = anual[variables].mean()
print("\n==== GRUPOS DE AÑOS ====")
for numero in range(1, k + 1):
    del_grupo = anual[anual["grupo"] == numero]
    centro = kmeans.cluster_centers_[numero - 1]

    descripcion = []
    for j in range(len(variables)):
        if centro[j] > 0.5:
            nivel = "alta"
        elif centro[j] < -0.5:
            nivel = "baja"
        else:
            nivel = "normal"
        descripcion.append(variables[j] + " " + nivel)

    print("\nGrupo", numero, "-", len(del_grupo), "años")
    print("  Cómo es:", " | ".join(descripcion))
    print("  Promedios del grupo (entre paréntesis, el promedio de todos los años):")
    for nombre in variables:
        print("    ", nombre.ljust(16), round(del_grupo[nombre].mean(), 1), "(" + str(round(promedios[nombre], 1)) + ")")

    anios = list(del_grupo["anio"])
    if len(anios) > 15:
        texto = ", ".join(str(a) for a in anios[:15]) + ", ..."
    else:
        texto = ", ".join(str(a) for a in anios)
    print("  Años:", texto)

# ----------------------------------------------------------------------
# 6. GUARDAR
# ----------------------------------------------------------------------
anual.to_csv("anios_agrupados.csv", index=False)
print("\nGuardado: anios_agrupados.csv (cada año con su grupo)")
print("Nota: con pocos años, otro random_state o más datos pueden cambiar los grupos.")
