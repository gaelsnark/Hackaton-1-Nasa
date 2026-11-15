"""
IMPORTANCIA DE LAS VARIABLES (feature_importances_, capítulo 2 de handson-ml2)

Abre el modelo que guardó ajustar_model.py (modelo_ajustado_regresion.pkl o
modelo_ajustado_clasificacion.pkl) y le pregunta al Random Forest:
"¿qué variables usaste más para predecir?"

Qué hace:
  1. Carga el modelo ya entrenado (no vuelve a entrenar nada)
  2. Saca la importancia de cada columna con feature_importances_
  3. Junta los 12 meses (ENE, FEB...) en una sola variable llamada "mes"
  4. Muestra la lista ordenada y la guarda en importancia_variables.csv

Antes hay que correr: limpieza_de_datos.py, dividir_datos.py, modelo_forest.py
(o clasificador_forest.py) y ajustar_model.py con el mismo TIPO que pongas aquí.

Funciona con cualquier estación: no tiene ningún dato de Amealco escrito.
"""
import os

import joblib
import pandas as pd

# Trabajar siempre en la carpeta donde está ESTE archivo: así lee y guarda ahí
# aunque lo ejecutes desde otra carpeta (por ejemplo desde la raíz del proyecto).
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------
TIPO = "regresion"   # "regresion" o "clasificacion": el mismo que usaste en ajustar_model.py

if TIPO != "regresion" and TIPO != "clasificacion":
    raise SystemExit('TIPO debe ser "regresion" o "clasificacion".')

ARCHIVO_MODELO = "modelo_ajustado_" + TIPO + ".pkl"

# ----------------------------------------------------------------------
# 1. CARGAR EL MODELO GUARDADO
# ----------------------------------------------------------------------
if not os.path.exists(ARCHIVO_MODELO):
    raise SystemExit("No encuentro " + ARCHIVO_MODELO + " en la carpeta " + os.getcwd() +
                     ". Primero corre ajustar_model.py con TIPO = \"" + TIPO + "\".")

modelo = joblib.load(ARCHIVO_MODELO)   # joblib.load = lo contrario de joblib.dump
print("Modelo cargado:", ARCHIVO_MODELO)

# El modelo es un Pipeline con dos pasos: "preparacion" (ColumnTransformer) y "bosque"
preparacion = modelo.named_steps["preparacion"]
bosque = modelo.named_steps["bosque"]

# ----------------------------------------------------------------------
# 2. IMPORTANCIA DE CADA COLUMNA (igual que en el capítulo 2 del repo)
# ----------------------------------------------------------------------
# feature_importances_ = una lista de números que suman 1 (100 %): cuánto
# ayudó cada columna a que el bosque tomara buenas decisiones.
importancias = bosque.feature_importances_

# El ColumnTransformer sabe cómo se llama cada columna que salió de la preparación,
# por ejemplo "num__temp_media" o "cat__mes_JUL". Nos quedamos con lo que va después de "__".
nombres = []
for nombre in preparacion.get_feature_names_out():
    nombres.append(nombre.split("__")[1])

# sorted(zip(...)) une cada importancia con su nombre y las ordena de mayor a menor
# (así lo hace el libro: sorted(zip(feature_importances, attributes), reverse=True))
detalle = sorted(zip(importancias, nombres), reverse=True)

print("\nImportancia de cada columna (los meses vienen separados):")
for importancia, nombre in detalle:
    print("  ", nombre.ljust(16), round(importancia * 100, 1), "%")

# ----------------------------------------------------------------------
# 3. JUNTAR LOS MESES EN UNA SOLA VARIABLE
# ----------------------------------------------------------------------
# El OneHotEncoder convirtió "mes" en 12 columnas (mes_ENE, mes_FEB...).
# Para leer mejor el resultado, sumamos esas 12 y las llamamos "mes".
por_variable = {}
for importancia, nombre in detalle:
    if nombre.startswith("mes_"):
        nombre = "mes"
    if nombre not in por_variable:
        por_variable[nombre] = 0
    por_variable[nombre] = por_variable[nombre] + importancia

resultado = pd.DataFrame({
    "variable": list(por_variable.keys()),
    "importancia_%": [round(v * 100, 1) for v in por_variable.values()],
})
resultado = resultado.sort_values("importancia_%", ascending=False).reset_index(drop=True)

print("\nImportancia por variable (con el mes junto):")
for i in range(len(resultado)):
    porcentaje = resultado["importancia_%"][i]
    barra = "#" * int(round(porcentaje / 2))     # una # por cada 2 %
    print("  ", resultado["variable"][i].ljust(16), str(porcentaje).rjust(5), "%", barra)

# ----------------------------------------------------------------------
# 4. CÓMO LEER EL RESULTADO
# ----------------------------------------------------------------------
primera = resultado["variable"][0]
print("\nLa variable que más usa el modelo es:", primera)
if "anio" in list(resultado["variable"][:3]):
    print("AVISO: 'anio' está entre las 3 primeras. Puede ser que el modelo use el año para")
    print("       aprender la época (por ejemplo cambios de clima o de estación de medición)")
    print("       y no una causa real. Tómalo con cuidado.")
print("Recuerda: importancia alta = el modelo se apoya mucho en esa variable para predecir.")
print("No significa que esa variable CAUSE el resultado, solo que está muy relacionada con él.")
print("Si dos variables se parecen mucho (por ejemplo temp_media y temp_max_prom),")
print("su importancia se reparte entre ellas y cada una parece menos importante de lo que es.")

# ----------------------------------------------------------------------
# 5. GUARDAR
# ----------------------------------------------------------------------
resultado.to_csv("importancia_variables.csv", index=False)
print("\nGuardado: importancia_variables.csv")
