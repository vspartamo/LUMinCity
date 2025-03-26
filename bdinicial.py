import numpy as np
import pandas as pd
import os

np.random.seed(42)
num_agentes = 100
num_inmuebles = 100
num_zonas = 10

# 1. Atributos AGENTE
df_agentes = pd.DataFrame({
    'AGENTE_ID': range(num_agentes),
    'ingreso_hogar': np.random.lognormal(mean=14, sigma=0.4, size=num_agentes).round(2),
    'miembros_hogar': np.random.poisson(lam=2.5, size=num_agentes) + 1,
})

percentiles = np.percentile(df_agentes['ingreso_hogar'], [33, 66])
df_agentes['grupo_agente'] = np.where(
    df_agentes['ingreso_hogar'] < percentiles[0], "bajo",
    np.where(df_agentes['ingreso_hogar'] < percentiles[1], "mediano", "alto"))

# 2. Atributos INMUEBLE
df_inmuebles = pd.DataFrame({
    'INMUEBLE_ID': range(num_inmuebles),
    'ZONA_ID': np.random.randint(0, num_zonas, size=num_inmuebles),
    'tipo_inmueble': np.random.choice(["casa", "dpto"], num_inmuebles, p=[0.4, 0.6]),
})
df_inmuebles['formato_inmueble'] = (df_inmuebles['tipo_inmueble'] == "dpto").astype(int)

es_casa = df_inmuebles['tipo_inmueble'] == "casa"
df_inmuebles['superficie_construida'] = np.where(
    es_casa,
    np.random.normal(150, 40, num_inmuebles).clip(80, 300).round(1),
    np.random.normal(90, 20, num_inmuebles).clip(50, 150).round(1)
)
df_inmuebles['superficie_terreno'] = np.where(
    es_casa,
    df_inmuebles['superficie_construida'] * np.random.uniform(1.5, 3, num_inmuebles),
    df_inmuebles['superficie_construida'] * np.random.uniform(0.8, 1.2, num_inmuebles)
).round(1)

# 3. Atributos ZONA
df_zonas = pd.DataFrame({
    'ZONA_ID': range(num_zonas),
    'macro_zona': np.random.choice([0, 1, 2, 3, 4], num_zonas),
    'num_estaciones_metro': np.random.poisson(lam=1.5, size=num_zonas),
    'sup_comercio': np.random.lognormal(mean=10, sigma=0.3, size=num_zonas).round(1),
    'sup_industria': np.random.lognormal(mean=9.5, sigma=0.4, size=num_zonas).round(1),
})

# 4. Precio (versión corregida)
precio_base = np.random.lognormal(mean=12.5, sigma=0.3, size=num_inmuebles)
df_inmuebles['precio'] = (precio_base * 
                         (1 + df_inmuebles['ZONA_ID'] * 0.05) * 
                         np.where(es_casa, 1.2, 1.0)).round(2)

# 5. Tiempo de viaje
tiempo_viaje = np.random.gamma(shape=2, scale=10, size=num_agentes).round(1)

# Unificación - MODIFICADO para asegurar ubicación correcta
df = df_agentes.copy()

# Asignación de inmuebles a agentes (permite que varios agentes compartan inmueble si se desea)
# En este caso mantenemos la relación 1:1 pero de forma más explícita
df['INMUEBLE_ID'] = np.random.choice(df_inmuebles['INMUEBLE_ID'], size=num_agentes, replace=False)

# Fusionar con información de inmuebles y zonas
df = df.merge(df_inmuebles, on='INMUEBLE_ID')
df = df.merge(df_zonas, on='ZONA_ID')

# Añadir tiempo de viaje
df['tiempo_viaje'] = tiempo_viaje

# Guardar
ruta = r"C:\\Users\\nicol\\OneDrive\\Escritorio\\LUMinCity\\LUMinCity"
df.to_csv(os.path.join(ruta, "base_agentes_localizados.csv"), index=False, sep=";")

print("¡Base generada con éxito!")
print(f"Total agentes: {len(df)}")
print("Estructura final:\n", df.dtypes)