import numpy as np
import pandas as pd
import os

# Configuración inicial
np.random.seed(42)
num_agentes = 100
num_inmuebles = 10
num_zonas = 10

# ======================== #
# Generación de atributos  #
# ======================== #
# Atributos con distribuciones más realistas
rvi = np.random.lognormal(mean=12.5, sigma=0.3, size=(num_inmuebles, num_zonas)).round(2)  # Precios (log-normal)
dvi = np.random.normal(100, 20, (num_inmuebles, num_zonas)).clip(50, 200)                # Superficie (m²)
zi = np.random.lognormal(10.5, 0.2, num_zonas).round(2)                                  # Ingreso zona (log-normal $)
ahi = np.random.gamma(shape=2, scale=10, size=(num_agentes, num_zonas)).round(1)         # Tiempo viaje (min)

# ======================== #
# Creación de DataFrames   #
# ======================== #
def crear_dataframe_inicial():
    """Crea un DataFrame con todas las combinaciones de agentes, inmuebles y zonas."""
    agentes, vs, is_ = np.indices((num_agentes, num_inmuebles, num_zonas))
    agentes = agentes.flatten()
    vs = vs.flatten()
    is_ = is_.flatten()
    
    # Crear DataFrame base
    df = pd.DataFrame({
        'AGENTE_ID': agentes,
        'INMUEBLE': vs,
        'ZONA': is_
    })
    
    # Agregar atributos
    df['PRECIO'] = rvi[df['INMUEBLE'], df['ZONA']]
    df['SUPERFICIE'] = dvi[df['INMUEBLE'], df['ZONA']]
    df['INGRESO_ZONA'] = zi[df['ZONA']]
    df['TIEMPO_VIAJE'] = ahi[df['AGENTE_ID'], df['ZONA']]
    
    return df

# Generar DataFrame inicial
df_inicial = crear_dataframe_inicial()

# ======================== #
# Guardar resultados       #
# ======================== #
ruta = r"C:\\Users\\nicol\\OneDrive\\Escritorio\\LUMinCity\\LUMinCity"
df_inicial.to_csv(os.path.join(ruta, "base_inicial.csv"), index=False, sep=";")

print("¡Base inicial generada exitosamente!")