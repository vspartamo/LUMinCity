import numpy as np
import pandas as pd
import os

# Cargar la base de datos inicial
ruta = r"C:\\Users\\nicol\\OneDrive\\Escritorio\\LUMinCity\\LUMinCity"
df_inicial = pd.read_csv(os.path.join(ruta, "base_inicial.csv"), sep=";")

# ======================== #
# Parámetros y errores     #
# ======================== #
num_agentes = df_inicial['AGENTE_ID'].nunique()
num_inmuebles = df_inicial['INMUEBLE'].nunique()
num_zonas = df_inicial['ZONA'].nunique()

# Parámetros heterogéneos por agente para Choice
beta_h_r = np.random.normal(-0.8, 0.2, num_agentes)   # Sensibilidad al precio
beta_h_d = np.random.normal(0.5, 0.1, num_agentes)    # Sensibilidad a superficie
beta_h_z = np.random.normal(1.2, 0.3, num_agentes)    # Sensibilidad a ingreso zona
beta_h_a = np.random.normal(-0.7, 0.2, num_agentes)   # Sensibilidad a tiempo de viaje
cvi = np.random.normal(2.5, 0.5, (num_inmuebles, num_zonas))  # Utilidad base

# Parámetros heterogéneos para Bid
alpha_h = np.random.normal(0.6, 0.1, num_agentes)     # Sensibilidad a superficie
gamma_h_z = np.random.normal(0.8, 0.2, num_agentes)   # Sensibilidad a ingreso zona
gamma_h_a = np.random.normal(-0.5, 0.1, num_agentes)  # Sensibilidad a tiempo de viaje
bh = np.random.normal(5.0, 1.0, num_agentes)          # Disposición base a pagar

# Función para errores Gumbel
def generar_gumbel(size, mu=0, beta=1):
    U = np.random.uniform(0, 1, size=size)
    return mu - beta * np.log(-np.log(U))

# Generar errores (escala 1 para logit)
epsilon_choice = generar_gumbel((num_agentes, num_inmuebles, num_zonas), beta=1)
epsilon_bid = generar_gumbel((num_agentes, num_inmuebles, num_zonas), beta=1)

# ======================== #
# Cálculo de utilidades y posturas
# ======================== #
def calcular_utilidades_posturas(df):
    # Mapear índices únicos
    agentes = df['AGENTE_ID'].unique()
    inmuebles = df['INMUEBLE'].unique()
    zonas = df['ZONA'].unique()
    
    # Crear arrays 3D
    utilidades = np.zeros((num_agentes, num_inmuebles, num_zonas))
    posturas = np.zeros((num_agentes, num_inmuebles, num_zonas))
    
    for idx, row in df.iterrows():
        a = np.where(agentes == row['AGENTE_ID'])[0][0]
        v = np.where(inmuebles == row['INMUEBLE'])[0][0]
        i = np.where(zonas == row['ZONA'])[0][0]
        
        # Calcular utilidad
        utilidades[a,v,i] = (
            cvi[v,i] +
            beta_h_r[a] * row['PRECIO'] +
            beta_h_d[a] * row['SUPERFICIE'] +
            beta_h_z[a] * row['INGRESO_ZONA'] +
            beta_h_a[a] * row['TIEMPO_VIAJE'] +
            epsilon_choice[a,v,i]
        )
        
        # Calcular postura
        posturas[a,v,i] = (
            bh[a] +
            alpha_h[a] * row['SUPERFICIE'] +
            gamma_h_z[a] * row['INGRESO_ZONA'] +
            gamma_h_a[a] * row['TIEMPO_VIAJE'] +
            epsilon_bid[a,v,i]
        )
    
    df['UTILIDAD'] = utilidades.reshape(-1)
    df['POSTURA'] = posturas.reshape(-1)
    return df

df_con_calculos = calcular_utilidades_posturas(df_inicial)

# ======================== #
# Funciones de agrupación
# ======================== #
def categorizar(row, col_agrupar, rangos):
    """Asigna un grupo según rangos personalizados."""
    for grupo, (min_val, max_val) in rangos.items():
        if min_val <= row[col_agrupar] < max_val:
            return grupo
    return 'fuera_de_rango'

def agrupar_agentes_bid(df, col_agrupar="TIEMPO_VIAJE", rango_grupos={"bajo": (0,10), "medio": (10,20), "alto": (20,100)}):
    """Agrupa agentes y calcula posturas por inmueble."""
    df_g = df.copy()
    df_g['GRUPO_AGENTE'] = df_g.apply(lambda x: categorizar(x, col_agrupar, rango_grupos), axis=1)
    
    # Calcular posturas por grupo
    posturas = df_g.groupby(['INMUEBLE', 'ZONA', 'GRUPO_AGENTE']).agg({
        'PRECIO': 'mean',
        'SUPERFICIE': 'mean',
        'INGRESO_ZONA': 'mean',
        'TIEMPO_VIAJE': 'mean',
        'POSTURA': 'mean'
    }).reset_index()
    
    # Asignar inmueble al grupo con mayor postura
    posturas['CHOICE'] = posturas.groupby(['INMUEBLE', 'ZONA'])['POSTURA'].transform('idxmax')
    posturas['CHOICE'] = posturas.loc[posturas['CHOICE'], 'GRUPO_AGENTE'].values
    
    # Agregar TIEMPO_VIAJE desde el dataframe inicial
    #posturas = posturas.merge(
    #    df_inicial[['INMUEBLE', 'ZONA', 'TIEMPO_VIAJE']].drop_duplicates(),
    #    on=['INMUEBLE', 'ZONA'],
    #    how='left'
    #)
    
    return posturas[['INMUEBLE', 'ZONA', 'PRECIO', 'SUPERFICIE', 'INGRESO_ZONA', 'TIEMPO_VIAJE', 'GRUPO_AGENTE', 'POSTURA', 'CHOICE']]

def agrupar_inmuebles_choice(df, col_agrupar="PRECIO", rango_grupos={"bajo": (0,200000), "medio": (200000,400000), "alto": (400000,600000)}):
    """Agrupa inmuebles y calcula elecciones por agente."""
    df_g = df.copy()
    df_g['GRUPO_INMUEBLE'] = df_g.apply(lambda x: categorizar(x, col_agrupar, rango_grupos), axis=1)
    
    # Calcular utilidades por grupo
    utilidades = df_g.groupby(['AGENTE_ID', 'GRUPO_INMUEBLE']).agg({
        'PRECIO': 'mean',
        'SUPERFICIE': 'mean',
        'INGRESO_ZONA': 'mean',
        'TIEMPO_VIAJE': 'mean',
        'UTILIDAD': 'mean'
    }).reset_index()

    # Asignar agente al grupo con mayor utilidad
    utilidades['CHOICE'] = utilidades.groupby('AGENTE_ID')['UTILIDAD'].transform('idxmax')
    utilidades['CHOICE'] = utilidades.loc[utilidades['CHOICE'], 'GRUPO_INMUEBLE'].values
    
    ## Agregar PRECIO, SUPERFICIE e INGRESO_ZONA desde el dataframe inicial
    #utilidades = utilidades.merge(
    #    df_inicial[['AGENTE_ID', 'PRECIO', 'SUPERFICIE', 'INGRESO_ZONA']].drop_duplicates(),
    #    on='AGENTE_ID',
    #    how='left'
    #)
    
    return utilidades[['AGENTE_ID', 'PRECIO', 'SUPERFICIE', 'INGRESO_ZONA', 'TIEMPO_VIAJE', 'GRUPO_INMUEBLE', 'UTILIDAD', 'CHOICE']]

# Función para transformar el DataFrame
def transformar_df(df):
    # Crear un DataFrame vacío para almacenar los resultados
    resultado = pd.DataFrame(columns=[
        "INMUEBLE", "ZONA", "PRECIO", "SUPERFICIE", "INGRESO_ZONA",
        "TIEMPO_VIAJE_ALTO", "TIEMPO_VIAJE_MEDIO", "TIEMPO_VIAJE_BAJO",
        "POSTURA_ALTO", "POSTURA_MEDIO", "POSTURA_BAJO", "CHOICE"
    ])
    
    # Agrupar por INMUEBLE y ZONA
    grupos = df.groupby(["INMUEBLE", "ZONA"])
    
    for (inmueble, zona), grupo in grupos:
        # Extraer valores únicos
        precio = grupo["PRECIO"].iloc[0]
        superficie = grupo["SUPERFICIE"].iloc[0]
        ingreso_zona = grupo["INGRESO_ZONA"].iloc[0]
        choice = grupo["CHOICE"].iloc[0]
        
        # Inicializar valores para cada grupo de agentes
        tiempo_viaje_alto = None
        tiempo_viaje_medio = None
        tiempo_viaje_bajo = None
        postura_alto = None
        postura_medio = None
        postura_bajo = None
        
        # Llenar los valores según el grupo de agentes
        for _, fila in grupo.iterrows():
            if fila["GRUPO_AGENTE"] == "alto":
                tiempo_viaje_alto = fila["TIEMPO_VIAJE"]
                postura_alto = fila["POSTURA"]
            elif fila["GRUPO_AGENTE"] == "medio":
                tiempo_viaje_medio = fila["TIEMPO_VIAJE"]
                postura_medio = fila["POSTURA"]
            elif fila["GRUPO_AGENTE"] == "bajo":
                tiempo_viaje_bajo = fila["TIEMPO_VIAJE"]
                postura_bajo = fila["POSTURA"]
        
        # Crear una nueva fila como DataFrame
        nueva_fila = pd.DataFrame({
            "INMUEBLE": [inmueble],
            "ZONA": [zona],
            "PRECIO": [precio],
            "SUPERFICIE": [superficie],
            "INGRESO_ZONA": [ingreso_zona],
            "TIEMPO_VIAJE_ALTO": [tiempo_viaje_alto],
            "TIEMPO_VIAJE_MEDIO": [tiempo_viaje_medio],
            "TIEMPO_VIAJE_BAJO": [tiempo_viaje_bajo],
            "POSTURA_ALTO": [postura_alto],
            "POSTURA_MEDIO": [postura_medio],
            "POSTURA_BAJO": [postura_bajo],
            "CHOICE": [choice]
        })
        
        # Concatenar la nueva fila al DataFrame resultado
        resultado = pd.concat([resultado, nueva_fila], ignore_index=True)
    
    return resultado
# ======================== #
# Ejecución y guardado
# ======================== #
# Agrupar agentes (Bid)
grupos_bid = transformar_df(agrupar_agentes_bid(
    df_con_calculos,
    col_agrupar="TIEMPO_VIAJE",
    rango_grupos={"bajo": (0,10), "medio": (10,20), "alto": (20,100)}
))

# Agrupar inmuebles (Choice)
grupos_choice = agrupar_inmuebles_choice(
    df_con_calculos,
    col_agrupar="PRECIO",
    rango_grupos={"bajo": (0,200000), "medio": (200000,400000), "alto": (400000,600000)}
)

# Guardar resultados
grupos_bid.to_csv(os.path.join(ruta, "grupos_bid.csv"), index=False, sep=";")
grupos_choice.to_csv(os.path.join(ruta, "grupos_choice.csv"), index=False, sep=";")

print("¡Agrupaciones generadas exitosamente!")