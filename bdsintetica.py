import numpy as np
import pandas as pd
import os

def configurar_parametros_modelo(num_agentes):
    """Configura los parámetros del modelo para utilidades y posturas"""
    # Parámetros para Choice Model (utilidad)
    params = {
        'beta_h_precio': np.random.normal(-0.5, 0.5, num_agentes),
        'beta_h_sup': np.random.gamma(shape=2, scale=0.3, size=num_agentes),
        'beta_h_sup_construida': np.random.gamma(shape=2, scale=0.3, size=num_agentes),
        'beta_h_sup_construida_comercio': np.random.gamma(shape=2, scale=0.3, size=num_agentes),
        'beta_h_sup_construida_industria': np.random.gamma(shape=2, scale=0.3, size=num_agentes),
        'beta_h_tiempo': np.random.normal(-0.4, 0.1, num_agentes),
        'beta_h_tipo': np.random.normal(0.3, 0.1, num_agentes),
        'beta_h_sup_terreno': np.random.gamma(shape=2, scale=0.3, size=num_agentes),
        'beta_h_nro_estaciones': np.random.normal(0.1, 0.05, num_agentes),

        
        # Parámetros para Bid Model (postura)
        'alpha_h_sup': np.random.gamma(shape=2, scale=0.2, size=num_agentes),
        'alpha_h_sup_construida': np.random.gamma(shape=2, scale=0.2, size=num_agentes),
        'gamma_h_tiempo': np.random.normal(-0.3, 0.1, num_agentes),
        'gamma_h_ingreso': np.random.lognormal(mean=-1.2, sigma=0.3, size=num_agentes),
        'base_bid': np.random.lognormal(mean=2.5, sigma=0.5, size=num_agentes),
        'gamma_h_comercio': np.random.normal(0.1, 0.05, num_agentes),
        'gamma_h_estaciones': np.random.normal(0.2, 0.1, num_agentes),
        'alpha_h_sup_terreno': np.random.gamma(shape=2, scale=0.2, size=num_agentes)
    }
    return params

def generar_gumbel(size, mu=0, beta=1):
    """Genera términos de error Gumbel para modelos logit"""
    U = np.random.uniform(0, 1, size=size)
    return mu - beta * np.log(-np.log(U))

def calcular_utilidad(row, a_idx, params):
    """Calcula la utilidad de un inmueble para un agente"""
    error_choice = generar_gumbel(1)
    return (
        params['beta_h_precio'][a_idx] * row['precio'] +
        params['beta_h_sup_terreno'][a_idx] * row['superficie_terreno'] +
        params['beta_h_sup_construida'][a_idx] * row['superficie_construida'] +
        params['beta_h_tiempo'][a_idx] * row['tiempo_viaje'] +
        params['beta_h_tipo'][a_idx] * row['formato_inmueble'] +
        params['beta_h_nro_estaciones'][a_idx] * row['nro_estaciones'] +
        params['beta_h_sup_construida_comercio'][a_idx] * row['sup_comercio'] +
        params['beta_h_sup_construida_industria'][a_idx] * row['sup_industria']+
        error_choice        
    )
    
def crear_dataframe_utilidades(df, params):
    """Crea un dataframe mostrando siempre las 3 opciones por agente, con asignación realista"""
    # 1. Clasificar inmuebles y contar disponibilidad
    df['rango_precio'] = pd.qcut(df['precio'], q=3, labels=['bajo', 'medio', 'alto'])
    disponibilidad = df['rango_precio'].value_counts().to_dict()
    
    # 2. Preparar datos
    inmuebles_df = df.copy()
    inmuebles_df['ocupado'] = False
    asignaciones = {rango: 0 for rango in ['bajo', 'medio', 'alto']}
    agentes = df['AGENTE_ID'].unique()
    np.random.shuffle(agentes)
    utilidades = []
    
    # 3. Función para calcular utilidad hipotética (promedio de los disponibles)
    def calcular_utilidad_hipotetica(agente_idx, rango):
        muestra = inmuebles_df[inmuebles_df['rango_precio'] == rango].sample(min(5, disponibilidad[rango]))
        if muestra.empty:
            return np.nan
        return np.mean([calcular_utilidad(row, agente_idx, params) for _, row in muestra.iterrows()])
    
    for agente_id in agentes:
        agente_data = df[df['AGENTE_ID'] == agente_id].iloc[0]
        a_idx = np.where(df['AGENTE_ID'].unique() == agente_id)[0][0]
        
        opciones = {}
        rango_elegido = None
        
        # Evaluar TODAS las opciones (aunque no estén disponibles)
        for rango in ['bajo', 'medio', 'alto']:
            # Opción disponible
            if asignaciones[rango] < disponibilidad[rango]:
                inmuebles_rango = inmuebles_df[
                    (inmuebles_df['rango_precio'] == rango) & 
                    (~inmuebles_df['ocupado'])]
                
                if not inmuebles_rango.empty:
                    mejor_utilidad = -np.inf
                    mejor_inmueble = None
                    
                    for _, inmueble in inmuebles_rango.iterrows():
                        utilidad = calcular_utilidad(inmueble, a_idx, params)
                        if utilidad > mejor_utilidad:
                            mejor_utilidad = utilidad
                            mejor_inmueble = inmueble
                    
                    opciones[rango] = {
                        'inmueble': mejor_inmueble,
                        'utilidad': mejor_utilidad,
                        'disponible': True
                    }
            # Opción no disponible (cálculo hipotético)
            else:
                opciones[rango] = {
                    'inmueble': None,
                    'utilidad': calcular_utilidad_hipotetica(a_idx, rango),
                    'disponible': False
                }
        
        # Elegir solo entre opciones disponibles
        opciones_disponibles = {k:v for k,v in opciones.items() if v['disponible']}
        if opciones_disponibles:
            rango_elegido = max(opciones_disponibles.keys(), 
                              key=lambda x: opciones_disponibles[x]['utilidad'])
            
            # Marcar inmueble como ocupado si se eligió
            if rango_elegido:
                inmueble_elegido = opciones[rango_elegido]['inmueble']
                inmuebles_df.loc[inmuebles_df['INMUEBLE_ID'] == inmueble_elegido['INMUEBLE_ID'], 'ocupado'] = True
                asignaciones[rango_elegido] += 1
        
        # Registrar las 3 opciones
        for rango in ['bajo', 'medio', 'alto']:
            dato = opciones[rango]
            if dato['inmueble'] is not None:
                fila = {
                    'AGENTE_ID': agente_id,
                    'PRECIO': dato['inmueble']['precio'],
                    'SUPERFICIE_CONSTRUIDA': dato['inmueble']['superficie_construida'],
                    'INGRESO_HOGAR': agente_data['ingreso_hogar'],
                    'MIEMBROS_HOGAR': agente_data['miembros_hogar'],
                    'GRUPO_AGENTE': agente_data['grupo_agente'],
                    'TIEMPO_VIAJE': dato['inmueble']['tiempo_viaje'],
                    'NRO_ESTACIONES': agente_data['num_estaciones_metro'],
                    'MACROZONA': agente_data['macro_zona'],
                    'SUPERFICIE_TERRENO': dato['inmueble']['superficie_terreno'],
                    'SUPERFICIE_COMERCIO': agente_data['sup_comercio'],
                    'SUPERFICIE_INDUSTRIA': agente_data['sup_industria'],
                    'GRUPO_INMUEBLE': rango,
                    'UTILIDAD': dato['utilidad'],
                    'CHOICE': 1 if rango == rango_elegido else 0,
                    'DISPONIBLE': dato['disponible']
                }
            else:
                # Para opciones no disponibles usamos valores promedio
                fila = {
                    'AGENTE_ID': agente_id,
                    'PRECIO': inmuebles_df[inmuebles_df['rango_precio'] == rango]['precio'].mean(),
                    'SUPERFICIE': inmuebles_df[inmuebles_df['rango_precio'] == rango]['superficie_construida'].mean(),
                    'INGRESO_ZONA': agente_data['ingreso_hogar'],
                    'TIEMPO_VIAJE': inmuebles_df[inmuebles_df['rango_precio'] == rango]['tiempo_viaje'].mean(),
                    'GRUPO_INMUEBLE': rango,
                    'UTILIDAD': dato['utilidad'],
                    'CHOICE': 0,
                    'DISPONIBLE': False
                }
            utilidades.append(fila)
    
    return pd.DataFrame(utilidades)
def main():
    # Cargar datos
    ruta = r"C:\\Users\\nicol\\OneDrive\\Escritorio\\LUMinCity\\LUMinCity"
    df = pd.read_csv(os.path.join(ruta, "base_agentes_localizados.csv"), sep=";")
    
    # Configurar parámetros
    num_agentes = df['AGENTE_ID'].nunique()
    num_inmuebles = df['INMUEBLE_ID'].nunique()
    num_zonas = df['ZONA_ID'].nunique()
    
    params = configurar_parametros_modelo(num_agentes)
    
    # Generar términos de error
    epsilon_choice = generar_gumbel((num_agentes, num_inmuebles, num_zonas))
    
    # Crear dataframe de utilidades con elecciones
    df_utilidades = crear_dataframe_utilidades(df, params)
    
    # Guardar resultados
    df_utilidades.to_csv(os.path.join(ruta, "resultados_choice.csv"), index=False, sep=";")
    
    # Mostrar resumen
    print("Registros en Choice:", len(df_utilidades))
    print("\nEjemplo de registros Choice:")
    print(df_utilidades[df_utilidades['AGENTE_ID'].isin([0,1])].head(6))
    print("\n¡Proceso completado con éxito!")

if __name__ == "__main__":
    main()