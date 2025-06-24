#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis de Consumo de Azúcar en América, Asia y Europa (1960-2023)

Pregunta principal:
¿Cómo ha evolucionado el consumo de azúcar en América, Asia y Europa entre 1960 y 2023, 
y cómo se espera que evolucione en los próximos años?

Preguntas relacionadas:
- ¿Cómo afectan las políticas gubernamentales (impuestos, subsidios, campañas educativas) 
  al consumo de azúcar per cápita?
- ¿Qué relación existe entre las condiciones climáticas y el consumo de azúcar per cápita 
  en diferentes continentes?
- ¿Qué continente presenta mayores tasas de consumo de azúcar?
- ¿Cómo ha variado la relación entre el consumo de azúcar y las tasas de obesidad 
  a lo largo del tiempo en América, Asia y Europa?

Hipótesis general: 
Existe una tendencia creciente en el consumo de azúcar per cápita a nivel global entre 1960 y 2023, 
con América mostrando patrones de mayor consumo en comparación con Asia y Europa, influenciados por 
factores económicos, climáticos y de políticas gubernamentales.

Hipótesis nula: 
No existe una tendencia clara ni diferencias significativas en el consumo de azúcar per cápita 
entre América, Asia y Europa entre 1960 y 2023, ni influencia significativa de factores 
económicos, climáticos ni de políticas gubernamentales.
"""

# =============================================================================
# IMPORTAR LIBRERÍAS NECESARIAS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CARGA Y EXPLORACIÓN INICIAL DE DATOS
# =============================================================================

# Cargar el dataset
df = pd.read_csv('sugar_consumption_dataset.csv')
print(f"Dataset cargado exitosamente. Shape: {df.shape}")

# Exploración inicial
print("\n=== PRIMERAS 5 FILAS ===")
print(df.head())

print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
print(df.describe())

print("\n=== VALORES FALTANTES ===")
print(df.isnull().sum())

# =============================================================================
# LIMPIEZA DEL DATASET
# =============================================================================
print("\n" + "="*50)
print("LIMPIEZA DEL SET DE DATOS")
print("="*50)

# 1. MANEJO DE DATOS FALTANTES
print("\n1. Manejo de datos faltantes")
print("Usando SimpleImputer para rellenar valores faltantes por la moda")

imputer = sk.impute.SimpleImputer(strategy='most_frequent')
df[['Gov_Tax']] = imputer.fit_transform(df[['Gov_Tax']])
df[['Education_Campaign']] = imputer.fit_transform(df[['Gov_Tax']])

# Definir variables numéricas y categóricas
variables_numericas = [
    'Year', 'Population', 'Per_Capita_Sugar_Consumption',
    'Total_Sugar_Consumption', 'Sugar_From_Sugarcane', 'Sugar_From_Beet',
    'Sugar_From_HFCS', 'Sugar_From_Other','Avg_Daily_Sugar_Intake',
    'Obesity_Rate','Sugar_Imports', 'Sugar_Exports',
    'Avg_Retail_Price_Per_Kg', 'Gov_Tax','Gov_Subsidies', 'Education_Campaign',
    'Climate_Conditions', 'Sugarcane_Production_Yield'
]

variables_categoricas = ['Country', 'Continent']

# 2. ELIMINACIÓN DE VALORES DUPLICADOS
print("\n2. Eliminación de valores duplicados")
print(f'El set de datos tiene {df.shape[0]} filas y {df.shape[1]} columnas')
df.drop_duplicates(inplace=True)
print(f'El nuevo set de datos tiene {df.shape[0]} filas y {df.shape[1]} columnas')

# 3. MANEJO DE VALORES ATÍPICOS
print("\n3. Análisis de valores atípicos para Total_Sugar_Consumption")

# Análisis de outliers para Total_Sugar_Consumption
outliers_tsc = df['Total_Sugar_Consumption']

x_min = outliers_tsc.min()
x_max = outliers_tsc.max()
q1 = outliers_tsc.quantile(0.25)
q2 = outliers_tsc.quantile(0.50)
q3 = outliers_tsc.quantile(0.75)

print(f'Min: {x_min}')
print(f'Q1: {q1}')
print(f'Q2: {q2}')
print(f'Q3: {q3}')
print(f'Max: {x_max}')

iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

print(f'Rango intercuartil: {iqr}')
print(f'Limite inferior: {limite_inferior}')
print(f'Limite superior: {limite_superior}')

# Eliminar outliers
print(f'Dataset antes de eliminar valores atípicos: {df.shape[0]} filas y {df.shape[1]} columnas')
df = df[(df['Total_Sugar_Consumption'] >= limite_inferior) & (df['Total_Sugar_Consumption'] <= limite_superior)]
print(f'Dataset después de eliminar valores atípicos: {df.shape[0]} filas y {df.shape[1]} columnas')

# 4. CORRECCIÓN DE ERRORES TIPOGRÁFICOS
print("\n4. Corrección de errores tipográficos")
for variable in variables_categoricas:
    print(f"{variable}: {df[variable].unique()}")

# Unificar América del Norte y América del Sur como "America"
df['Continent'] = df['Continent'].replace('North America', 'America')
df['Continent'] = df['Continent'].replace('South America', 'America')

print("Continentes después de la corrección:", df['Continent'].unique())

# =============================================================================
# ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================
print("\n" + "="*50)
print("ANÁLISIS EXPLORATORIO DE DATOS")
print("="*50)

# Estadísticas descriptivas de variables numéricas
print("\n=== ESTADÍSTICAS DESCRIPTIVAS DE VARIABLES NUMÉRICAS ===")
print(df[variables_numericas].describe())

# =============================================================================
# FUNCIONES AUXILIARES PARA VISUALIZACIONES
# =============================================================================

def crear_graficos_paises(df, colores, tipo):
    """Crear gráficos de líneas para países"""
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=df, x='Year', y='Per_Capita_Sugar_Consumption', 
                palette=colores, hue='Country', marker='o', errorbar=None)
    plt.xlabel('Año')
    plt.ylabel('Consumo de azúcar por kg')

def crear_graficos_continentes(df, variable, color):
    """Crear gráficos para análisis de políticas gubernamentales"""
    plt.figure(figsize=(16, 5))
    
    # Boxplot general
    plt.subplot(1,2,1)
    sns.boxplot(data=df, x=variable, y='Per_Capita_Sugar_Consumption', 
               palette=['red', 'green'], hue=variable)
    plt.title(f'{variable}')
    plt.xticks([0, 1], ['No', 'Si'], rotation=90)

    # Gráfico de pastel
    plt.subplot(1, 2, 2)
    counts = df[variable].value_counts()
    labels = ['No', 'Sí']
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
    plt.title(f'Proporción de {variable}')
    plt.show()

    # Análisis por país
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x=variable, y='Per_Capita_Sugar_Consumption', 
               palette=color, hue='Country')
    plt.title(f'{variable}')
    plt.xticks([0, 1], ['No', 'Si'], rotation=90)

    plt.subplot(1, 2, 2)
    ax = sns.barplot(data=df, x=variable, y='Per_Capita_Sugar_Consumption', 
                    palette=color, hue='Country')
    for i in ax.containers:
        ax.bar_label(i, fmt='%.0f', label_type='edge', padding=3)
    plt.title(f'{variable}')
    plt.xticks([0, 1], ['No', 'Si'], rotation=90)
    plt.tight_layout()
    plt.show()

def crear_graficos_climaticos(df, agrupar, colores):
    """Crear gráficos para análisis climático"""
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='Climate_Conditions', y='Per_Capita_Sugar_Consumption', 
               hue=agrupar, palette=colores)
    plt.xlabel('Condiciones climáticas')
    plt.xticks([0, 1, 2, 3, 4], ['1.Tropical', '2.Árido', '3.Templado', '4.Frío', '5.Montaña'])
    plt.ylabel('Consumo de azúcar per cápita (kg)')
    plt.title('Distribución por condiciones climáticas')
    plt.legend(title=agrupar, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.subplot(1, 2, 2)
    ax = sns.barplot(data=df, x='Climate_Conditions', y='Per_Capita_Sugar_Consumption', 
                    hue=agrupar, palette=colores)
    plt.xlabel('Condiciones climáticas')
    plt.xticks([0, 1, 2, 3, 4], ['1.Tropical', '2.Árido', '3.Templado', '4.Frío', '5.Montaña'])
    plt.ylabel('Consumo promedio de azúcar (kg)')
    plt.title('Promedio por condiciones climáticas')
    plt.legend(title=agrupar, bbox_to_anchor=(1.05, 1), loc='upper left')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', label_type='edge', padding=3)

    plt.tight_layout()
    plt.show()

def crear_mapa(df, variable, color_densidad, titulo):
    """Crear mapas interactivos con Plotly"""
    fig = px.choropleth(
        df,
        locations='Country',
        locationmode='country names',
        color=color_densidad,
        hover_name='Country',
        animation_frame='Year',
        color_continuous_scale='Reds',
        title=titulo
    )
    fig.show()

def crear_barras(df, variable_x, variable_y, titulo):
    """Crear gráficos de barras"""
    plt.figure(figsize=(20, 6))
    ax = sns.barplot(data=df, x=variable_x, y=variable_y, palette='dark:red', hue=variable_x)
    
    for i in ax.containers:
        ax.bar_label(i, fmt='%.0f', label_type='edge', padding=3)
    plt.xticks(rotation=90)
    plt.title(titulo)
    plt.xlabel("Country")
    plt.tight_layout()
    plt.show()

# =============================================================================
# ANÁLISIS 1: EVOLUCIÓN DEL CONSUMO DE AZÚCAR A LO LARGO DEL TIEMPO
# =============================================================================
print("\n=== ANÁLISIS 1: EVOLUCIÓN DEL CONSUMO DE AZÚCAR ===")

# Consumo total por año
consumo_total = df.groupby('Year')['Total_Sugar_Consumption'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=consumo_total, x='Year', y='Total_Sugar_Consumption', marker='o')
plt.title('Consumo de azúcar por año a lo largo del tiempo')
plt.xlabel('Año')
plt.ylabel('Consumo de azúcar por Tonelada')
plt.show()

# Separar por continentes
df_america = df[df['Continent'] == 'America']
df_asia = df[df['Continent'] == 'Asia']
df_europa = df[df['Continent'] == 'Europe']

# Comparación entre continentes
plt.figure(figsize=(10, 4))
sns.lineplot(data=df_america, x='Year', y='Total_Sugar_Consumption', color='red', marker='o', errorbar=None)
sns.lineplot(data=df_asia, x='Year', y='Total_Sugar_Consumption', color='orange', marker='o', errorbar=None)
sns.lineplot(data=df_europa, x='Year', y='Total_Sugar_Consumption', color='green', marker='o', errorbar=None)
plt.legend(['America', 'Asia', 'Europa'])
plt.title('Consumo de azúcar por año a lo largo del tiempo: America vs Asia vs Europa')
plt.xlabel('Año')
plt.ylabel('Consumo de azúcar por Tonelada')
plt.show()

print("""
CONCLUSIONES DEL ANÁLISIS TEMPORAL:

América:
- Presenta las mayores fluctuaciones a lo largo de los años
- Fuerte caída alrededor del año 2002
- Desde 2010, el consumo tiende a estabilizarse con ligera disminución hacia 2020-2023

Asia:
- Mantiene una tendencia más estable, con menos picos extremos que América
- Ligera tendencia creciente desde finales de los 90 hacia 2010
- A partir de 2010 se mantiene estable

Europa:
- Presenta cierta estabilidad con variaciones menos significativas que América
- Ligera tendencia descendente desde 1990 en adelante
""")

# =============================================================================
# ANÁLISIS 2: CONSUMO POR PAÍSES EN CADA CONTINENTE
# =============================================================================
print("\n=== ANÁLISIS 2: CONSUMO POR PAÍSES ===")

# Colores para cada continente
colores_paises_america = {'USA': 'darkblue', 'Mexico': 'darkgreen', 'Brazil': 'orange'}
colores_paises_asia = {'Japan': 'gray', 'Indonesia': 'darkred', 'China': 'red', 'India': 'orange'}
colores_paises_europa = {'France': 'Blue', 'Germany': 'darkgray', 'Russia': 'red'}

# América
print("\n--- AMÉRICA ---")
crear_graficos_paises(df_america, colores_paises_america, 'lineplot')
plt.title('Consumo de azúcar per cápita en los diferentes países de América')
plt.show()

# Asia
print("\n--- ASIA ---")
crear_graficos_paises(df_asia, colores_paises_asia, 'lineplot')
plt.title('Consumo de azúcar per cápita en los diferentes países de Asia')
plt.show()

# Europa
print("\n--- EUROPA ---")
crear_graficos_paises(df_europa, colores_paises_europa, 'lineplot')
plt.title('Consumo de azúcar per cápita en los diferentes países de Europa')
plt.show()

# =============================================================================
# ANÁLISIS 3: IMPACTO DE POLÍTICAS GUBERNAMENTALES
# =============================================================================
print("\n=== ANÁLISIS 3: IMPACTO DE POLÍTICAS GUBERNAMENTALES ===")

gov = ['Gov_Tax', 'Gov_Subsidies', 'Education_Campaign']

# Análisis para América
print("\n--- POLÍTICAS GUBERNAMENTALES EN AMÉRICA ---")
for v in gov:
    print(f"\nAnalizando: {v}")
    crear_graficos_continentes(df_america, v, colores_paises_america)

# Análisis para Asia
print("\n--- POLÍTICAS GUBERNAMENTALES EN ASIA ---")
for v in gov:
    print(f"\nAnalizando: {v}")
    crear_graficos_continentes(df_asia, v, colores_paises_asia)

# Análisis para Europa
print("\n--- POLÍTICAS GUBERNAMENTALES EN EUROPA ---")
for v in gov:
    print(f"\nAnalizando: {v}")
    crear_graficos_continentes(df_europa, v, colores_paises_europa)

# =============================================================================
# ANÁLISIS 4: RELACIÓN CON CONDICIONES CLIMÁTICAS
# =============================================================================
print("\n=== ANÁLISIS 4: RELACIÓN CON CONDICIONES CLIMÁTICAS ===")

# Análisis comparativo entre continentes
continentes = df[(df['Continent'] == 'America') | (df['Continent'] == 'Asia') | (df['Continent'] == 'Europe')][
    ['Continent', 'Country', 'Year', 'Per_Capita_Sugar_Consumption', 'Climate_Conditions']].dropna()

colores_continentes = {'America': 'red', 'Asia': 'orange', 'Europe': 'green'}

print("\n--- COMPARACIÓN ENTRE CONTINENTES ---")
crear_graficos_climaticos(continentes, 'Continent', colores_continentes)

print("\n--- ANÁLISIS POR PAÍS EN AMÉRICA ---")
crear_graficos_climaticos(df_america, 'Country', colores_paises_america)

print("\n--- ANÁLISIS POR PAÍS EN ASIA ---")
crear_graficos_climaticos(df_asia, 'Country', colores_paises_asia)

print("\n--- ANÁLISIS POR PAÍS EN EUROPA ---")
crear_graficos_climaticos(df_europa, 'Country', colores_paises_europa)

# =============================================================================
# ANÁLISIS 5: DISTRIBUCIÓN GEOGRÁFICA
# =============================================================================
print("\n=== ANÁLISIS 5: DISTRIBUCIÓN GEOGRÁFICA ===")

# Consumo total de azúcar
print("\n--- CONSUMO TOTAL DE AZÚCAR ---")
mapa_consumo = df[(df['Continent'] == 'America') | (df['Continent'] == 'Asia') | (df['Continent'] == 'Europe')][
    ['Continent', 'Country', 'Year', 'Total_Sugar_Consumption']].dropna()

df_mapa_consumo = mapa_consumo.groupby(['Continent', 'Country', 'Year'], as_index=False)['Total_Sugar_Consumption'].mean()

# Crear mapa interactivo (comentado para ejecución en script)
# crear_mapa(df_mapa_consumo, 'Total_Sugar_Consumption', 'Total_Sugar_Consumption', 'Consumo total de azúcar a lo largo del tiempo')

# Gráficos de barras
ordenar_consumo = df_mapa_consumo.groupby(['Continent', 'Country'], as_index=False)['Total_Sugar_Consumption'].mean().sort_values(by='Total_Sugar_Consumption', ascending=False)

crear_barras(ordenar_consumo, 'Continent', 'Total_Sugar_Consumption', 'Consumo total de azúcar en toneladas por continente')
crear_barras(ordenar_consumo, 'Country', 'Total_Sugar_Consumption', 'Consumo total de azúcar en toneladas por país')

print("""
RESPUESTA A LA PREGUNTA: ¿Qué continente presenta mayores tasas de consumo de azúcar?

Europa es el principal consumidor de azúcar, seguido de Asia y luego América.
América aparece en la última posición del grupo, con el consumo total más bajo.
""")

# Tasa de obesidad
print("\n--- TASA DE OBESIDAD ---")
mapa_obesidad = df[(df['Continent'] == 'America') | (df['Continent'] == 'Asia') | (df['Continent'] == 'Europe')][
    ['Country', 'Year', 'Obesity_Rate']].dropna()

df_mapa_obesidad = mapa_obesidad.groupby(['Country', 'Year'], as_index=False)['Obesity_Rate'].mean()

# crear_mapa(df_mapa_obesidad, 'Obesity_Rate', 'Obesity_Rate', 'Tasa de obesidad a lo largo del tiempo')

ordenar_obesidad = df_mapa_obesidad.groupby('Country', as_index=False)['Obesity_Rate'].mean().sort_values(by='Obesity_Rate', ascending=False)

crear_barras(ordenar_obesidad, 'Country', 'Obesity_Rate', 'Tasa de obesidad por país')

# =============================================================================
# ANÁLISIS 6: MATRIZ DE CORRELACIÓN
# =============================================================================
print("\n=== ANÁLISIS 6: MATRIZ DE CORRELACIÓN ===")

corr = df.corr(numeric_only=True)
plt.figure(figsize=(20, 8))
sns.heatmap(corr, annot=True, cmap='inferno')
plt.title('Matriz de correlación')
plt.show()

# =============================================================================
# ANÁLISIS 7: MODELO PREDICTIVO
# =============================================================================
print("\n=== ANÁLISIS 7: MODELO PREDICTIVO ===")

# Preparar datos para regresión
df_regresion = df[['Continent', 'Per_Capita_Sugar_Consumption', 'Year', 'Climate_Conditions']].dropna()
df_regresion = pd.get_dummies(df_regresion, columns=['Continent'], drop_first=True)

X = df_regresion.drop(columns='Per_Capita_Sugar_Consumption')
y = df_regresion['Per_Capita_Sugar_Consumption']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Obtener coeficientes
coeficientes = modelo.coef_
intercepto = modelo.intercept_

print(f'Coeficientes: {coeficientes}')
print(f'Intercepto: {intercepto}')

# Predicción para 2030
consumo_2030 = {
    'Year': 2030,
    'Climate_Conditions': 4,
    'Continent_Africa': 0,
    'Continent_Asia': 0,
    'Continent_Europe': 0,
    'Continent_Oceania': 0,
    'Continent_America': 1
}

df_consumo_2030 = pd.DataFrame([consumo_2030])[X.columns]
prediccion = modelo.predict(df_consumo_2030)

print(f'Predicción de consumo per cápita para América en 2030: {prediccion[0]:.2f} kg')

# Evaluar modelo
y_pred = modelo.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'R² del modelo: {r2:.4f}')

# =============================================================================
# CONCLUSIONES FINALES
# =============================================================================
print("\n" + "="*70)
print("CONCLUSIONES FINALES")
print("="*70)

print("""
RESPUESTAS A LAS PREGUNTAS DE INVESTIGACIÓN:

1. ¿Cómo ha evolucionado el consumo de azúcar per cápita en América, Asia y Europa entre 1960 y 2023?
   - América: Mayor volatilidad, caída significativa en 2002, estabilización desde 2010
   - Asia: Tendencia más estable, crecimiento moderado desde los 90
   - Europa: Relativa estabilidad, ligera tendencia descendente desde 1990

2. ¿Cómo afectan las políticas gubernamentales al consumo de azúcar per cápita?
   - Los impuestos muestran efectos mixtos según el país
   - Los subsidios pueden tener efectos variables
   - Las campañas educativas muestran impacto limitado en el consumo total

3. ¿Qué relación existe entre las condiciones climáticas y el consumo de azúcar per cápita?
   - No se observa una relación clara y consistente entre clima y consumo
   - Las diferencias son más marcadas entre países que entre tipos de clima

4. ¿Qué continente presenta mayores tasas de consumo de azúcar?
   - Europa lidera el consumo total, seguido de Asia y América

5. ¿Cómo ha variado la relación entre el consumo de azúcar y las tasas de obesidad?
   - Francia, México y USA presentan las tasas más altas de obesidad (23%)
   - La correlación entre consumo de azúcar y obesidad requiere análisis adicional

VALIDACIÓN DE HIPÓTESIS:
La hipótesis general se confirma parcialmente. Existe variabilidad en las tendencias de consumo
entre continentes, con América mostrando mayor volatilidad pero no necesariamente mayor consumo
que Europa. Los factores gubernamentales y climáticos muestran influencia limitada y variable.
""")

print("\nAnálisis completado. Todos los gráficos han sido generados y las conclusiones presentadas.")
