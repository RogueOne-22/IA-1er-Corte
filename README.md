# IA-1er-Corte - Laboratorio 2
Introduccion a IA

# 🧠 IA y Campo de Potenciales Artificiales

## 📚 Librerías utilizadas

## 🔹 Manipulación y Procesamiento de Datos
- **Pandas**: Librería esencial para manipular y analizar datos estructurados en DataFrames.  
- **Polars**: Alternativa a Pandas, optimizada en **Rust**, muy rápida para grandes volúmenes de datos.  
- **Dask**: Permite procesamiento distribuido y paralelo, ideal para datasets que no caben en memoria.  
- **Xarray**: Diseñada para trabajar con **datos multidimensionales** (ej. datos climáticos o científicos).  
- **GeoPandas**: Extensión de Pandas para trabajar con **datos geoespaciales**.  
- **Intake**: Librería para **gestión y carga de datos** desde múltiples fuentes con catálogos.  
- **Rapids (cuDF, cuML, cuGraph)**: Ecosistema de NVIDIA que acelera análisis de datos y ML con **GPU**.  
- **DuckDB**: Motor de base de datos en memoria, pensado como el “SQLite para análisis analítico”.  
- **Ibis**: Framework que unifica consultas en distintos motores (Pandas, DuckDB, BigQuery, etc.).  
- **NetworkX**: Para análisis y visualización de **redes y grafos**.  

---

## 🔹 Visualización de Datos
- **Matplotlib**: La librería clásica para gráficos estáticos y personalizables.  
- **Bokeh**: Orientada a la **web**, permite gráficos interactivos en HTML/JS.  
- **Plotly**: Visualizaciones **interactivas** de alta calidad, con soporte para dashboards.  
- **HoloViews**: API declarativa que simplifica la visualización sobre diferentes backends.  
- **Datashader**: Para visualizar **datasets masivos** mediante rasterización.  
- **Streamlit**: Framework para crear **apps interactivas** de ciencia de datos de forma rápida.  
- **.Plot() API**: Interfaz de alto nivel común en librerías de datos (Pandas, Polars, etc.) para generar gráficos básicos rápidamente (`df.plot()`).  

---

## 🤖 Conceptos teóricos

### 🔹 Agente Inteligente
Un **agente inteligente** es una entidad que percibe su entorno mediante sensores, procesa la información y ejecuta acciones que modifican su entorno. No necesariamente es “inteligente” en el sentido humano, sino que actúa de manera **racional** en función de los datos de entrada.

### 🔹 Cambio de Potenciales en IA
El **campo de potenciales artificiales** es un modelo utilizado en robótica móvil para la navegación:
- **Fuerza atractiva**: atrae al agente hacia el objetivo.  
- **Fuerza repulsiva**: aleja al agente de los obstáculos.  
- La suma de estas fuerzas genera un **campo de potencial total**.  
- El agente se mueve en dirección descendente del potencial (mínimos energéticos).  

Este método es simple y eficiente, aunque puede tener problemas con mínimos locales que atrapan al agente.

---

## 🧩 Resumen del Código de Campo de Potenciales
1. **Definición de funciones**  
   - `calcular_potencial`: combina el atractivo y el repulsivo.  
   - `calcular_direccion`: obtiene la dirección de movimiento mediante gradiente.  
   - `visualizar_campo_potencial`: muestra el campo y la trayectoria.  

2. **Parámetros iniciales**  
   - Posición del agente: `(0,0)`  
   - Objetivo: `(10,10)`  
   - Obstáculos: puntos definidos en el espacio.  

3. **Simulación**  
   - En cada paso, el agente se mueve un poco hacia la dirección de menor potencial.  
   - Se guarda la trayectoria.  

4. **Visualización**  
   - El campo de potencial se muestra en colores.  
   - Obstáculos como ❌ negras.  
   - Objetivo como ⚪ rojo.  
   - El recorrido del agente se anima paso a paso.  

---

## 🎯 Finalidad
Este programa ejemplifica cómo un **agente inteligente** puede navegar hacia una meta evitando obstáculos mediante un enfoque de **IA basada en campos de potenciales artificiales**, útil en robótica móvil y planificación de trayectorias.  

