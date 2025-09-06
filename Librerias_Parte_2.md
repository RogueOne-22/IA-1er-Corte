# 📊  Librerías de Visualización en Python
---

## **🚀 1. Matplotlib**

Librerías mas comunes de de visualización en Python. Es una herramienta poderosa y fundamental para crear gráficos estáticos y personalizables.

* **Características Principales:**
  
    * **Versatilidad:** Puedes crear desde simples gráficos de líneas hasta complejos mapas de calor y visualizaciones 3D.
    * **Personalización:** cada detalle es configurable. Ideal para publicaciones científicas y proyectos que exigen precisión. 🔬
    * **Dos Estilos de Código:** Ofrece el estilo **`pyplot`** (muy similar a MATLAB) para un trazado rápido y sencillo, y la **API orientada a objetos**.
      
* **Casos de Uso Típicos:**
    * Generar gráficos para **artículos científicos** y **presentaciones académicas**. ✍️
    * Realizar **análisis de datos exploratorio** cuando necesitas entender la estructura de un dataset rápidamente. 🕵️‍♂️
    * Crear **gráficos personalizados** que no se ajustan a los estilos predefinidos de otras librerías. ✨
 
* **Ejemplo**
  ```
  ```python
  import matplotlib.pyplot as plt
  
  x = [1, 2, 3, 4, 5]
  y = [2, 4, 6, 8, 10]
  
  plt.plot(x, y, marker='o')
  plt.title("Gráfico simple con Matplotlib")
  plt.xlabel("Eje X")
  plt.ylabel("Eje Y")
  plt.show()
  ```
      
* **Enlaces de Consulta:**
    * [Página oficial de Matplotlib](https://matplotlib.org/)
    * [Tutorial de Matplotlib para principiantes](https://www.datacamp.com/es/tutorial/matplotlib-tutorial-python)

---

### **2. Bokeh**

🌐 A diferencia de las librerías que solo crean imágenes, Bokeh produce código HTML y JavaScript, permitiendo que los usuarios interactúen con los datos directamente desde su navegador. 

* **Características Principales:**
    * **Diseñado para la Web:** Los gráficos de Bokeh son interactivos por naturaleza y se visualizan en cualquier navegador sin necesidad de plugins. 💻
    * **Manejo de Grandes Datos:**  puede manejar datasets enormes de manera eficiente, lo que lo hace una excelente opción para **Big Data**.
    * **Herramientas Interactivas:** Ofrece funciones de zoom, paneo, selección de datos y `hover` (mostrar información al pasar el ratón) de forma nativa. 🖱️
      
* **Casos de Uso Típicos:**
    * Crear **dashboards interactivos** y aplicaciones web para visualizar datos en tiempo real. 📈
    * Desarrollar herramientas para **explorar datos masivos** sin tener que cargar todo el dataset en la memoria.
    * Compartir **visualizaciones interactivas** con equipos o clientes. 🤝
 
    ```
    Python
    from bokeh.plotting import figure, show

    p = figure(title="Gráfico interactivo con Bokeh", x_axis_label="x", y_axis_label="y")
    p.line([1, 2, 3, 4], [2, 5, 8, 12], line_width=2)
    
    show(p)
    ```
      
* **Enlaces de Consulta:**
    * [Documentación de Bokeh](https://docs.bokeh.org/)

---

### **3. Plotly**

Su punto fuerte es la **interactividad** y su API de alto nivel, `Plotly Express`, que te permite crear visualizaciones complejas con una sola línea de código.

* **Características Principales:**
    * **Gráficos Impresionantes:** Plotly crea gráficos interactivos listos para publicar, con animaciones y transiciones suaves. ✨
    * **`Plotly Express`:** Esta API es un atajo mágico. Con una sola función (como `px.line()` o `px.scatter()`), puedes generar un gráfico completo y hermoso a partir de un DataFrame. 🪄
    * **Soporte Multiplataforma:** Disponible no solo para Python, sino también para R y MATLAB, facilitando la colaboración entre equipos.
      
* **Casos de Uso Típicos:**
    * Crear **dashboards dinámicos** y **presentaciones ejecutivas** que impresionen. 📊
    * Generar **visualizaciones interactivas** para artículos de blog o informes de análisis.
    * Trazar **mapas geoespaciales** con datos, como rutas de viaje o ubicaciones de ventas. 🗺️
    ```
    Python
    import plotly.express as px

    df = px.data.iris()  # Dataset de ejemplo
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                     title="Gráfico Scatter con Plotly")
    fig.show()
    ```

* **Enlaces de Consulta:**
    * [Guía de Plotly Express](https://plotly.com/python/plotly-express/)
    * [Documentación de Plotly en Python](https://plotly.com/python/)

---

### ** 🚀 4. Streamlit**

 **Framework para construir aplicaciones web**. Es en donde tus gráficos de Matplotlib, Plotly y Bokeh se pueden visualizar. Permite convertir  scripts de Python en aplicaciones web interactivas.

* **Características Principales:**
    * **Simplicidad Extrema:** Olvídate del HTML, CSS y JavaScript. Con unas pocas líneas de código, tu script de Python se convierte en una app funcional. 🥳
    * **Widgets Interactivos:** Incluye `sliders`, `buttons` y `checkboxes` que puedes usar para que los usuarios interactúen con tus datos en tiempo real.
    * **`Streamlit Cloud`:** Ofrece una manera sencilla de desplegar tus aplicaciones en la nube para que cualquiera pueda verlas. ☁️
      
* **Casos de Uso Típicos:**
    * **Desarrollar prototipos rápidos** para proyectos de ciencia de datos. 🤖
    * Crear **dashboards para compartir** con equipos de negocio que no programan.
    * Construir **herramientas interactivas** para demostrar el funcionamiento de modelos de machine learning. 💡
 
    ```
    Python
    import streamlit as st
    import pandas as pd
    
    st.title("Ejemplo con Streamlit")
    data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    st.line_chart(data)

    streamlit run archivo.py
    ```

* **Enlaces de Consulta:**
    * [Página oficial de Streamlit](https://streamlit.io/)
    * [Tutoriales de Streamlit](https://docs.streamlit.io/)

---

### **🧠 5. HoloViews**

 Se enfoca en la **descripción de los datos** y sus relaciones, y luego usa un motor de trazado (como Bokeh o Matplotlib) para renderizar la visualización.

* **Características Principales:**
    * **API Declarativa:** Su código es conciso y expresa lo que quieres mostrar, no cómo hacerlo. Esto reduce drásticamente las líneas de código.
    * **Independiente del Backend:** se puede cambiar de Bokeh a Plotly con una sola línea de código.
    * **Exploración de Datos Fácil:** Es ideal para la exploración, ya que genera visualizaciones interactivas y complejas con un esfuerzo mínimo.
      
* **Casos de Uso Típicos:**
    * Realizar **análisis exploratorio de datos** de alto nivel.
    * **Experimentar** con diferentes tipos de gráficos para encontrar la mejor manera de representar tus datos.
    * Construir **visualizaciones complejas** que combinan diferentes tipos de gráficos. 🧩

    ```
    Python
    import holoviews as hv
    hv.extension("bokeh")
    
    data = [(1, 2), (2, 3), (3, 5)]
    scatter = hv.Scatter(data, "x", "y")
    scatter
    ```
    
* **Enlaces de Consulta:**
    * [Documentación de HoloViews](https://holoviews.org/)
    * [Galería de ejemplos de HoloViews](https://holoviews.org/gallery/index.html)

---

### **🗺️ 6. Datashader**

 No es una librería de visualización en sí, sino un **pipeline de procesamiento** que convierte datasets masivos en imágenes para que puedas ver patrones.

* **Características Principales:**
    * **Visualización de Big Data:** Su principal objetivo es hacer que la visualización de datos masivos sea posible y eficiente. 📦
    * **Rasterización:** En lugar de trazar cada punto, Datashader "rasteriza" los datos en una grilla de píxeles, lo que permite visualizar la densidad y los patrones generales.
    * **Integración Perfecta:** Se integra con librerías como Bokeh y HoloViews para el renderizado final de las imágenes generadas.
      
* **Casos de Uso Típicos:**
    * Analizar **datos geoespaciales** con millones de puntos de GPS. 🌍
    * Visualizar **series temporales masivas** o datos de sensores. 📈
    * Identificar **patrones ocultos** en datasets extremadamente grandes que serían imposibles de trazar con otras librerías. 🔍
 
      ```
      phyton
      import datashader as ds
      import pandas as pd
      import numpy as np
      
      # Dataset masivo
      n = 100000
      df = pd.DataFrame({'x': np.random.normal(size=n), 'y': np.random.normal(size=n)})
      
      canvas = ds.Canvas(plot_width=400, plot_height=400)
      agg = canvas.points(df, 'x', 'y')
      ```
    
* **Enlaces de Consulta:**
    * [Página oficial de Datashader](https://datashader.org/)

 ### Comparacion 

| Librería       | Estático  | Interactivo | Web/Dashboards             | Big Data | Facilidad de Uso |
| -------------- | --------- | ----------- | -------------------------- | -------- | ---------------- |
| **Matplotlib** | ✅ Sí      | ❌ No        | ❌ No                       | ❌ No     | ⚡ Media          |
| **Bokeh**      | ❌ No      | ✅ Sí        | ✅ Sí                       | ⚡ Medio  | ⚡ Media          |
| **Plotly**     | ⚡ Parcial | ✅ Sí        | ✅ Sí                       | ❌ No     | ✅ Alta           |
| **Streamlit**  | ❌ No      | ✅ Sí        | ✅ Sí                       | ❌ No     | ✅ Muy Alta       |
| **HoloViews**  | ⚡ Parcial | ✅ Sí        | ✅ Sí                       | ⚡ Medio  | ✅ Alta           |
| **Datashader** | ❌ No      | ⚡ Parcial   | ✅ Sí (con Bokeh/HoloViews) | ✅ Sí     | ⚡ Media          |
