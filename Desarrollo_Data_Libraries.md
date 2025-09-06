# 📊 ibrerías Python para Data Science

> **Versión:** 1.0| **Fecha:** Septiembre 2025 | **Autor:** Paula Sandoval
> **Descripción:** Resumen completo de las 10 librerías más importantes para Data Science en Python.
---
### 🏆 **Mejores 3 liberias**
1. **Pandas** 🐼 - El estándar universal
2. **Polars** ⚡ - El futuro del rendimiento  
3. **DuckDB** 🦆 - La revolución de SQL analítico

### 💡 **Tendencias 2025**
- Polars y DuckDB lideran en velocidad
- RAPIDS hace accesible la computación GPU
- **Backend Unification:** Ibis permite código portable
- **Domain Specialization:** Herramientas específicas para cada necesidad

---

### 1. 🚀 **Dask - Computación Paralela y Distribuida**

#### ✨ **Características**
- **📈 Escalabilidad Automática:** De laptop a cluster sin modificaciones
- **🔄 API Idéntica:** `import dask.dataframe as dd` → todo igual que pandas
- **🧠 Lazy Evaluation:** Optimiza antes de ejecutar
- **🔗 Ecosistema:** Integra con NumPy, Pandas, Scikit-Learn

#### 💻 **Código de Ejemplo**
```python
import dask.dataframe as dd

# Leer archivo de 10GB como si fuera pequeño
df = dd.read_csv('massive_data.csv')

# Operaciones familiares
result = (df
    .groupby('category')
    .sales.mean()
    .compute()  # Solo aquí se ejecuta
)

# Paralelización automática en 8 cores
df.to_parquet('output/', engine='pyarrow')
```

#### 📊 **Casos de Uso Perfectos**
- ✅ Datasets > 1GB que no caben en memoria
- ✅ Migrar código pandas existente a big data
- ✅ Procesamiento en múltiples cores/máquinas
- ✅ Análisis de series temporales grandes

#### 🔗 **Enlaces Esenciales**
- 🌐 **Web:** https://www.dask.org/
- 📖 **Docs:** https://docs.dask.org/
- 🎓 **Tutorial:** https://tutorial.dask.org/
---

### 2. ⚡ **Polars - DataFrame Ultrarrápido**

#### ✨ **Características**
- **🦀 Backend Rust:** Velocidad extrema sin sacrificar funcionalidad
- **🧠 Query Optimizer:** Como tener un DBA personal
- **💾 Memoria Eficiente:** Maneja datasets 10x más grandes
- **📝 API Moderna:** Sintaxis más expresiva que pandas

#### 💻 **Código de ejemplo**
```python
import polars as pl

# Lazy evaluation con optimización automática
result = (
    pl.scan_csv("sales_2023.csv")
    .filter(pl.col("amount") > 1000)
    .group_by(["region", "product"])
    .agg([
        pl.col("amount").sum().alias("total_sales"),
        pl.col("quantity").mean().alias("avg_quantity"),
        pl.count().alias("transactions")
    ])
    .sort("total_sales", descending=True)
    .collect()  # Query optimizada se ejecuta aquí
)

# Expresiones complejas son simples
df_enhanced = df.with_columns([
    # Crear múltiples columnas en una pasada
    pl.when(pl.col("amount") > pl.col("amount").quantile(0.9))
      .then(pl.lit("Premium"))
      .when(pl.col("amount") > pl.col("amount").quantile(0.7))
      .then(pl.lit("Standard"))
      .otherwise(pl.lit("Basic"))
      .alias("customer_tier"),
      
    # Operaciones de ventana eficientes
    pl.col("amount").sum().over("region").alias("region_total")
])
```

#### 📊 **Cuándo Usar Polars**
- ✅ Nuevos proyectos que priorizan velocidad
- ✅ ETL pesado con millones de filas
- ✅ Análisis financiero de alta frecuencia
- ✅ Cuando pandas se vuelve lento

#### 🎯 **Migración desde Pandas**
```python
# PANDAS (lento)
df.groupby('category')['sales'].mean().reset_index()

# POLARS (30x más rápido)
df.group_by('category').agg(pl.col('sales').mean())
```

#### 🔗 **Enlaces Esenciales**
- 🌐 **Web:** https://pola.rs/
- 📖 **Docs:** https://docs.pola.rs/
- 💻 **GitHub:** https://github.com/pola-rs/polars
- 🏆 **Benchmarks:** https://pola.rs/posts/benchmarks/

---

### 3. 🏎️ **RAPIDS - Aceleración GPU para Data Science**

#### ✨ **Características**
- **🎮 GPU NVIDIA:** Aprovecha miles de cores en paralelo
- **🔄 API Compatible:** Drop-in replacement para pandas/sklearn
- **📊 Suite Completa:** DataFrame (cuDF) + ML (cuML) + Grafos (cuGraph)
- **💰 ROI Inmediato:** Una GPU = 10-100 CPUs en analytics

#### 💻 **Código de ejemplo**
```python
import cudf  # GPU DataFrame
import cuml  # GPU Machine Learning
import cugraph  # GPU Graph Analytics

# DataFrame en GPU (sintaxis idéntica a pandas)
df = cudf.read_csv('large_dataset.csv')
gpu_result = df.groupby('category').sales.mean()

# Machine Learning 100x más rápido
from cuml.cluster import KMeans
from cuml.ensemble import RandomForestClassifier

# Clustering en millones de puntos
kmeans_gpu = KMeans(n_clusters=50)
clusters = kmeans_gpu.fit_predict(df[['lat', 'lon', 'amount']])

# Random Forest en GPU
rf_gpu = RandomForestClassifier(n_estimators=100)
rf_gpu.fit(X_train, y_train)
predictions = rf_gpu.predict(X_test)

# Análisis de grafos masivos
G = cugraph.Graph()
G.from_cudf_edgelist(edges_df, source='src', destination='dst')
pagerank_scores = cugraph.pagerank(G)
```

#### 🎯 **Hardware Requirements**
- **Mínimo:** NVIDIA GTX 1060 (6GB VRAM)
- **Recomendado:** RTX 3080+ (12GB+ VRAM)
- **Empresarial:** A100/V100 (40GB+ VRAM)

#### 🔗 **Enlaces Esenciales**
- 🌐 **Web:** https://rapids.ai/
- 📖 **Docs:** https://docs.rapids.ai/
- 🚀 **Quickstart:** https://rapids.ai/start.html
- 💻 **GitHub:** https://github.com/rapidsai/

---

### 4. 🗺️ **GeoPandas - Análisis de Datos Geoespaciales**

#### ✨ **Características **
- **🐼 Extensión Natural:** Todas las funciones de pandas + geometrías
- **📐 Operaciones Espaciales:** Buffer, intersect, union, dissolve
- **🗺️ Coordenadas:** Manejo automático de sistemas de referencia
- **📊 Visualización:** Mapas integrados con matplotlib

#### 💻 **Código de ejemplo**
```python
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Leer datos geoespaciales
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
cities = gpd.read_file('world_cities.shp')

# Crear geometrías desde coordenadas
df = pd.DataFrame({
    'city': ['Madrid', 'Barcelona', 'Valencia'],
    'lat': [40.4168, 41.3851, 39.4699],
    'lon': [-3.7038, 2.1734, -0.3763]
})
gdf = gpd.GeoDataFrame(df, 
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs='EPSG:4326'
)

# Operaciones geoespaciales
# Buffer de 100km alrededor de cada ciudad
city_buffers = gdf.to_crs('EPSG:3857').buffer(100000)

# Encontrar países que intersectan con los buffers
spain = world[world.name == 'Spain']
intersections = gpd.overlay(city_buffers, spain, how='intersection')

# Análisis de proximidad
nearest_city = cities.sindex.nearest(gdf.geometry)

# Visualización avanzada
fig, ax = plt.subplots(figsize=(15, 10))
world.plot(ax=ax, color='lightgray', edgecolor='white')
gdf.plot(ax=ax, color='red', markersize=100, alpha=0.7)
city_buffers.plot(ax=ax, alpha=0.3, color='blue')
plt.title('Análisis Geoespacial con GeoPandas')
```

#### 📊 **Ejemplos empresariales**
- 🚚 **Logística:** Optimización de rutas y zonas de entrega
- 🏙️ **Smart Cities:** Planificación urbana basada en datos
- 🌍 **Medio Ambiente:** Monitoreo de deforestación y cambio climático

#### 🔗 **Enlaces Esenciales**
- 🌐 **Web:** https://geopandas.org/
- 📖 **Docs:** https://geopandas.org/en/stable/
- 🌍 **Ecosistema:** https://geopandas.org/en/latest/community/ecosystem.html
- 🎓 **Gallery:** https://geopandas.org/en/stable/gallery/index.html

---

### 5. 🕸️ **NetworkX - Análisis de Redes y Grafos**

#### ✨ **Características **
- **🧠 Algoritmos Avanzados:** 200+ algoritmos de grafos implementados
- **📊 Métricas Completas:** Centralidad, clustering, modularidad
- **🎨 Visualización:** Integración nativa con matplotlib
- **⚡ Escalabilidad:** Compatible con RAPIDS para GPU

#### 💻 **Código de ejemplo**
```python
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Crear red desde datos
edges = pd.DataFrame([
    ('Alice', 'Bob', 0.8),
    ('Bob', 'Charlie', 0.6),
    ('Charlie', 'Diana', 0.9),
    ('Diana', 'Alice', 0.7),
    ('Alice', 'Eve', 0.5)
], columns=['source', 'target', 'weight'])

G = nx.from_pandas_edgelist(edges, edge_attr='weight')

# Análisis de centralidad
centrality_metrics = {
    'betweenness': nx.betweenness_centrality(G),
    'closeness': nx.closeness_centrality(G), 
    'eigenvector': nx.eigenvector_centrality(G),
    'pagerank': nx.pagerank(G)
}

# Detección de comunidades
communities = list(nx.community.greedy_modularity_communities(G))
print(f"Encontradas {len(communities)} comunidades")

# Análisis de conectividad
print(f"Densidad de la red: {nx.density(G):.3f}")
print(f"Coeficiente de clustering: {nx.average_clustering(G):.3f}")

# Visualización avanzada
pos = nx.spring_layout(G, k=1, iterations=50)
plt.figure(figsize=(12, 8))

# Tamaño de nodos proporcional a centralidad
node_sizes = [centrality_metrics['betweenness'][node] * 3000 for node in G.nodes()]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                       node_color='lightblue', alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray')

plt.title("Red Social - Análisis de Centralidad")
plt.axis('off')
plt.show()
```

#### 📊 **Algoritmos Destacados**
- **🎯 Centralidad:** Betweenness, Closeness, Eigenvector, PageRank
- **👥 Comunidades:** Modularity, Louvain, Label Propagation
- **🛣️ Caminos:** Shortest Path, All Pairs, A* Search
- **🔍 Matching:** Maximum Bipartite, Minimum Weight
- **🌊 Flujo:** Max Flow, Min Cut, Cost Flow

#### 🎯 **Casos de Uso por Industria**
- **📱 Redes Sociales:** Influencers, comunidades, propagación viral
- **🏦 Finanzas:** Detección de fraude, análisis de riesgo sistémico
- **🧬 Bioinformática:** Redes de proteínas, pathways metabólicos
- **🚚 Supply Chain:** Optimización logística, vulnerabilidades

#### 🔗 **Enlaces Esenciales**
- 🌐 **Web:** https://networkx.org/
- 📖 **Docs:** https://networkx.org/documentation/stable/
- 🎓 **Tutorial:** https://networkx.org/documentation/stable/tutorial.html
- 🗺️ **Geo Examples:** https://networkx.org/documentation/stable/auto_examples/geospatial/

---

### 6. 🔢 **Xarray - Arrays Multidimensionales Etiquetados**

#### ✨ **Características **
- **🏷️ Dimensiones Nombradas:** Adiós a confusión con ejes
- **📅 Coordenadas Inteligentes:** Tiempo, latitud, longitud automáticos  
- **🚀 Dask Integration:** Datasets de terabytes sin problemas
- **💾 Formatos Científicos:** NetCDF, HDF5, Zarr nativos

#### 💻 **Código de ejemplo**
```python
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Crear DataArray con dimensiones etiquetadas
temperature = xr.DataArray(
    np.random.randn(365, 50, 100),  # 1 año, 50 latitudes, 100 longitudes
    dims=['time', 'lat', 'lon'],
    coords={
        'time': pd.date_range('2023-01-01', periods=365),
        'lat': np.linspace(-90, 90, 50),
        'lon': np.linspace(-180, 180, 100)
    },
    attrs={
        'units': 'degrees_celsius',
        'description': 'Daily temperature anomalies'
    }
)

# Operaciones intuitivas con etiquetas
summer = temperature.sel(time=temperature.time.dt.season == 'JJA')
tropics = temperature.sel(lat=slice(-23.5, 23.5))
madrid_temp = temperature.sel(lat=40.4, lon=-3.7, method='nearest')

# Agregaciones por dimensión
monthly_avg = temperature.groupby('time.month').mean()
annual_cycle = temperature.groupby('time.dayofyear').mean()

# Operaciones matemáticas preservan metadatos
anomaly = temperature - temperature.mean(dim='time')
normalized = (temperature - temperature.mean()) / temperature.std()

# Visualización integrada
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Mapa promedio anual
temperature.mean(dim='time').plot(ax=axes[0,0], cmap='RdBu_r')
axes[0,0].set_title('Temperatura Promedio Anual')

# Serie temporal en Madrid
madrid_temp.plot(ax=axes[0,1])
axes[0,1].set_title('Serie Temporal - Madrid')

# Ciclo anual promedio
annual_cycle.mean(dim=['lat', 'lon']).plot(ax=axes[1,0])
axes[1,0].set_title('Ciclo Anual Global')

# Diferencia estacional
(summer.mean() - temperature.sel(time=temperature.time.dt.season == 'DJF').mean()).plot(
    ax=axes[1,1], cmap='RdBu_r'
)
axes[1,1].set_title('Diferencia Verano-Invierno')

plt.tight_layout()
plt.show()
```

#### 📊 **Ventajas vs NumPy/Pandas**
| Operación | NumPy | Pandas | Xarray |
|-----------|--------|---------|---------|
| Selección | `arr[0, :, 5]` | `df.iloc[0, 5]` | `da.sel(time='2023-01', lon=-3.7)` |
| Agregación | `arr.mean(axis=0)` | `df.mean()` | `da.mean(dim='time')` |
| Metadata | ❌ | Limitado | ✅ Completo |
| Broadcasting | Manual | ❌ | Automático |

#### 🔬 **Casos de Uso Científicos**
- **🌡️ Climatología:** Modelos climáticos, reanálisis, proyecciones
- **🛰️ Teledetección:** Imágenes satelitales multiespectrales
- **🌊 Oceanografía:** Temperatura, salinidad, corrientes 4D
- **🧬 Bioimagen:** Microscopía 4D (x, y, z, tiempo)

#### 🔗 **Enlaces Esenciales**
- 🌐 **Web:** https://xarray.dev/
- 📖 **Docs:** https://docs.xarray.dev/
- 🎓 **Tutorial:** https://tutorial.xarray.dev/
- 🎨 **Gallery:** https://docs.xarray.dev/en/stable/gallery.html

---

### 7. 🦆 **DuckDB - Base de Datos Analítica Embebida**

#### ✨ **Características **
- **📊 OLAP Puro:** 100x más rápido que MySQL en agregaciones
- **📦 Zero Setup:** `pip install duckdb` y listo
- **📁 Consulta Directa:** Parquet, CSV, JSON sin importar
- **🔄 SQL Moderno:** Window functions, CTEs, arrays

#### 💻 **Código de ejemplo**
```python
import duckdb
import pandas as pd

# Crear conexión (en memoria)
con = duckdb.connect()

# Consultar archivos directamente (¡sin importar!)
result = con.execute("""
    SELECT 
        category,
        DATE_TRUNC('month', date) as month,
        SUM(amount) as monthly_sales,
        COUNT(*) as transactions,
        AVG(amount) as avg_transaction
    FROM 'sales_data.parquet'  -- Consulta directa
    WHERE date >= '2023-01-01'
    GROUP BY category, month
    ORDER BY monthly_sales DESC
""").fetchdf()

# Mezclar SQL con DataFrames
df_customers = pd.read_csv('customers.csv')
con.register('customers', df_customers)  # Registrar DataFrame

sales_analysis = con.execute("""
    WITH customer_stats AS (
        SELECT 
            customer_id,
            SUM(amount) as total_spent,
            COUNT(*) as visit_count,
            AVG(amount) as avg_purchase
        FROM 'sales_data.parquet'
        GROUP BY customer_id
    )
    SELECT 
        c.customer_name,
        c.segment,
        cs.total_spent,
        cs.visit_count,
        cs.avg_purchase,
        -- Ventana deslizante
        AVG(cs.total_spent) OVER (
            PARTITION BY c.segment 
            ORDER BY cs.total_spent 
            ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
        ) as segment_avg_nearby
    FROM customers c
    JOIN customer_stats cs ON c.customer_id = cs.customer_id
    WHERE cs.total_spent > 1000
    ORDER BY cs.total_spent DESC
""").fetchdf()

# Arrays y JSON (funciones modernas)
json_analysis = con.execute("""
    SELECT 
        json_extract(metadata, '$.campaign') as campaign,
        json_extract_string(metadata, '$.source') as source,
        array_length(string_split(tags, ',')) as tag_count
    FROM marketing_data
    WHERE json_valid(metadata)
""").fetchdf()

# Performance: comparar con pandas
import time

# DuckDB: Aggregación en 50M filas
start = time.time()
duck_result = con.execute("""
    SELECT category, AVG(amount) 
    FROM huge_sales_file.parquet 
    GROUP BY category
""").fetchdf()
duck_time = time.time() - start

print(f"DuckDB: {duck_time:.2f}s")
# Típicamente 5-20x más rápido que pandas equivalente
```

#### 📊 **Performance Benchmarks**
| Operación | Pandas | DuckDB | Speedup |
|-----------|---------|--------|---------|
| GROUP BY (10M filas) | 15.2s | 1.8s | 8.4x |
| JOIN (5M + 2M) | 12.7s | 2.1s | 6.0x |  
| Window Functions | 25.3s | 3.2s | 7.9x |
| Parquet Read | 8.1s | 0.9s | 9.0x |

#### 🎯 **Casos de Uso Ideales**
- ✅ **Prototipado Rápido:** Analytics locales sin setup
- ✅ **ETL Pesado:** Transformaciones complejas en CSV/Parquet
- ✅ **Reporting:** Dashboards con queries complejas
- ✅ **Data Apps:** Embedded analytics en aplicaciones

#### 🔗 **Enlaces Esenciales**
- 🌐 **Web:** https://duckdb.org/
- 📖 **Docs:** https://duckdb.org/docs/stable/
- 🚀 **Installation:** https://duckdb.org/docs/installation/
- 📚 **Guides:** https://duckdb.org/docs/stable/guides/

---

### 8. 🌉 **Ibis - Interfaz Unificada para Backends Múltiples**

#### ✨ **Características **
- **🔗 Backend Agnostic:** PostgreSQL, BigQuery, Spark, mismo código
- **⚡ Smart Execution:** DuckDB por defecto para máximo rendimiento
- **🧠 Lazy Evaluation:** Solo ejecuta cuando es necesario
- **📊 DataFrame API:** Sintaxis familiar para SQL subyacente

#### 💻 **Código de ejemplo**
```python
import ibis
from ibis import _

# Múltiples backends con la misma API
backends = {
    'local': ibis.duckdb.connect(),
    'postgres': ibis.postgres.connect('postgresql://user:pass@host/db'),
    'bigquery': ibis.bigquery.connect(project_id='my-project'),
    'snowflake': ibis.snowflake.connect(
        user='user', password='pass', account='account'
    )
}

# Función analítica reutilizable
def customer_segmentation(con, table_name):
    """Segmentación RFM que funciona en cualquier backend"""
    
    customers = con.table(table_name)
    
    # Calcular métricas RFM
    rfm = (
        customers
        .group_by('customer_id')
        .aggregate(
            # Recency: días desde última compra
            recency=_.purchase_date.max().delta(
                ibis.today(), 'day'
            ),
            # Frequency: número de compras
            frequency=_.customer_id.count(),
            # Monetary: total gastado
            monetary=_.amount.sum()
        )
    )
    
    # Crear percentiles para segmentación
    rfm_scored = (
        rfm
        .mutate(
            r_score=_.recency.percent_rank().round(0) + 1,
            f_score=_.frequency.percent_rank().round(0) + 1, 
            m_score=_.monetary.percent_rank().round(0) + 1
        )
        .mutate(
            rfm_score=(_.r_score * 100 + _.f_score * 10 + _.m_score)
        )
    )
    
    # Segmentos de negocio
    return (
        rfm_scored
        .mutate(
            segment=ibis.cases(
                (_.rfm_score >= 544, 'Champions'),
                (_.rfm_score >= 334, 'Loyal Customers'), 
                (_.rfm_score >= 313, 'Potential Loyalists'),
                (_.rfm_score >= 155, 'At Risk'),
                else_='Lost Customers'
            )
        )
        .group_by('segment')
        .aggregate(
            customers=_.customer_id.count(),
            avg_monetary=_.monetary.mean(),
            avg_frequency=_.frequency.mean()
        )
        .order_by(_.customers.desc())
    )

# Ejecutar en diferentes sistemas
for name, backend in backends.items():
    print(f"\n--- Análisis en {name.upper()} ---")
    try:
        result = customer_segmentation(backend, 'sales_data')
        print(result.execute())
    except Exception as e:
        print(f"Error: {e}")

# Migración transparente
def migrate_analysis(source_backend, target_backend, query):
    """Migra análisis entre sistemas sin cambio de código"""
    
    # Ejecutar en sistema origen
    result = query.execute()
    
    # Crear tabla en destino
    target_backend.create_table('migrated_analysis', result)
    
    return "Migración completada"

# Ejemplo: Local → BigQuery
local_analysis = customer_segmentation(backends['local'], 'customers')
migrate_analysis(backends['local'], backends['bigquery'], local_analysis)
```

#### 🎯 **Casos de Uso Estratégicos**
- **🔄 Multi-Cloud:** Código que funciona en AWS, GCP, Azure
- **📈 Escalado Gradual:** Empezar local, escalar a cloud sin reescribir  
- **🏢 Migración:** Cambiar de Postgres a BigQuery sin drama
- **🧪 Prototipado:** Desarrollar en DuckDB, producir en Snowflake

#### 📊 **Backends Soportados**
| Categoría | Backends |
|-----------|----------|
| **SQL Local** | DuckDB, SQLite, PostgreSQL |
| **Cloud** | BigQuery, Snowflake, Redshift |
| **Big Data** | Spark, Impala, Clickhouse |
| **Especializados** | Polars, Pandas, Dask |

#### 🔗 **Enlaces Esenciales**  
- 🌐 **Web:** https://ibis-project.org/
- 📖 **Backends:**
