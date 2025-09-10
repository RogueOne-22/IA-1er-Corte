import polars as pl
import pandas as pd
import plotly.graph_objects as go

# --- 1. Cargar datos ---
ruta = "BD_SENSORES_ETL.parquet" 
df_final = pl.read_parquet(ruta)

# Convertir a pandas para Plotly
df_pd = df_final.to_pandas()


fig = go.Figure()

# Lista de sensores (nombres de columnas numéricas 1–53)
sensores = [col for col in df_pd.columns if col.isdigit() and 1 <= int(col) <= 53]

# Agregar una traza por sensor, inicialmente ocultas
for i, sensor in enumerate(sensores):
    fig.add_trace(go.Scatter(
        y=df_pd[sensor],
        mode="lines",
        name=f"Sensor {sensor}",
        visible=(i == 0)  
    ))


botones = []
for i, sensor in enumerate(sensores):
    estado_visible = [False] * len(sensores)
    estado_visible[i] = True  # activar solo el sensor elegido

    botones.append(dict(
        label=f"Sensor {sensor}",
        method="update",
        args=[{"visible": estado_visible},
              {"title": f"📈 Valores normalizados - Sensor {sensor}"}]
    ))

fig.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=botones,
        x=1.05,  # posición derecha
        y=0.5,
        xanchor="left",
        yanchor="middle"
    )],
    title="📊 Valores normalizados de Sensores (1–53)",
    xaxis_title="Índice de muestra",
    yaxis_title="Valor normalizado (0–3)",
    legend=dict(x=1.05, y=1)
)

# --- 4. Mostrar dashboard ---
fig.show()
