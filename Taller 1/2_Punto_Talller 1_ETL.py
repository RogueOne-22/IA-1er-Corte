import polars as pl

# --- 1. Cargar archivo Excel ---
ruta_excel = "BD_SENSORES.xlsx"

# Leer todas las pestañas (sheets)
sheets = pl.read_excel(ruta_excel, sheet_id=None)  # dict {nombre: DataFrame}
print(f"✅ Se cargaron {len(sheets)} pestañas del archivo Excel")

# --- 2. Unificar todas las pestañas ---
df_list = []

for nombre, df in sheets.items():
    print(f"📂 Procesando pestaña: {nombre}")

    # Normalizar nombres de columnas
    df = df.rename({col: col.strip().lower().replace(" ", "_") for col in df.columns})

    # Seleccionar SOLO columnas con nombre numérico entre 1 y 53
    cols_relevantes = [
        col for col in df.columns if col.isdigit() and 1 <= int(col) <= 53
    ]

    # Si no hay columnas relevantes, saltar hoja
    if not cols_relevantes:
        continue

    # Forzar a numérico (float), ignorando strings no convertibles
    df = df.select([
        df[col].cast(pl.Float64, strict=False).alias(col) for col in cols_relevantes
    ])

    # --- Normalización de datos al rango [0.0, 3.0] ---
    for col in cols_relevantes:
        min_val = df[col].min()
        max_val = df[col].max()

        if min_val == max_val or min_val is None or max_val is None:
            df = df.with_columns(pl.lit(0.0).alias(col))
        else:
            df = df.with_columns(
                (((pl.col(col) - min_val) / (max_val - min_val)) * 3.0).alias(col)
            )

    # Agregar columna de trazabilidad
    df = df.with_columns(pl.lit(nombre).alias("origen_pestaña"))

    # Eliminar filas nulas
    df = df.drop_nulls()

    df_list.append(df)

# Concatenar resultados
if df_list:
    df_final = pl.concat(df_list, how="vertical")

    # --- 3. Guardar resultado único ---
    df_final.write_csv("BD_SENSORES_ETL.csv")
    df_final.write_parquet("BD_SENSORES_ETL.parquet")

    print("💾 Archivo consolidado generado: BD_SENSORES_ETL.csv y BD_SENSORES_ETL.parquet")

    # --- 4. Resumen ---
    print("\n📊 Resumen final:")
    print(df_final.describe())
else:
    print("⚠️ No se encontraron columnas relevantes en ninguna pestaña.")



