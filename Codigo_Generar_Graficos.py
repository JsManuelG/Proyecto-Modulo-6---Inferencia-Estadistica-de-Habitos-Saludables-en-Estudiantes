# generar_graficos.py

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm

# =========================
# CONFIGURACIÓN GENERAL
# =========================
DATASET_PATH = "dataset_habitos_saludables.csv"
OUTPUT_DIR = "images"
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# FUNCIONES AUXILIARES
# =========================
def guardar_fig(nombre_archivo: str) -> None:
    ruta = os.path.join(OUTPUT_DIR, nombre_archivo)
    plt.tight_layout()
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado: {ruta}")


def intervalo_confianza_media(series: pd.Series, nivel: float = 0.95):
    datos = series.dropna().astype(float)
    n = len(datos)
    media = datos.mean()
    std = datos.std(ddof=1)
    z = norm.ppf(1 - (1 - nivel) / 2)
    error = z * (std / math.sqrt(n))
    return media, media - error, media + error


def cargar_o_simular_dataset(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Dataset cargado desde: {path}")
        return df

    print("No se encontró el dataset. Se generará uno simulado.")
    n = 150

    horas_sueno = np.clip(np.random.normal(6.8, 1.1, n), 3.5, 10)
    actividad_fisica = np.clip(np.random.normal(3.2, 1.8, n), 0, 10)
    calidad_alimentacion = np.clip(np.random.normal(6.5, 1.7, n), 1, 10)
    nivel_estres = np.clip(np.random.normal(6.0, 1.9, n), 1, 10)

    bienestar_general = (
        40
        + horas_sueno * 4
        + actividad_fisica * 2.5
        + calidad_alimentacion * 2.8
        - nivel_estres * 3.2
        + np.random.normal(0, 5, n)
    )
    bienestar_general = np.clip(bienestar_general, 0, 100)

    buen_sueno = np.where(horas_sueno >= 7, 1, 0)
    alimentacion_saludable = np.where(calidad_alimentacion >= 7, 1, 0)

    df = pd.DataFrame(
        {
            "horas_sueno": horas_sueno,
            "actividad_fisica_horas_semana": actividad_fisica,
            "calidad_alimentacion": calidad_alimentacion,
            "nivel_estres": nivel_estres,
            "bienestar_general": bienestar_general,
            "buen_sueno": buen_sueno,
            "alimentacion_saludable": alimentacion_saludable,
        }
    )

    df.to_csv(path, index=False)
    print(f"Dataset simulado guardado en: {path}")
    return df


def buscar_columna(df: pd.DataFrame, posibles_nombres: list[str]) -> str | None:
    columnas_lower = {c.lower(): c for c in df.columns}
    for nombre in posibles_nombres:
        if nombre.lower() in columnas_lower:
            return columnas_lower[nombre.lower()]
    return None


# =========================
# GRÁFICOS
# =========================
def grafico_histograma_sueno(df: pd.DataFrame) -> None:
    col = buscar_columna(df, ["horas_sueno", "sueno_horas", "sleep_hours"])
    if not col:
        print("No se encontró columna de horas de sueño.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(df[col].dropna(), bins=12, edgecolor="black")
    plt.title("Distribución de Horas de Sueño")
    plt.xlabel("Horas de sueño")
    plt.ylabel("Frecuencia")
    guardar_fig("distribucion_sueno.png")


def grafico_histograma_actividad(df: pd.DataFrame) -> None:
    col = buscar_columna(
        df,
        [
            "actividad_fisica_horas_semana",
            "actividad_fisica",
            "horas_actividad_fisica",
            "physical_activity_hours",
        ],
    )
    if not col:
        print("No se encontró columna de actividad física.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(df[col].dropna(), bins=12, edgecolor="black")
    plt.title("Distribución de Actividad Física")
    plt.xlabel("Horas por semana")
    plt.ylabel("Frecuencia")
    guardar_fig("distribucion_actividad_fisica.png")


def grafico_boxplot_variables(df: pd.DataFrame) -> None:
    posibles = [
        "horas_sueno",
        "actividad_fisica_horas_semana",
        "calidad_alimentacion",
        "nivel_estres",
        "bienestar_general",
    ]
    cols = [c for c in posibles if c in df.columns]

    if len(cols) < 2:
        print("No hay suficientes variables para boxplot.")
        return

    plt.figure(figsize=(11, 6))
    plt.boxplot([df[c].dropna() for c in cols], tick_labels=cols)
    plt.title("Boxplot de Variables Cuantitativas")
    plt.ylabel("Valores")
    plt.xticks(rotation=20)
    guardar_fig("boxplot_variables.png")


def grafico_dispersion_sueno_bienestar(df: pd.DataFrame) -> None:
    x_col = buscar_columna(df, ["horas_sueno", "sueno_horas", "sleep_hours"])
    y_col = buscar_columna(df, ["bienestar_general", "wellbeing", "score_bienestar"])

    if not x_col or not y_col:
        print("No se encontraron columnas para dispersión sueño-bienestar.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.7)
    plt.title("Relación entre Horas de Sueño y Bienestar General")
    plt.xlabel("Horas de sueño")
    plt.ylabel("Bienestar general")
    guardar_fig("dispersion_sueno_bienestar.png")


def grafico_dispersion_actividad_bienestar(df: pd.DataFrame) -> None:
    x_col = buscar_columna(
        df,
        [
            "actividad_fisica_horas_semana",
            "actividad_fisica",
            "physical_activity_hours",
        ],
    )
    y_col = buscar_columna(df, ["bienestar_general", "wellbeing", "score_bienestar"])

    if not x_col or not y_col:
        print("No se encontraron columnas para dispersión actividad-bienestar.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.7)
    plt.title("Relación entre Actividad Física y Bienestar General")
    plt.xlabel("Actividad física (horas/semana)")
    plt.ylabel("Bienestar general")
    guardar_fig("dispersion_actividad_bienestar.png")


def grafico_barras_habitos(df: pd.DataFrame) -> None:
    col_sueno = buscar_columna(df, ["buen_sueno"])
    col_alim = buscar_columna(df, ["alimentacion_saludable"])

    if not col_sueno or not col_alim:
        print("No se encontraron columnas binarias de hábitos.")
        return

    proporciones = {
        "Buen sueño": df[col_sueno].mean(),
        "Alimentación saludable": df[col_alim].mean(),
    }

    plt.figure(figsize=(8, 6))
    plt.bar(list(proporciones.keys()), list(proporciones.values()))
    plt.title("Proporción de Hábitos Saludables")
    plt.ylabel("Proporción")
    plt.ylim(0, 1)
    guardar_fig("barras_habitos_saludables.png")


def grafico_tlc(df: pd.DataFrame) -> None:
    col = buscar_columna(df, ["horas_sueno", "sueno_horas", "sleep_hours"])
    if not col:
        print("No se encontró columna para demostrar TLC.")
        return

    datos = df[col].dropna().to_numpy()

    tamanos = [5, 30, 50]
    repeticiones = 1000

    for n in tamanos:
        medias = [
            np.mean(np.random.choice(datos, size=n, replace=True))
            for _ in range(repeticiones)
        ]

        plt.figure(figsize=(10, 6))
        plt.hist(medias, bins=25, edgecolor="black")
        plt.title(f"Distribución Muestral de la Media (n={n})")
        plt.xlabel("Media muestral de horas de sueño")
        plt.ylabel("Frecuencia")
        guardar_fig(f"tlc_media_muestral_n_{n}.png")


def grafico_comparacion_poblacion_vs_muestral(df: pd.DataFrame) -> None:
    col = buscar_columna(df, ["horas_sueno", "sueno_horas", "sleep_hours"])
    if not col:
        print("No se encontró columna para comparación poblacional/muestral.")
        return

    datos = df[col].dropna().to_numpy()
    medias = [np.mean(np.random.choice(datos, size=30, replace=True)) for _ in range(1000)]

    plt.figure(figsize=(10, 6))
    plt.hist(datos, bins=15, alpha=0.7, label="Población")
    plt.hist(medias, bins=15, alpha=0.7, label="Medias muestrales")
    plt.title("Comparación: Distribución Poblacional vs Distribución Muestral")
    plt.xlabel("Horas de sueño")
    plt.ylabel("Frecuencia")
    plt.legend()
    guardar_fig("comparacion_poblacion_muestral.png")


def grafico_intervalos_confianza(df: pd.DataFrame) -> None:
    variables = [
        ("horas_sueno", "Horas de sueño"),
        ("actividad_fisica_horas_semana", "Actividad física"),
    ]

    niveles = [0.90, 0.95, 0.99]

    etiquetas = []
    medias = []
    inferiores = []
    superiores = []

    for col, nombre in variables:
        if col not in df.columns:
            continue
        for nivel in niveles:
            media, li, ls = intervalo_confianza_media(df[col], nivel)
            etiquetas.append(f"{nombre}\n{int(nivel * 100)}%")
            medias.append(media)
            inferiores.append(media - li)
            superiores.append(ls - media)

    if not etiquetas:
        print("No se pudieron calcular intervalos de confianza.")
        return

    x = np.arange(len(etiquetas))

    plt.figure(figsize=(11, 6))
    plt.errorbar(
        x,
        medias,
        yerr=[inferiores, superiores],
        fmt="o",
        capsize=6,
    )
    plt.xticks(x, etiquetas, rotation=20)
    plt.title("Intervalos de Confianza para la Media")
    plt.ylabel("Valor medio estimado")
    guardar_fig("intervalos_confianza.png")


def grafico_ancho_intervalo_vs_n(df: pd.DataFrame) -> None:
    col = buscar_columna(df, ["horas_sueno", "sueno_horas", "sleep_hours"])
    if not col:
        print("No se encontró columna para impacto del tamaño muestral.")
        return

    datos = df[col].dropna().astype(float)
    std = datos.std(ddof=1)
    nivel = 0.95
    z = norm.ppf(1 - (1 - nivel) / 2)

    tamanos = np.array([10, 20, 30, 50, 80, 100, 150])
    anchos = 2 * z * (std / np.sqrt(tamanos))

    plt.figure(figsize=(10, 6))
    plt.plot(tamanos, anchos, marker="o")
    plt.title("Impacto del Tamaño Muestral en el Ancho del IC")
    plt.xlabel("Tamaño muestral")
    plt.ylabel("Ancho del intervalo")
    guardar_fig("ancho_intervalo_vs_n.png")


# =========================
# REPORTE TABULAR
# =========================
def guardar_tabla_resumen(df: pd.DataFrame) -> None:
    tabla = df.describe(include="all").transpose()
    ruta = os.path.join(OUTPUT_DIR, "tabla_resumen_estadistico.csv")
    tabla.to_csv(ruta)
    print(f"Tabla guardada: {ruta}")


# =========================
# MAIN
# =========================
def main():
    df = cargar_o_simular_dataset(DATASET_PATH)

    print("\nColumnas detectadas:")
    print(list(df.columns))

    guardar_tabla_resumen(df)

    grafico_histograma_sueno(df)
    grafico_histograma_actividad(df)
    grafico_boxplot_variables(df)
    grafico_dispersion_sueno_bienestar(df)
    grafico_dispersion_actividad_bienestar(df)
    grafico_barras_habitos(df)
    grafico_tlc(df)
    grafico_comparacion_poblacion_vs_muestral(df)
    grafico_intervalos_confianza(df)
    grafico_ancho_intervalo_vs_n(df)

    print("\nProceso finalizado.")


if __name__ == "__main__":
    main()