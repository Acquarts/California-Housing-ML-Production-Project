# app/streamlit_app.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# =========================
# Paths y Config UI
# =========================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
DATA_PATH = BASE_DIR / "data" / "housing.csv"

st.set_page_config(page_title="California Housing – Predictor", page_icon="🏠", layout="centered")
st.title("🏠 California Housing — Predictor")
st.caption("Predicción individual y por lotes usando el modelo local (sin API).")

# =========================
# Utilidades
# =========================
def clip(v, lo, hi):
    return float(np.clip(v, lo, hi))

# =========================
# Carga de modelo (cacheado, ruta robusta)
# =========================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Añádelo al repo o ajusta la ruta."
        )
    return joblib.load(MODEL_PATH)

model = load_model()

# =========================
# Defaults realistas y límites seguros
# =========================
FALLBACK_BOUNDS = {
    "longitude": (-124.5, -114.0),      # California aprox
    "latitude":  (32.0, 42.0),
    "housing_median_age": (1, 52),
    "total_rooms": (1, 8000),
    "total_bedrooms": (1, 2000),
    "population": (1, 5000),
    "households": (1, 2000),
    "median_income": (0.5, 15.0),
}
FALLBACK_MEDIANS = {
    "longitude": -119.5,
    "latitude": 36.5,
    "housing_median_age": 29.0,
    "total_rooms": 2127.0,
    "total_bedrooms": 435.0,
    "population": 1166.0,
    "households": 409.0,
    "median_income": 3.53,
}
FALLBACK_OCEANS = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
FALLBACK_OCEAN_DEFAULT = "INLAND"

@st.cache_data
def load_defaults():
    """
    Si existe data/housing.csv, usamos min/max reales del dataset y medianas como defaults.
    Si no existe, usamos rangos y medianas realistas (fallbacks), nunca percentiles raros.
    """
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        num_cols = [
            "longitude","latitude","housing_median_age","total_rooms","total_bedrooms",
            "population","households","median_income"
        ]
        med = df[num_cols].median(numeric_only=True).to_dict()
        bounds = {
            c: (float(df[c].min()), float(df[c].max()))
            for c in num_cols
        }
        # Para enteros, dejamos luego el casting en los inputs
        oceans = sorted(df["ocean_proximity"].dropna().unique().tolist())
        default_ocean = df["ocean_proximity"].mode().iat[0]
        return med, bounds, default_ocean, oceans
    else:
        st.info("No se encontró data/housing.csv. Usando valores por defecto realistas.")
        return FALLBACK_MEDIANS, FALLBACK_BOUNDS, FALLBACK_OCEAN_DEFAULT, FALLBACK_OCEANS

med, bounds, default_ocean, oceans = load_defaults()

# =========================
# 1) Predicción individual (local)
# =========================
st.subheader("🔹 Predicción individual (modelo local)")

with st.form("form_single"):
    c1, c2 = st.columns(2)
    # Longitude / Latitude con clip para garantizar que value ∈ [min,max]
    lon_lo, lon_hi = bounds["longitude"]
    lat_lo, lat_hi = bounds["latitude"]

    longitude = c1.number_input(
        "Longitude",
        value=clip(med["longitude"], lon_lo, lon_hi),
        min_value=float(lon_lo), max_value=float(lon_hi),
        step=0.01,
        help="Longitud del distrito."
    )
    latitude = c2.number_input(
        "Latitude",
        value=clip(med["latitude"], lat_lo, lat_hi),
        min_value=float(lat_lo), max_value=float(lat_hi),
        step=0.01,
        help="Latitud del distrito."
    )

    c3, c4 = st.columns(2)
    age_lo, age_hi = bounds["housing_median_age"]
    income_lo, income_hi = bounds["median_income"]

    housing_median_age = c3.number_input(
        "Housing median age (años, distrito)",
        value=int(round(clip(med["housing_median_age"], age_lo, age_hi))),
        min_value=int(age_lo), max_value=int(age_hi),
        step=1,
        help="Edad mediana de las viviendas del distrito."
    )
    median_income = c4.number_input(
        "Median income (×10k USD, distrito)",
        value=clip(med["median_income"], income_lo, income_hi),
        min_value=float(income_lo), max_value=float(income_hi),
        step=0.01, format="%.4f",
        help="Ej: 8.3 ≈ 83.000 USD."
    )

    rooms_lo, rooms_hi = bounds["total_rooms"]
    beds_lo, beds_hi = bounds["total_bedrooms"]
    pop_lo, pop_hi = bounds["population"]
    hh_lo, hh_hi = bounds["households"]

    total_rooms = st.number_input(
        "Total rooms (distrito)",
        value=int(round(clip(med["total_rooms"], rooms_lo, rooms_hi))),
        min_value=int(rooms_lo), max_value=int(rooms_hi),
        step=10,
        help="Suma de habitaciones en el distrito (no por vivienda)."
    )
    total_bedrooms = st.number_input(
        "Total bedrooms (distrito)",
        value=int(round(clip(med["total_bedrooms"], beds_lo, beds_hi))),
        min_value=int(beds_lo), max_value=int(beds_hi),
        step=1,
        help="Suma de dormitorios en el distrito."
    )
    population = st.number_input(
        "Population (distrito)",
        value=int(round(clip(med["population"], pop_lo, pop_hi))),
        min_value=int(pop_lo), max_value=int(pop_hi),
        step=1
    )
    households = st.number_input(
        "Households (distrito)",
        value=int(round(clip(med["households"], hh_lo, hh_hi))),
        min_value=int(hh_lo), max_value=int(hh_hi),
        step=1
    )

    # Ocean proximity
    # Si el default no está en la lista (por fallback), usamos el primero.
    try:
        default_idx = oceans.index(default_ocean)
    except ValueError:
        default_idx = 0

    ocean_proximity = st.selectbox(
        "Ocean proximity",
        options=oceans,
        index=default_idx,
        help="Categoría del distrito respecto a la costa."
    )

    submit_single = st.form_submit_button("Predecir")

if submit_single:
    row = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }
    try:
        X_one = pd.DataFrame([row])
        yhat = float(model.predict(X_one)[0])
        st.success(f"Precio estimado: ${yhat:,.2f}")
        with st.expander("Datos usados"):
            st.json(row)
    except Exception as e:
        st.error(f"Error prediciendo con el modelo local: {e}")
        st.info("¿Existe artifacts/model.joblib y coincide con el pipeline de entrenamiento?")

st.markdown("---")

# =========================
# 2) Predicción por lotes (CSV) — usa el modelo local
# =========================
st.subheader("📦 Predicción por lotes (sube un CSV)")

uploaded = st.file_uploader(
    "Sube un CSV con estas columnas:",
    type=["csv"],
    help="""Columnas requeridas:
longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
population, households, median_income, ocean_proximity"""
)

required_cols = [
    "longitude","latitude","housing_median_age","total_rooms","total_bedrooms",
    "population","households","median_income","ocean_proximity"
]

if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        missing = [c for c in required_cols if c not in df_in.columns]
        if missing:
            st.error(f"Faltan columnas en tu CSV: {missing}")
        else:
            preds = model.predict(df_in[required_cols])
            df_out = df_in.copy()
            df_out["predicted_price"] = preds

            st.success(f"Predicciones generadas: {len(df_out)} filas")
            st.dataframe(df_out.head(20))

            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Descargar CSV con predicciones",
                data=csv_bytes,
                file_name="predicciones_california.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error procesando el CSV: {e}")
