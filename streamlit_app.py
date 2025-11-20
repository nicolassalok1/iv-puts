#!/usr/bin/env python3
"""Streamlit app to download put option data and plot a 3D implied volatility surface."""

from __future__ import annotations

import datetime as dt
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Configure matplotlib to not use LaTeX if it's available
# This prevents "pdflatex introuvable" errors when matplotlib is installed
try:
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False
except ImportError:
    # matplotlib not installed, no configuration needed
    pass

MAX_LOOKAHEAD_YEARS = 3
MIN_MATURITY = 0.1

st.set_page_config(page_title="Put IV Surface", layout="wide")
st.title("3D Implied Volatility Surface (Puts)")
st.write(
    "Fetch put option data via `yfinance`, filter strikes around the current spot, "
    "and display an interpolated implied volatility surface."
)


def fetch_spot(ticker: yf.Ticker) -> float:
    history = ticker.history(period="1d")
    if history.empty:
        raise RuntimeError("Unable to retrieve spot price.")
    return float(history["Close"].iloc[-1])


def _select_monthly_expirations(
    expirations: List[str], years_ahead: float
) -> List[Tuple[dt.datetime, str]]:
    today = dt.datetime.utcnow().date()
    limit_date = today + dt.timedelta(days=365 * years_ahead)
    monthly: Dict[Tuple[int, int], Tuple[dt.date, str]] = {}
    for exp in expirations:
        exp_date = dt.datetime.strptime(exp, "%Y-%m-%d").date()
        if not (today < exp_date <= limit_date):
            continue
        key = (exp_date.year, exp_date.month)
        if key not in monthly or exp_date < monthly[key][0]:
            monthly[key] = (exp_date, exp)
    selected = sorted(monthly.values(), key=lambda item: item[0])
    return [(dt.datetime.combine(item[0], dt.time()), item[1]) for item in selected]


@st.cache_data(show_spinner=False, ttl=3600)
def download_option_data(symbol: str, years_ahead: float) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    spot = fetch_spot(ticker)
    expirations = ticker.options
    if not expirations:
        raise RuntimeError(f"No option expirations found for {symbol}")
    selected = _select_monthly_expirations(expirations, years_ahead)
    if not selected:
        raise RuntimeError("No expirations found within the requested horizon.")

    rows: List[dict] = []
    now = dt.datetime.utcnow()
    for expiry_dt, expiry_str in selected:
        T = max((expiry_dt - now).total_seconds() / (365.0 * 24 * 3600), 0.0)
        puts = ticker.option_chain(expiry_str).puts
        for _, row in puts.iterrows():
            rows.append(
                {
                    "S0": spot,
                    "K": float(row["strike"]),
                    "T": T,
                    "P_mkt": float(row["lastPrice"]),
                    "iv": float(row["impliedVolatility"]),
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def prepare_surface(df_raw: pd.DataFrame, strike_width: float) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    required_cols = {"S0", "K", "T", "iv"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"Data missing required columns: {missing}")
    spot = float(df_raw["S0"].median())
    lower_bound = math.ceil((spot - strike_width) / 10.0) * 10.0
    upper_bound = math.ceil((spot + strike_width) / 10.0) * 10.0
    mask = (df_raw["K"] >= lower_bound) & (df_raw["K"] <= upper_bound)
    df = df_raw.loc[mask].copy()
    df = df[df["T"] >= MIN_MATURITY]
    if df.empty:
        raise ValueError("No strikes/maturities within the specified constraints.")
    df = df.sort_values(["T", "K"]).reset_index(drop=True)

    k_values = np.sort(df["K"].unique())
    t_values = np.sort(df["T"].unique())
    surface = df.pivot_table(index="T", columns="K", values="iv", aggfunc="mean")
    surface = surface.reindex(index=t_values, columns=k_values)
    surface = surface.interpolate(axis=1, limit_direction="both").interpolate(
        axis=1, limit_direction="both"
    )
    surface = surface.loc[surface.index >= MIN_MATURITY]
    if surface.empty:
        raise ValueError("Not enough maturities to build a surface.")
    return df, surface, spot


with st.sidebar:
    ticker_input = st.text_input("Ticker", value="SPY").strip().upper()
    st.caption(f"Put expirations pulled up to {MAX_LOOKAHEAD_YEARS} years ahead.")
    strike_width = st.slider("Strike window around S₀", min_value=50, max_value=200, value=100, step=10)
    run_button = st.button("Fetch & Plot")


def plot_surface(surface: pd.DataFrame, spot: float) -> go.Figure:
    k_values = surface.columns.to_numpy(dtype=float)
    t_values = surface.index.to_numpy(dtype=float)
    t_min = float(t_values.min())
    t_max = float(t_values.max())
    KK, TT = np.meshgrid(k_values, t_values)
    plot_data = surface.to_numpy(dtype=float)
    if np.isnan(plot_data).any():
        plot_data = np.where(np.isnan(plot_data), np.nanmean(plot_data), plot_data)
    z_mean = float(np.nanmean(plot_data))
    z_std = float(np.nanstd(plot_data))
    z_min = z_mean - 2.0 * z_std
    z_max = z_mean + 2.0 * z_std

    fig = go.Figure(
        data=[
            go.Surface(
                x=KK,
                y=TT,
                z=plot_data,
                colorscale="Viridis",
                colorbar=dict(title="IV"),
                showscale=True,
            )
        ]
    )
    fig.update_layout(
        title="Interpolated Put Implied Volatility Surface (S₀ ± window)",
        scene=dict(
            xaxis=dict(title=dict(text="Strike K — spot price ≈ {:.2f}".format(spot))),
            yaxis=dict(title="Time to Maturity T (years)", range=[t_min, t_max]),
            zaxis=dict(title="Implied Volatility", range=[z_min, z_max]),
        ),
        width=900,
        height=650,
    )
    return fig


if run_button:
    if not ticker_input:
        st.warning("Enter a ticker to proceed.")
    else:
        try:
            df_raw = download_option_data(ticker_input, MAX_LOOKAHEAD_YEARS)
            st.success(f"Fetched {len(df_raw)} put rows for {ticker_input}.")
            try:
                df_filtered, surface, spot = prepare_surface(df_raw, strike_width)
            except ValueError as err:
                st.error(str(err))
            else:
                st.write(
                    f"Spot ≈ {spot:.2f}. Keeping strikes in "
                    f"[{df_filtered['K'].min():.2f}, {df_filtered['K'].max():.2f}] "
                    f"with maturities ≥ {MIN_MATURITY:.2f} years."
                )
                st.dataframe(
                    df_filtered[["S0", "K", "T", "P_mkt", "iv"]],
                    height=300,
                    use_container_width=True,
                )
                fig = plot_surface(surface, spot)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to download or process data: {exc}")
else:
    st.info("Configure parameters in the sidebar and click **Fetch & Plot** to build the surface.")
