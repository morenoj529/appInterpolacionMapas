from pathlib import Path

import geopandas as gpd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from shapely.geometry import Point
from shapely.ops import unary_union

DATA_FILE = Path("TEMP2025.xls")
SHAPE_FILE = Path("AGEM_LMM2026.shp")
REQUIRED_COLUMNS = {"ID Estación", "Longitud", "Latitud", "Temperatura"}
DEFAULT_BOUNDS = (-109.05, 25.72, -108.93, 25.85)


@st.cache_data
def cargar_datos(data_path=DATA_FILE):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {data_path}")

    df = pd.read_excel(data_path)
    df.columns = df.columns.str.strip()

    faltantes = REQUIRED_COLUMNS - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(sorted(faltantes))}")

    if "Precipitación" not in df.columns:
        df["Precipitación"] = 0.0

    columnas_numericas = ["Longitud", "Latitud", "Temperatura", "Precipitación"]
    for col in columnas_numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalidas = [col for col in columnas_numericas if df[col].isna().any()]
    if invalidas:
        raise ValueError(
            f"Hay valores faltantes o no numéricos en: {', '.join(invalidas)}"
        )

    if df["ID Estación"].isna().any() or (df["ID Estación"].astype(str).str.strip() == "").any():
        raise ValueError("La columna 'ID Estación' contiene valores faltantes.")

    if df.empty:
        raise ValueError("No hay estaciones disponibles en el archivo de datos.")

    return df


@st.cache_data
def cargar_shapefile(shape_path=SHAPE_FILE):
    shape_path = Path(shape_path)
    if not shape_path.exists():
        raise FileNotFoundError(f"No se encontró el shapefile: {shape_path}")

    gdf = gpd.read_file(shape_path)
    if gdf.empty:
        raise ValueError("El shapefile no contiene geometrías.")
    if gdf.crs is None:
        raise ValueError("El shapefile no tiene CRS definido.")

    if "CVE_LOC" in gdf.columns:
        gdf = gdf[gdf["CVE_LOC"] == "0001"]
    if "CVE_MUN" in gdf.columns:
        gdf = gdf[gdf["CVE_MUN"] == "001"]
    if gdf.empty:
        raise ValueError("No se encontraron geometrías tras aplicar los filtros CVE_LOC/CVE_MUN.")

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def obtener_extension(gdf_ageb, pad=0.008):
    if gdf_ageb is not None and not gdf_ageb.empty:
        minx, miny, maxx, maxy = gdf_ageb.total_bounds
        minx, maxx = minx - pad, maxx + pad
        miny, maxy = miny - pad, maxy + pad
        return (minx, miny, maxx, maxy), unary_union(gdf_ageb.geometry)
    return DEFAULT_BOUNDS, None


def idw_calc(x, y, z, gx, gy, p=2.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    gx = np.asarray(gx, dtype=float)
    gy = np.asarray(gy, dtype=float)

    if x.size == 0 or y.size == 0 or z.size == 0:
        raise ValueError("No hay estaciones disponibles para interpolar.")
    if not (x.size == y.size == z.size):
        raise ValueError("Las coordenadas y valores de estaciones no tienen la misma longitud.")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(z).all():
        raise ValueError("Las coordenadas/valores contienen NaN o infinitos.")
    if not np.isfinite(p) or p <= 0:
        raise ValueError("La potencia IDW debe ser un número positivo.")

    dist = np.sqrt((gx[..., np.newaxis] - x) ** 2 + (gy[..., np.newaxis] - y) ** 2)
    exactas = dist == 0.0
    dist_segura = np.where(exactas, np.nan, dist)

    with np.errstate(divide="ignore", invalid="ignore"):
        pesos = 1.0 / np.power(dist_segura, p)
        num = np.nansum(pesos * z, axis=-1)
        den = np.nansum(pesos, axis=-1)
        resultado = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    if exactas.any():
        idx_estacion = np.argmax(exactas, axis=-1)
        coincide = exactas.any(axis=-1)
        resultado = np.where(coincide, z[idx_estacion], resultado)

    return resultado


def crear_mascara(geometria, gx, gy):
    if geometria is None:
        return np.ones(gx.shape, dtype=bool)

    try:
        from shapely import contains_xy

        return contains_xy(geometria, gx, gy)
    except Exception:
        try:
            from shapely import vectorized

            return vectorized.contains(geometria, gx, gy)
        except Exception:
            puntos = np.column_stack((gx.ravel(), gy.ravel()))
            mask = np.fromiter(
                (geometria.contains(Point(px, py)) for px, py in puntos),
                dtype=bool,
                count=puntos.shape[0],
            )
            return mask.reshape(gx.shape)


def obtener_crs_local(gdf_ageb, lons, lats):
    if gdf_ageb is not None and not gdf_ageb.empty:
        try:
            return gdf_ageb.estimate_utm_crs()
        except Exception:
            pass

    lon_media = np.nanmean(np.asarray(lons, dtype=float))
    lat_media = np.nanmean(np.asarray(lats, dtype=float))
    if not np.isfinite(lon_media) or not np.isfinite(lat_media):
        return None

    zona = int(np.floor((lon_media + 180.0) / 6.0) + 1)
    zona = max(1, min(zona, 60))
    epsg = 32600 + zona if lat_media >= 0 else 32700 + zona
    return f"EPSG:{epsg}"


def proyectar_xy(lons, lats, crs_destino):
    puntos = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(np.asarray(lons), np.asarray(lats)),
        crs="EPSG:4326",
    ).to_crs(crs_destino)
    return puntos.geometry.x.to_numpy(), puntos.geometry.y.to_numpy()


def interpolar_idw(lons, lats, vals, gx, gy, potencia_idw, gdf_ageb=None):
    crs_local = obtener_crs_local(gdf_ageb, lons, lats)
    if crs_local is not None:
        try:
            x_est, y_est = proyectar_xy(lons, lats, crs_local)
            xg, yg = proyectar_xy(gx.ravel(), gy.ravel(), crs_local)
            raster = idw_calc(
                x_est,
                y_est,
                vals,
                xg.reshape(gx.shape),
                yg.reshape(gy.shape),
                p=potencia_idw,
            )
            return raster, True
        except Exception:
            pass

    return idw_calc(lons, lats, vals, gx, gy, p=potencia_idw), False


def obtener_estilo(variable_seleccionada):
    if variable_seleccionada == "Temperatura (°C)":
        colores = ["#228B22", "#7CFC00", "#FFFF00", "#FFA500", "#FF4500", "#B22222", "#800000", "#300000"]
        return colores, "Land Surface Temperature (LST)", "°C", "#E6E6FA"

    colores = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
    return colores, "Precipitación Acumulada", "mm", "#FFD700"


def deg_to_dms(val, is_lat=True):
    hemi = ("N" if val >= 0 else "S") if is_lat else ("E" if val >= 0 else "W")
    v = abs(val)
    d = int(v)
    m = int((v - d) * 60)
    s = int(round((v - d - m / 60) * 3600))
    if s == 60:
        m += 1
        s = 0
    if m == 60:
        d += 1
        m = 0
    return f"{d}°{m}'{s}\"{hemi}"


def crear_figura(
    raster_idw,
    minx,
    maxx,
    miny,
    maxy,
    gdf_ageb,
    lons,
    lats,
    ids_estaciones,
    vals,
    variable_seleccionada,
):
    colores, titulo_mapa, unidad, color_pts = obtener_estilo(variable_seleccionada)
    cmap_personalizado = LinearSegmentedColormap.from_list("CustomMap", colores, N=256)

    vmin_data = float(np.nanmin(vals))
    vmax_data = float(np.nanmax(vals))
    vmin_plot, vmax_plot = vmin_data, vmax_data
    if vmin_plot == vmax_plot:
        vmin_plot = max(0.0, vmin_plot - 1.0)
        vmax_plot += 1.0

    fig, ax = plt.subplots(figsize=(11, 10), dpi=200)
    im = ax.imshow(
        raster_idw,
        extent=[minx, maxx, miny, maxy],
        origin="lower",
        cmap=cmap_personalizado,
        vmin=vmin_plot,
        vmax=vmax_plot,
        alpha=0.95,
        interpolation="bilinear",
    )

    if gdf_ageb is not None and not gdf_ageb.empty:
        gdf_ageb.boundary.plot(ax=ax, color="black", linewidth=0.35, alpha=0.45)

    ax.scatter(lons, lats, color=color_pts, edgecolor="black", s=55, linewidth=1, zorder=6)
    for idx, txt in enumerate(ids_estaciones):
        ax.annotate(
            f"{txt}\n{vals[idx]:.1f} {unidad}",
            (lons[idx], lats[idx]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7,
            weight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="gray"),
            zorder=7,
        )

    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: deg_to_dms(val, False)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: deg_to_dms(val, True)))
    ax.tick_params(axis="both", which="major", labelsize=8, direction="out")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.grid(True, linestyle="-", linewidth=0.4, color="gray", alpha=0.6)
    ax.set_title(titulo_mapa, fontsize=14, fontweight="bold", pad=15)

    ax.annotate(
        "N",
        xy=(0.95, 0.94),
        xytext=(0.95, 0.88),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        arrowprops=dict(facecolor="black", edgecolor="black", width=2, headwidth=7),
    )

    cajetin = patches.Rectangle(
        (0.68, 0.03),
        0.30,
        0.16,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=8,
    )
    ax.add_patch(cajetin)
    ax.text(0.70, 0.16, titulo_mapa, transform=ax.transAxes, fontsize=7, fontweight="bold", zorder=9)
    ax.text(0.70, 0.135, f"Max:  {vmax_data:.1f} {unidad}", transform=ax.transAxes, fontsize=6.5, zorder=9)
    ax.text(0.70, 0.115, f"Mean: {np.nanmean(vals):.1f} {unidad}", transform=ax.transAxes, fontsize=6.5, zorder=9)
    ax.text(0.70, 0.095, f"Min:  {vmin_data:.1f} {unidad}", transform=ax.transAxes, fontsize=6.5, zorder=9)

    cax = fig.add_axes([0.89, 0.17, 0.015, 0.10])
    cbar = plt.colorbar(im, cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=6)
    return fig


def inicializar_estado(df_estaciones):
    temp_default = df_estaciones["Temperatura"].astype(float).tolist()
    prec_default = df_estaciones["Precipitación"].astype(float).tolist()

    if "valores_aplicados" not in st.session_state:
        st.session_state["valores_aplicados"] = {
            "Temperatura (°C)": temp_default,
            "Precipitación (mm)": prec_default,
        }

    for variable, default in {
        "Temperatura (°C)": temp_default,
        "Precipitación (mm)": prec_default,
    }.items():
        actuales = st.session_state["valores_aplicados"].get(variable)
        if actuales is None or len(actuales) != len(default):
            st.session_state["valores_aplicados"][variable] = default

    if "potencia_idw_aplicada" not in st.session_state:
        st.session_state["potencia_idw_aplicada"] = 2.0


def construir_sidebar(df_estaciones):
    st.sidebar.header("Configuración")
    variable_seleccionada = st.sidebar.radio(
        "Selecciona la Variable a Interpolar:",
        ["Temperatura (°C)", "Precipitación (mm)"],
    )
    inicializar_estado(df_estaciones)

    with st.sidebar.form(key="formulario_interpolacion"):
        st.subheader(f"Valores de {variable_seleccionada.split(' ')[0]}")
        valores_dinamicos = []
        valores_base = st.session_state["valores_aplicados"][variable_seleccionada]

        for idx, row in df_estaciones.iterrows():
            if variable_seleccionada == "Temperatura (°C)":
                val = st.number_input(
                    label=f"{row['ID Estación']}:",
                    min_value=-10.0,
                    max_value=70.0,
                    value=float(valores_base[idx]),
                    step=0.1,
                    key=f"valor_temp_{idx}",
                )
            else:
                val = st.number_input(
                    label=f"{row['ID Estación']}:",
                    min_value=0.0,
                    max_value=300.0,
                    value=float(valores_base[idx]),
                    step=1.0,
                    key=f"valor_prec_{idx}",
                )
            valores_dinamicos.append(val)

        st.markdown("---")
        potencia_form = st.slider(
            "Potencia IDW (Power)",
            1.0,
            5.0,
            float(st.session_state["potencia_idw_aplicada"]),
            0.5,
        )
        submit_button = st.form_submit_button(label="Aplicar")

    if submit_button:
        st.session_state["valores_aplicados"][variable_seleccionada] = valores_dinamicos
        st.session_state["potencia_idw_aplicada"] = potencia_form

    vals = np.asarray(st.session_state["valores_aplicados"][variable_seleccionada], dtype=float)
    potencia_idw = float(st.session_state["potencia_idw_aplicada"])
    return variable_seleccionada, vals, potencia_idw


def main():
    st.set_page_config(layout="wide", page_title="Monitor Ambiental - Los Mochis")
    st.title("Interpolación Espacial (IDW) - Los Mochis")

    try:
        df_estaciones = cargar_datos(DATA_FILE)
    except Exception as exc:
        st.error(f"Error al cargar '{DATA_FILE.name}': {exc}")
        st.stop()

    try:
        gdf_ageb = cargar_shapefile(SHAPE_FILE)
    except Exception as exc:
        st.error(f"Error al cargar '{SHAPE_FILE.name}': {exc}")
        st.stop()

    variable_seleccionada, vals, potencia_idw = construir_sidebar(df_estaciones)

    lons = df_estaciones["Longitud"].to_numpy(dtype=float)
    lats = df_estaciones["Latitud"].to_numpy(dtype=float)
    (minx, miny, maxx, maxy), limite_urbano = obtener_extension(gdf_ageb)

    grid_res = 350
    xi = np.linspace(minx, maxx, grid_res)
    yi = np.linspace(miny, maxy, grid_res)
    gx, gy = np.meshgrid(xi, yi)

    try:
        raster_idw, usa_metros = interpolar_idw(lons, lats, vals, gx, gy, potencia_idw, gdf_ageb=gdf_ageb)
    except Exception as exc:
        st.error(f"No se pudo interpolar los datos: {exc}")
        st.stop()

    if not usa_metros:
        st.info(
            "No fue posible estimar un CRS proyectado local; la interpolación IDW usa distancias en grados."
        )

    mask = crear_mascara(limite_urbano, gx, gy)
    raster_idw = np.where(mask, raster_idw, np.nan)

    fig = crear_figura(
        raster_idw=raster_idw,
        minx=minx,
        maxx=maxx,
        miny=miny,
        maxy=maxy,
        gdf_ageb=gdf_ageb,
        lons=lons,
        lats=lats,
        ids_estaciones=df_estaciones["ID Estación"].astype(str).tolist(),
        vals=vals,
        variable_seleccionada=variable_seleccionada,
    )
    st.pyplot(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()