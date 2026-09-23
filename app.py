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
    total_seconds = int(round(abs(float(val)) * 3600))
    d = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{d}°{m}'{s}\"{hemi}"


def calcular_ancho_metros(minx, maxx, miny, maxy, gdf_ageb=None):
    ancho_grados = float(maxx) - float(minx)
    if ancho_grados <= 0:
        return 0.0

    lat_media = (float(miny) + float(maxy)) / 2.0
    if gdf_ageb is not None and not gdf_ageb.empty:
        try:
            crs_local = gdf_ageb.estimate_utm_crs()
            if crs_local is not None:
                px, py = proyectar_xy([minx, maxx], [lat_media, lat_media], crs_local)
                ancho = abs(float(px[1] - px[0]))
                if np.isfinite(ancho) and ancho > 0:
                    return ancho
        except Exception:
            pass

    metros_por_grado_lon = 111320.0 * np.cos(np.radians(lat_media))
    ancho = abs(ancho_grados * metros_por_grado_lon)
    return float(ancho) if np.isfinite(ancho) and ancho > 0 else 0.0


def elegir_longitud_barra(ancho_m):
    if not np.isfinite(ancho_m) or ancho_m <= 0:
        return 0.0
    objetivo = ancho_m * 0.22
    pot = 10.0 ** np.floor(np.log10(objetivo))
    for factor in (1.0, 2.0, 5.0, 10.0):
        cand = factor * pot
        if cand >= objetivo:
            return float(cand)
    return float(10.0 * pot)


def _crear_rosa_vientos(ax, cx=0.915, cy=0.865, tam=0.045):
    vertices = {
        "N": np.array([[cx, cy + tam], [cx - tam * 0.26, cy], [cx + tam * 0.26, cy]]),
        "E": np.array([[cx + tam, cy], [cx, cy + tam * 0.26], [cx, cy - tam * 0.26]]),
        "S": np.array([[cx, cy - tam], [cx - tam * 0.26, cy], [cx + tam * 0.26, cy]]),
        "W": np.array([[cx - tam, cy], [cx, cy + tam * 0.26], [cx, cy - tam * 0.26]]),
    }
    colores = {"N": "black", "E": "white", "S": "black", "W": "white"}
    for rumbo, pts in vertices.items():
        ax.add_patch(
            patches.Polygon(
                pts,
                closed=True,
                facecolor=colores[rumbo],
                edgecolor="black",
                linewidth=1.0,
                transform=ax.transAxes,
                zorder=11,
            )
        )
    ax.add_patch(
        patches.Circle((cx, cy), tam * 0.08, transform=ax.transAxes, facecolor="black", edgecolor="black", zorder=12)
    )
    ax.text(cx, cy + tam * 1.22, "N", transform=ax.transAxes, ha="center", va="center", fontsize=8, fontweight="bold", zorder=12)
    ax.text(cx + tam * 1.22, cy, "E", transform=ax.transAxes, ha="center", va="center", fontsize=8, fontweight="bold", zorder=12)
    ax.text(cx, cy - tam * 1.22, "S", transform=ax.transAxes, ha="center", va="center", fontsize=8, fontweight="bold", zorder=12)
    ax.text(cx - tam * 1.22, cy, "W", transform=ax.transAxes, ha="center", va="center", fontsize=8, fontweight="bold", zorder=12)


def _configurar_marco(ax, minx, maxx, miny, maxy):
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")

    xticks = np.linspace(minx, maxx, 6)
    yticks = np.linspace(miny, maxy, 6)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: deg_to_dms(val, False)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: deg_to_dms(val, True)))

    ax.set_xticks(np.linspace(minx, maxx, 21), minor=True)
    ax.set_yticks(np.linspace(miny, maxy, 21), minor=True)
    ax.grid(True, which="major", linestyle="-", linewidth=0.45, color="black", alpha=0.35)
    ax.grid(True, which="minor", linestyle="-", linewidth=0.25, color="black", alpha=0.22)
    ax.tick_params(axis="both", which="major", labelsize=8, direction="out", pad=3)

    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks(xticks)
    top_ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: deg_to_dms(val, False)))
    top_ax.tick_params(axis="x", which="major", labelsize=8, direction="out", pad=3)

    right_ax = ax.secondary_yaxis("right")
    right_ax.set_yticks(yticks)
    right_ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: deg_to_dms(val, True)))
    right_ax.tick_params(axis="y", which="major", labelsize=8, direction="out", pad=3, labelrotation=90)
    for lbl in ax.get_yticklabels():
        lbl.set_rotation(90)


def _agregar_cajetin(ax, fig, titulo_mapa, unidad, vals, gdf_ageb, minx, maxx, miny, maxy):
    x0, y0, w, h = 0.695, 0.02, 0.29, 0.29
    ax.add_patch(
        patches.Rectangle((x0, y0), w, h, transform=ax.transAxes, facecolor="white", edgecolor="black", linewidth=1.0, zorder=10)
    )

    y_split_1 = y0 + h * 0.54
    y_split_2 = y0 + h * 0.33
    ax.plot([x0, x0 + w], [y_split_1, y_split_1], transform=ax.transAxes, color="black", linewidth=0.8, zorder=11)
    ax.plot([x0, x0 + w], [y_split_2, y_split_2], transform=ax.transAxes, color="black", linewidth=0.8, zorder=11)

    vmin_data = float(np.nanmin(vals))
    vmax_data = float(np.nanmax(vals))
    media = float(np.nanmean(vals))

    ax.text(x0 + 0.015, y0 + h * 0.48, titulo_mapa, transform=ax.transAxes, fontsize=7.2, fontweight="bold", zorder=12)
    ax.text(x0 + 0.015, y0 + h * 0.41, f"Unidad: {unidad}", transform=ax.transAxes, fontsize=6.7, zorder=12)
    ax.text(x0 + 0.015, y0 + h * 0.34, f"Max:  {vmax_data:.1f}", transform=ax.transAxes, fontsize=6.7, zorder=12)
    ax.text(x0 + 0.015, y0 + h * 0.28, f"Mean: {media:.1f}", transform=ax.transAxes, fontsize=6.7, zorder=12)
    ax.text(x0 + 0.015, y0 + h * 0.22, f"Min:  {vmin_data:.1f}", transform=ax.transAxes, fontsize=6.7, zorder=12)

    ax.text(x0 + 0.015, y_split_1 - 0.03, "Leyenda", transform=ax.transAxes, fontsize=7.0, fontweight="bold", zorder=12)
    if gdf_ageb is not None and not gdf_ageb.empty:
        y_leg = y_split_1 - 0.07
        ax.plot([x0 + 0.015, x0 + 0.075], [y_leg, y_leg], transform=ax.transAxes, color="black", linewidth=1.2, zorder=12)
        ax.text(x0 + 0.085, y_leg, "Límite urbano / AGEB", transform=ax.transAxes, va="center", fontsize=6.5, zorder=12)
    else:
        ax.text(x0 + 0.015, y_split_1 - 0.07, "Sin geometría urbana disponible", transform=ax.transAxes, fontsize=6.4, zorder=12)

    ax.text(x0 + 0.015, y_split_2 - 0.02, "SCALE", transform=ax.transAxes, fontsize=7.0, fontweight="bold", zorder=12)
    ancho_m = calcular_ancho_metros(minx, maxx, miny, maxy, gdf_ageb=gdf_ageb)
    barra_m = elegir_longitud_barra(ancho_m)
    frac = min(0.22, (barra_m / ancho_m) * 0.9) if ancho_m > 0 and barra_m > 0 else 0.15
    bar_x = x0 + 0.015
    bar_y = y0 + 0.03
    bar_w = w * 0.9 * frac
    bar_h = 0.012
    ax.add_patch(
        patches.Rectangle((bar_x, bar_y), bar_w, bar_h, transform=ax.transAxes, facecolor="black", edgecolor="black", linewidth=0.8, zorder=12)
    )
    ax.add_patch(
        patches.Rectangle(
            (bar_x + bar_w, bar_y),
            bar_w,
            bar_h,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=12,
        )
    )
    distancia_label = f"{barra_m / 1000:.1f} km" if barra_m >= 1000 else f"{barra_m:.0f} m"
    ax.text(bar_x, bar_y + 0.016, "0", transform=ax.transAxes, fontsize=6.2, zorder=12)
    ax.text(bar_x + bar_w, bar_y + 0.016, distancia_label, transform=ax.transAxes, fontsize=6.2, ha="center", zorder=12)
    ax.text(bar_x + 2 * bar_w, bar_y + 0.016, distancia_label, transform=ax.transAxes, fontsize=6.2, ha="center", zorder=12)

    escala_txt = "Escala aprox.: N/D"
    if barra_m > 0:
        ax_pos = ax.get_position()
        ancho_eje_m = fig.get_figwidth() * ax_pos.width * 0.0254
        if ancho_eje_m > 0:
            barra_fisica_m = ancho_eje_m * (2 * bar_w / ax_pos.width)
            if barra_fisica_m > 0:
                escala = barra_m * 2 / barra_fisica_m
                escala_txt = f"Escala aprox.: 1:{int(round(escala)):,}".replace(",", " ")
    ax.text(x0 + 0.015, y0 + 0.005, escala_txt, transform=ax.transAxes, fontsize=6.2, zorder=12)


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

    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.07, top=0.90)
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

    _configurar_marco(ax, minx, maxx, miny, maxy)
    ax.set_title(titulo_mapa, fontsize=20, fontweight="bold", pad=18)
    _crear_rosa_vientos(ax)
    _agregar_cajetin(ax, fig, titulo_mapa, unidad, vals, gdf_ageb, minx, maxx, miny, maxy)

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