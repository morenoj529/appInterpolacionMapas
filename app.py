from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import contextily as cx
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
REQUIRED_COLUMNS = {"ID Estación", "Longitud", "Latitud", "Temperatura", "Precipitación"}
DEFAULT_BOUNDS = (-109.05, 25.72, -108.93, 25.85)
DEFAULT_GRID_RESOLUTION = 350
DEFAULT_IDW_POWER = 2.0

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def obtener_firma_archivo(path: Path) -> tuple[str, int, int]:
    """Devuelve una firma sencilla para invalidar cachés cuando cambia un archivo."""
    path = Path(path)
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


def obtener_firma_shapefile(shape_path: Path) -> tuple[tuple[str, int, int], ...]:
    """Incluye los archivos auxiliares que pueden afectar la lectura del shapefile."""
    shape_path = Path(shape_path)
    extensiones = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix")
    existentes = [shape_path.with_suffix(ext) for ext in extensiones]
    return tuple(obtener_firma_archivo(path) for path in existentes if path.exists())


@st.cache_data
def cargar_datos(data_path: Path, file_signature: tuple[str, int, int]) -> pd.DataFrame:
    del file_signature  # Se usa para invalidar la caché cuando cambia el archivo.
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {data_path}")

    df = pd.read_excel(data_path)
    df.columns = df.columns.astype(str).str.strip()
    df = df.reset_index(drop=True)

    faltantes = REQUIRED_COLUMNS - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(sorted(faltantes))}")

    columnas_numericas = ["Longitud", "Latitud", "Temperatura", "Precipitación"]
    for col in columnas_numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalidas = [col for col in columnas_numericas if df[col].isna().any()]
    if invalidas:
        raise ValueError(
            f"Hay valores faltantes o no numéricos en: {', '.join(invalidas)}"
        )

    if not np.isfinite(df[columnas_numericas].to_numpy(dtype=float)).all():
        raise ValueError("Las columnas numéricas contienen valores no finitos.")
    if not df["Longitud"].between(-180, 180).all():
        raise ValueError("Hay longitudes fuera del rango [-180, 180].")
    if not df["Latitud"].between(-90, 90).all():
        raise ValueError("Hay latitudes fuera del rango [-90, 90].")

    if df["ID Estación"].isna().any() or (
        df["ID Estación"].astype(str).str.strip() == ""
    ).any():
        raise ValueError("La columna 'ID Estación' contiene valores faltantes.")

    if df.empty:
        raise ValueError("No hay estaciones disponibles en el archivo de datos.")

    return df


@st.cache_data
def cargar_shapefile(
    shape_path: Path,
    file_signature: tuple[tuple[str, int, int], ...],
) -> gpd.GeoDataFrame:
    del file_signature  # Se usa para invalidar la caché cuando cambia cualquier sidecar.
    shape_path = Path(shape_path)
    if not shape_path.exists():
        raise FileNotFoundError(f"No se encontró el shapefile: {shape_path}")

    gdf = gpd.read_file(shape_path)
    if gdf.empty:
        raise ValueError("El shapefile no contiene geometrías.")
    if gdf.crs is None:
        raise ValueError("El shapefile no tiene CRS definido.")

    if "CVE_LOC" in gdf.columns:
        gdf = gdf[gdf["CVE_LOC"].astype(str) == "0001"]
    if "CVE_MUN" in gdf.columns:
        gdf = gdf[gdf["CVE_MUN"].astype(str) == "001"]
    if gdf.empty:
        raise ValueError("No se encontraron geometrías tras aplicar los filtros CVE_LOC/CVE_MUN.")

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def obtener_extension(
    gdf_ageb: Optional[gpd.GeoDataFrame], pad: float = 0.008
) -> tuple[tuple[float, float, float, float], Optional[object]]:
    if gdf_ageb is not None and not gdf_ageb.empty:
        minx, miny, maxx, maxy = gdf_ageb.total_bounds
        minx, maxx = minx - pad, maxx + pad
        miny, maxy = miny - pad, maxy + pad
        union = getattr(gdf_ageb.geometry, "union_all", None)
        geometria = union() if union is not None else unary_union(gdf_ageb.geometry)
        return (minx, miny, maxx, maxy), geometria
    return DEFAULT_BOUNDS, None


def idw_calc(x, y, z, gx, gy, p: float = DEFAULT_IDW_POWER) -> np.ndarray:
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


def crear_mascara(geometria, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    if geometria is None:
        return np.ones(gx.shape, dtype=bool)

    try:
        from shapely import covers_xy

        return covers_xy(geometria, gx, gy)
    except (ImportError, AttributeError):
        pass

    puntos = np.column_stack((gx.ravel(), gy.ravel()))
    mask = np.fromiter(
        (geometria.covers(Point(px, py)) for px, py in puntos),
        dtype=bool,
        count=puntos.shape[0],
    )
    return mask.reshape(gx.shape)


def obtener_crs_local(gdf_ageb, lons, lats):
    if gdf_ageb is not None and not gdf_ageb.empty:
        try:
            return gdf_ageb.estimate_utm_crs()
        except (ValueError, RuntimeError) as exc:
            LOGGER.warning("No se pudo estimar el UTM del área: %s", exc)

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


def agrupar_estaciones_duplicadas(lons, lats, vals):
    """Promedia valores de estaciones con coordenadas idénticas de forma determinista."""
    tabla = pd.DataFrame({"lon": lons, "lat": lats, "valor": vals})
    agrupada = tabla.groupby(["lon", "lat"], sort=True, as_index=False)["valor"].mean()
    return (
        agrupada["lon"].to_numpy(float),
        agrupada["lat"].to_numpy(float),
        agrupada["valor"].to_numpy(float),
    )


def interpolar_idw(lons, lats, vals, gx, gy, potencia_idw, gdf_ageb=None):
    lons, lats, vals = agrupar_estaciones_duplicadas(lons, lats, vals)
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
        except (ValueError, TypeError, RuntimeError) as exc:
            LOGGER.exception("Falló la interpolación en CRS proyectado %s", crs_local)
            st.warning(
                "No se pudo usar el CRS proyectado local; se usará una aproximación en grados."
            )
            LOGGER.warning("Motivo de la aproximación geográfica: %s", exc)

    if crs_local is None:
        LOGGER.warning("No fue posible determinar un CRS proyectado local.")

    return idw_calc(lons, lats, vals, gx, gy, p=potencia_idw), False


@st.cache_data
def crear_grid(minx: float, miny: float, maxx: float, maxy: float, grid_res: int):
    if grid_res < 2:
        raise ValueError("La resolución del grid debe ser al menos 2.")
    xi = np.linspace(minx, maxx, grid_res)
    yi = np.linspace(miny, maxy, grid_res)
    return np.meshgrid(xi, yi)


def obtener_estilo(variable_seleccionada):
    if variable_seleccionada == "Temperatura (°C)":
        colores = ["#228B22", "#7CFC00", "#FFFF00", "#FFA500", "#FF4500", "#B22222", "#800000", "#300000"]
        return colores, "Temperatura del aire", "°C", "#E6E6FA"

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
    return f'{d}°{m}\'{s}"{hemi}'


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
    area_km2: Optional[float] = None,
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
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: deg_to_dms(val, False)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: deg_to_dms(val, True)))
    ax.tick_params(axis="both", which="major", labelsize=8, direction="out")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    try:
        cx.add_basemap(
            ax,
            crs="EPSG:4326",
            source=cx.providers.Esri.WorldTopoMap,
            zorder=0,
            attribution=False,
        )
    except Exception as exc:
        LOGGER.warning("No se pudo cargar el mapa base: %s", exc)
        st.warning("No se pudo cargar el mapa base; se mostrará el mapa sin teselas externas.")

    if gdf_ageb is not None and not gdf_ageb.empty:
        gdf_ageb.plot(ax=ax, facecolor="white", edgecolor="none", zorder=1)

    ax.imshow(
        raster_idw,
        extent=[minx, maxx, miny, maxy],
        origin="lower",
        cmap=cmap_personalizado,
        vmin=vmin_plot,
        vmax=vmax_plot,
        alpha=0.95,
        interpolation="bilinear",
        zorder=2,
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

    box_x, box_y = 0.60, 0.02
    box_w, box_h = 0.38, 0.175
    mid_y = box_y + 0.072
    mid_x = box_x + 0.20
    cajetin = patches.Rectangle(
        (box_x, box_y), box_w, box_h, transform=ax.transAxes,
        facecolor="#fffff5", edgecolor="black", linewidth=0.8, zorder=8,
    )
    ax.add_patch(cajetin)
    ax.plot([box_x, box_x + box_w], [mid_y, mid_y], color="black", linewidth=0.7, transform=ax.transAxes, zorder=9)
    ax.plot([mid_x, mid_x], [mid_y, box_y + box_h], color="black", linewidth=0.7, transform=ax.transAxes, zorder=9)

    ax.text(box_x + 0.008, box_y + 0.158, titulo_mapa, transform=ax.transAxes, fontsize=5.6, fontweight="bold", zorder=9)
    ax.text(box_x + 0.008, box_y + 0.138, variable_seleccionada, transform=ax.transAxes, fontsize=5.5, fontweight="bold", zorder=9)

    cb_x, cb_y = box_x + 0.010, box_y + 0.082
    cb_w, cb_h = 0.018, 0.048
    n_steps = 64
    step_h = cb_h / n_steps
    for i in range(n_steps):
        ax.add_patch(patches.Rectangle(
            (cb_x, cb_y + i * step_h), cb_w, step_h * 1.05,
            transform=ax.transAxes, facecolor=cmap_personalizado(i / (n_steps - 1)),
            edgecolor="none", zorder=10,
        ))
    ax.add_patch(patches.Rectangle(
        (cb_x, cb_y), cb_w, cb_h, transform=ax.transAxes,
        facecolor="none", edgecolor="black", linewidth=0.6, zorder=11,
    ))
    ax.text(box_x + 0.034, box_y + 0.120, f"- Max:  {vmax_data:.2f}", transform=ax.transAxes, fontsize=5.6, zorder=9)
    ax.text(box_x + 0.034, box_y + 0.102, f"- Mean: {np.nanmean(vals):.2f}", transform=ax.transAxes, fontsize=5.6, zorder=9)
    ax.text(box_x + 0.034, box_y + 0.084, f"- Min:  {vmin_data:.2f}", transform=ax.transAxes, fontsize=5.6, zorder=9)

    ax.text(mid_x + 0.008, box_y + 0.158, "AGEB_LMM", transform=ax.transAxes, fontsize=5.8, fontweight="bold", zorder=9)
    ax.add_patch(patches.Rectangle(
        (mid_x + 0.008, box_y + 0.112), 0.022, 0.018,
        transform=ax.transAxes, facecolor="white", edgecolor="black", linewidth=0.7, zorder=9,
    ))
    ax.text(mid_x + 0.035, box_y + 0.123, "Área urbana de Los Mochis", transform=ax.transAxes, fontsize=5.0, zorder=9)
    area_texto = f"({area_km2:.2f} km²)" if area_km2 is not None else "(área no disponible)"
    ax.text(mid_x + 0.035, box_y + 0.107, area_texto, transform=ax.transAxes, fontsize=5.0, zorder=9)

    ax.text(box_x + 0.02, box_y + 0.048, "ESCALA GRÁFICA", transform=ax.transAxes, fontsize=7.0, fontweight="bold", zorder=9)
    base_x = box_x + 0.02
    base_sy = box_y + 0.024
    alto = 0.009
    anchos = [0.028, 0.028, 0.056, 0.056, 0.056]
    colores_barra = ["black", "white", "black", "white", "black"]
    x_actual = base_x
    for ancho, color in zip(anchos, colores_barra):
        ax.add_patch(patches.Rectangle(
            (x_actual, base_sy), ancho, alto, transform=ax.transAxes,
            facecolor=color, edgecolor="black", linewidth=0.7, zorder=9,
        ))
        x_actual += ancho
    ax.text(x_actual + 0.008, base_sy + 0.001, "km aprox.", transform=ax.transAxes, fontsize=5.6, zorder=9)

    etiquetas_x = [base_x, base_x + 0.028, base_x + 0.056, base_x + 0.112, base_x + 0.168, base_x + 0.224]
    etiquetas_val = ["0", "0.5", "1", "2", "3", "4"]
    for ex, ev in zip(etiquetas_x, etiquetas_val):
        ax.text(ex, base_sy - 0.013, ev, transform=ax.transAxes, fontsize=5.2, ha="center", zorder=9)

    fig.tight_layout()
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
        st.session_state["potencia_idw_aplicada"] = DEFAULT_IDW_POWER


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
                    label=f"{row['ID Estación']}:", min_value=-10.0, max_value=70.0,
                    value=float(valores_base[idx]), step=0.1, key=f"valor_temp_{idx}",
                )
            else:
                val = st.number_input(
                    label=f"{row['ID Estación']}:", min_value=0.0, max_value=300.0,
                    value=float(valores_base[idx]), step=1.0, key=f"valor_prec_{idx}",
                )
            valores_dinamicos.append(val)

        st.markdown("---")
        potencia_form = st.slider(
            "Potencia IDW (Power)", 1.0, 5.0,
            float(st.session_state["potencia_idw_aplicada"]), 0.5,
        )
        submit_button = st.form_submit_button(label="Aplicar")

    if submit_button:
        st.session_state["valores_aplicados"][variable_seleccionada] = valores_dinamicos
        st.session_state["potencia_idw_aplicada"] = potencia_form

    vals = np.asarray(st.session_state["valores_aplicados"][variable_seleccionada], dtype=float)
    return variable_seleccionada, vals, float(st.session_state["potencia_idw_aplicada"])


def calcular_area_km2(gdf_ageb: Optional[gpd.GeoDataFrame]) -> Optional[float]:
    if gdf_ageb is None or gdf_ageb.empty:
        return None
    try:
        crs_metrico = gdf_ageb.estimate_utm_crs()
        return float(gdf_ageb.to_crs(crs_metrico).geometry.area.sum() / 1_000_000)
    except (ValueError, RuntimeError) as exc:
        LOGGER.warning("No se pudo calcular el área del AGEB: %s", exc)
        return None


def main():
    st.set_page_config(layout="wide", page_title="Monitor Ambiental - Los Mochis")
    st.title("Interpolación Espacial (IDW) - Los Mochis")

    try:
        df_estaciones = cargar_datos(DATA_FILE, obtener_firma_archivo(DATA_FILE))
    except Exception as exc:
        st.error(f"Error al cargar '{DATA_FILE.name}': {exc}")
        st.stop()

    try:
        gdf_ageb = cargar_shapefile(SHAPE_FILE, obtener_firma_shapefile(SHAPE_FILE))
    except Exception as exc:
        st.error(f"Error al cargar '{SHAPE_FILE.name}': {exc}")
        st.stop()

    variable_seleccionada, vals, potencia_idw = construir_sidebar(df_estaciones)
    lons = df_estaciones["Longitud"].to_numpy(dtype=float)
    lats = df_estaciones["Latitud"].to_numpy(dtype=float)
    (minx, miny, maxx, maxy), limite_urbano = obtener_extension(gdf_ageb)

    grid_res = st.sidebar.slider(
        "Resolución del mapa", min_value=100, max_value=600,
        value=DEFAULT_GRID_RESOLUTION, step=50,
    )
    gx, gy = crear_grid(minx, miny, maxx, maxy, grid_res)

    try:
        raster_idw, usa_metros = interpolar_idw(
            lons, lats, vals, gx, gy, potencia_idw, gdf_ageb=gdf_ageb
        )
    except Exception as exc:
        st.error(f"No se pudo interpolar los datos: {exc}")
        st.stop()

    if not usa_metros:
        st.info("La interpolación IDW usa distancias geográficas en grados; el resultado es aproximado.")

    raster_idw = np.where(crear_mascara(limite_urbano, gx, gy), raster_idw, np.nan)
    fig = crear_figura(
        raster_idw=raster_idw, minx=minx, maxx=maxx, miny=miny, maxy=maxy,
        gdf_ageb=gdf_ageb, lons=lons, lats=lats,
        ids_estaciones=df_estaciones["ID Estación"].astype(str).tolist(),
        vals=vals, variable_seleccionada=variable_seleccionada,
        area_km2=calcular_area_km2(gdf_ageb),
    )
    try:
        st.pyplot(fig)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
