import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib.path import Path
from shapely.ops import unary_union

# ==========================================
# 1. CONFIGURACIÓN INICIAL
# ==========================================
st.set_page_config(layout="wide", page_title="Monitor Ambiental - Los Mochis")
st.title("Interpolación Espacial (IDW) - Los Mochis")

# ==========================================
# 2. CARGA DE DATOS (EN CACHÉ)
# ==========================================
@st.cache_data
def cargar_datos():
    df = pd.read_excel('TEMP2025.xls')
    df.columns = df.columns.str.strip()
    # Si la columna Precipitación no existe en el Excel, la creamos con 0 por defecto
    if 'Precipitación' not in df.columns:
        df['Precipitación'] = 0.0
    return df

@st.cache_data
def cargar_shapefile():
    try:
        gdf = gpd.read_file('AGEM_LMM2026.shp')
        if 'CVE_LOC' in gdf.columns:
            gdf = gdf[gdf['CVE_LOC'] == '0001']
        if 'CVE_MUN' in gdf.columns:
            gdf = gdf[gdf['CVE_MUN'] == '001']
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception:
        return None

df_estaciones = cargar_datos()
gdf_ageb = cargar_shapefile()

# ==========================================
# 3. INTERFAZ Y FORMULARIO (BARRA LATERAL)
# ==========================================
st.sidebar.header("Configuración")
variable_seleccionada = st.sidebar.radio(
    "Selecciona la Variable a Interpolar:", 
    ["Temperatura (°C)", "Precipitación (mm)"]
)

with st.sidebar.form(key='formulario_interpolacion'):
    st.subheader(f"Valores de {variable_seleccionada.split(' ')[0]}")
    valores_dinamicos = []

    for idx, row in df_estaciones.iterrows():
        if variable_seleccionada == "Temperatura (°C)":
            val = st.number_input(
                label=f"{row['ID Estación']}:",
                min_value=-10.0, max_value=70.0,
                value=float(row['Temperatura']), step=0.1
            )
        else:
            val = st.number_input(
                label=f"{row['ID Estación']}:",
                min_value=0.0, max_value=300.0,
                value=float(row['Precipitación']), step=1.0
            )
        valores_dinamicos.append(val)

    st.markdown("---")
    potencia_idw = st.slider("Potencia IDW (Power)", 1.0, 5.0, 2.0, 0.5)
    submit_button = st.form_submit_button(label='Aplicar')

# Extraer coordenadas y valores
lons = df_estaciones['Longitud'].values
lats = df_estaciones['Latitud'].values
vals = np.array(valores_dinamicos)

# ==========================================
# 4. PREPARACIÓN ESPACIAL Y MÁSCARA
# ==========================================
if gdf_ageb is not None and not gdf_ageb.empty:
    minx, miny, maxx, maxy = gdf_ageb.total_bounds
    pad = 0.008
    minx, maxx = minx - pad, maxx + pad
    miny, maxy = miny - pad, maxy + pad

    limite_urbano = unary_union(gdf_ageb.geometry)
    poligonos = [limite_urbano] if limite_urbano.geom_type == 'Polygon' else list(limite_urbano.geoms)
else:
    minx, miny, maxx, maxy = -109.05, 25.72, -108.93, 25.85
    poligonos = []

# ==========================================
# 5. CÁLCULO IDW
# ==========================================
def idw_calc(x, y, z, gx, gy, p=2.0):
    dist = np.sqrt((gx[..., np.newaxis] - x)**2 + (gy[..., np.newaxis] - y)**2)
    dist = np.where(dist < 1e-10, 1e-10, dist)
    w = 1.0 / (dist ** p)
    return np.sum(w * z, axis=-1) / np.sum(w, axis=-1)

grid_res = 350
xi = np.linspace(minx, maxx, grid_res)
yi = np.linspace(miny, maxy, grid_res)
gx, gy = np.meshgrid(xi, yi)

raster_idw = idw_calc(lons, lats, vals, gx, gy, p=potencia_idw)

# Recorte urbano
if poligonos:
    mask = np.zeros(gx.shape, dtype=bool)
    puntos_malla = np.column_stack((gx.flatten(), gy.flatten()))
    for poly in poligonos:
        path = Path(np.asarray(poly.exterior.coords))
        mask = mask | path.contains_points(puntos_malla).reshape(gx.shape)
    raster_idw = np.where(mask, raster_idw, np.nan)

# ==========================================
# 6. ESTILOS DINÁMICOS SEGÚN VARIABLE
# ==========================================
if variable_seleccionada == "Temperatura (°C)":
    colores = ['#228B22', '#7CFC00', '#FFFF00', '#FFA500', '#FF4500', '#B22222', '#800000', '#300000']
    titulo_mapa = "Land Surface Temperature (LST)"
    unidad = "°C"
    color_pts = "#E6E6FA"
else:
    colores = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
    titulo_mapa = "Precipitación Acumulada"
    unidad = "mm"
    color_pts = "#FFD700"

cmap_personalizado = LinearSegmentedColormap.from_list("CustomMap", colores, N=256)

# Escala dinámica estricta
vmin = vals.min()
vmax = vals.max()
if vmin == vmax: 
    vmin = max(0.0, vmin - 1.0)
    vmax += 1.0

# ==========================================
# 7. RENDERIZADO CARTOGRÁFICO
# ==========================================
fig, ax = plt.subplots(figsize=(11, 10), dpi=200)

im = ax.imshow(
    raster_idw,
    extent=[minx, maxx, miny, maxy],
    origin="lower",
    cmap=cmap_personalizado,
    vmin=vmin,
    vmax=vmax,
    alpha=0.95,
    interpolation='bilinear'
)

if gdf_ageb is not None and not gdf_ageb.empty:
    gdf_ageb.boundary.plot(ax=ax, color='black', linewidth=0.35, alpha=0.45)

ax.scatter(lons, lats, color=color_pts, edgecolor='black', s=55, linewidth=1, zorder=6)

for idx, txt in enumerate(df_estaciones['ID Estación']):
    ax.annotate(
        f"{txt}\n{vals[idx]:.1f} {unidad}",
        (lons[idx], lats[idx]),
        textcoords="offset points",
        xytext=(0, 6),
        ha="center",
        fontsize=7,
        weight="bold",
        color='black',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75, edgecolor='gray'),
        zorder=7
    )

def deg_to_dms(val, is_lat=True):
    hemi = ('N' if val >= 0 else 'S') if is_lat else ('E' if val >= 0 else 'W')
    v = abs(val)
    d = int(v)
    m = int((v - d) * 60)
    s = int(round((v - d - m/60) * 3600))
    if s == 60:
        m += 1; s = 0
    return f"{d}°{m}'{s}\"{hemi}"

ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: deg_to_dms(val, False)))
ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: deg_to_dms(val, True)))
ax.tick_params(axis='both', which='major', labelsize=8, direction='out')

ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.grid(True, linestyle='-', linewidth=0.4, color='gray', alpha=0.6)
ax.set_title(titulo_mapa, fontsize=14, fontweight='bold', pad=15)

# Rosa de los vientos (Norte)
ax.annotate('N', xy=(0.95, 0.94), xytext=(0.95, 0.88),
            xycoords='axes fraction', ha='center', va='center',
            fontsize=12, fontweight='bold',
            arrowprops=dict(facecolor='black', edgecolor='black', width=2, headwidth=7))

# Cajetín de leyenda
cajetin = patches.Rectangle((0.68, 0.03), 0.30, 0.16, transform=ax.transAxes,
                            facecolor='white', edgecolor='black', linewidth=0.8, zorder=8)
ax.add_patch(cajetin)

ax.text(0.70, 0.16, titulo_mapa, transform=ax.transAxes, fontsize=7, fontweight='bold', zorder=9)
ax.text(0.70, 0.135, f"Max:  {vmax:.1f} {unidad}", transform=ax.transAxes, fontsize=6.5, zorder=9)
ax.text(0.70, 0.115, f"Mean: {vals.mean():.1f} {unidad}", transform=ax.transAxes, fontsize=6.5, zorder=9)
ax.text(0.70, 0.095, f"Min:  {vmin:.1f} {unidad}", transform=ax.transAxes, fontsize=6.5, zorder=9)

cax = fig.add_axes([0.89, 0.17, 0.015, 0.10])
cbar = plt.colorbar(im, cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=6)

st.pyplot(fig)