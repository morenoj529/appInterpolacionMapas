import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


def _load_app_module():
    if "streamlit" not in sys.modules:
        def cache_data(func=None, **kwargs):
            if func is None:
                return lambda f: f
            return func

        streamlit_stub = types.SimpleNamespace(
            cache_data=cache_data,
            session_state={},
        )
        sys.modules["streamlit"] = streamlit_stub

    return importlib.import_module("app")


app = _load_app_module()
cargar_datos_impl = getattr(app.cargar_datos, "__wrapped__", app.cargar_datos)


class CargarDatosValidationTests(unittest.TestCase):
    def test_falla_si_faltan_columnas_requeridas(self):
        df = pd.DataFrame(
            {
                "ID Estación": ["A"],
                "Longitud": [-108.99],
                "Temperatura": [31.5],
            }
        )
        with patch("app.pd.read_excel", return_value=df):
            with self.assertRaisesRegex(ValueError, "Faltan columnas obligatorias"):
                cargar_datos_impl(Path(__file__))

    def test_falla_si_columnas_numericas_son_invalidas(self):
        df = pd.DataFrame(
            {
                "ID Estación": ["A"],
                "Longitud": ["no-num"],
                "Latitud": [25.8],
                "Temperatura": [31.5],
            }
        )
        with patch("app.pd.read_excel", return_value=df):
            with self.assertRaisesRegex(ValueError, "valores faltantes o no numéricos"):
                cargar_datos_impl(Path(__file__))


class IdwTests(unittest.TestCase):
    def test_falla_con_estaciones_vacias(self):
        gx = np.array([[0.0]])
        gy = np.array([[0.0]])
        with self.assertRaisesRegex(ValueError, "No hay estaciones"):
            app.idw_calc([], [], [], gx, gy, p=2.0)

    def test_falla_con_potencia_invalida(self):
        gx = np.array([[0.0]])
        gy = np.array([[0.0]])
        with self.assertRaisesRegex(ValueError, "potencia IDW"):
            app.idw_calc([0.0], [0.0], [10.0], gx, gy, p=0.0)

    def test_usa_valor_exacto_en_coincidencia(self):
        gx = np.array([[1.0, 2.0]])
        gy = np.array([[1.0, 2.0]])
        salida = app.idw_calc([1.0, 5.0], [1.0, 5.0], [42.0, 10.0], gx, gy, p=2.0)
        self.assertEqual(salida.shape, gx.shape)
        self.assertEqual(salida[0, 0], 42.0)


class CrearFiguraTests(unittest.TestCase):
    def test_agrega_basemap_mascara_y_zorder_interpolacion(self):
        raster_idw = np.array([[1.0, 2.0], [3.0, 4.0]])
        gdf_ageb = types.SimpleNamespace(
            empty=False,
            plot=types.SimpleNamespace(),
            boundary=types.SimpleNamespace(),
        )
        gdf_ageb.plot = unittest.mock.Mock()
        gdf_ageb.boundary.plot = unittest.mock.Mock()

        with patch("app.cx.add_basemap") as add_basemap_mock:
            fig = app.crear_figura(
                raster_idw=raster_idw,
                minx=-109.0,
                maxx=-108.9,
                miny=25.7,
                maxy=25.8,
                gdf_ageb=gdf_ageb,
                lons=np.array([-108.95]),
                lats=np.array([25.75]),
                ids_estaciones=["EST-1"],
                vals=np.array([30.0]),
                variable_seleccionada="Temperatura (°C)",
            )

        add_basemap_mock.assert_called_once_with(
            fig.axes[0],
            crs="EPSG:4326",
            source=app.cx.providers.Esri.WorldTopoMap,
            zorder=0,
            attribution=False,
        )
        gdf_ageb.plot.assert_called_once_with(
            ax=fig.axes[0], facecolor="white", edgecolor="none", zorder=1
        )
        self.assertGreaterEqual(fig.axes[0].images[0].get_zorder(), 2)
        app.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
