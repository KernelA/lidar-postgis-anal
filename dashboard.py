from operator import attrgetter
import numbers

from sqlalchemy import create_engine
from sqlalchemy import sql
from sqlalchemy.orm import sessionmaker
from geoalchemy2 import func as geosql
import geopandas as geo
import numpy as np
import dash_vtk
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, dash_table
from dash.long_callback import DiskcacheLongCallbackManager
import dash_bootstrap_components as dbc
import diskcache
import hydra

from db import LazPoints

with hydra.initialize("configs") as config_dir:
    CONFIG = hydra.compose("insert_data")

FILE_DROPDOWN = "file-selection-id"
CHUNK_DROP_DOWN = "chunk-dropdown-id"
PLOT_3D_ID = "3d-scatter-id"
PROGRESS_BAR_ID = "progress-bar-id"
BUTTON_ID = "draw-button-id"
TABLE_ID = "datatable-id"

CACHE_DIR = "./dashboard-cache"

cache = diskcache.Cache(CACHE_DIR)
long_callback_manager = DiskcacheLongCallbackManager(cache)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = dbc.Container([
    html.Div(children=[dcc.Link("View control", href="https://dash.plotly.com/vtk/intro#view")]),
    html.Label("File:", id="start"),
    dcc.Dropdown(id=FILE_DROPDOWN),
    html.Label("Chunk id:"),
    dcc.Dropdown(id=CHUNK_DROP_DOWN, multi=True),
    html.Div(children=[html.Button("Query points and draw", id=BUTTON_ID, style={"width": "25%"})]),
    html.Progress(id=PROGRESS_BAR_ID, max=str(100),
                  title="Loading...", style={"visibility": "hidden"}),
    dbc.Row(
        children=[
            dbc.Col(
                dash_table.DataTable(id=TABLE_ID, page_action="native", sort_action="native", page_size=10, page_current=0), width=2),
            dbc.Col(id=PLOT_3D_ID)
        ]
    ),
], fluid=True
)


@ app.callback(
    Output(FILE_DROPDOWN, "value"),
    Output(FILE_DROPDOWN, "options"),
    Input("start", "style")
)
def select_files(_):
    engine = create_engine(CONFIG.db.url)
    Session = sessionmaker(engine)

    with Session.begin() as session:
        all_files = list(map(attrgetter("file"), session.query(LazPoints.file).distinct()))

    engine.dispose()

    return all_files[0], all_files


@ app.callback(
    Output(CHUNK_DROP_DOWN, "value"),
    Output(CHUNK_DROP_DOWN, "options"),
    Output(TABLE_ID, "data"),
    Output(TABLE_ID, "columns"),
    Input(FILE_DROPDOWN, "value"),
)
def select_chunk_ids(file):
    engine = create_engine(CONFIG.db.url)
    Session = sessionmaker(engine)

    with Session.begin() as session:
        info_query = sql.select(
            LazPoints.chunk_id.label("chunk id"), geosql.ST_NPoints(LazPoints.points).label("number of points"))\
            .where(LazPoints.file == file).order_by("chunk id")

        data = pd.read_sql(info_query, session.connection())

    engine.dispose()

    return data["chunk id"][0], data["chunk id"], data.to_dict("records"), [{"name": i, "id": i} for i in data.columns]


@ app.long_callback(
    Output(PLOT_3D_ID, "children"),
    State(FILE_DROPDOWN, "value"),
    State(CHUNK_DROP_DOWN, "value"),
    Input(BUTTON_ID, "n_clicks"),
    manager=long_callback_manager,
    running=[
        (Output(FILE_DROPDOWN, "disabled"), True, False),
        (Output(CHUNK_DROP_DOWN, "disabled"), True, False),
        (Output(BUTTON_ID, "disabled"), True, False),
        (
            Output(PROGRESS_BAR_ID, "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        )
    ],
    progress=[Output(PROGRESS_BAR_ID, "value")],
    prevent_initial_call=True


)
def select_chunk(set_progress, file_path, chunk_ids, n_click):
    if isinstance(chunk_ids, numbers.Number):
        chunk_ids = [chunk_ids]

    if not chunk_ids:
        return dash_vtk.View(
            background=[0, 0, 0]
        )

    set_progress([str(10)])

    engine = create_engine(CONFIG.db.url)
    Session = sessionmaker(engine)

    geom_col_name = "geom"
    color_col_name = "color"

    with Session.begin() as session:
        query_points = sql.select(LazPoints.points.label(
            geom_col_name), LazPoints.colors.label(color_col_name)) \
            .filter(LazPoints.file == file_path) \
            .filter(LazPoints.chunk_id.in_(chunk_ids))

        points_data = geo.read_postgis(
            query_points, session.connection(), geom_col=geom_col_name)

    engine.dispose()

    set_progress([str(50)])

    xyz = []
    colors = []

    for row in points_data.itertuples(index=False):
        for point in getattr(row, geom_col_name).geoms:
            for coord in point.coords:
                xyz.extend(coord)

        colors.extend(np.array(getattr(row, color_col_name)).reshape(-1))

    xyz = np.array(xyz)
    xyz -= xyz.mean(axis=0)
    scale = 1 / max(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)), 1e-4)

    xyz *= scale
    xyz = xyz.reshape(-1).tolist()

    vtk_view = dash_vtk.View(
        [
            dash_vtk.PointCloudRepresentation(
                xyz=xyz,
                rgb=colors,
                property={"pointSize": 2}
            )
        ],

        background=[0, 0, 0]
    )

    set_progress([str(100)])

    return vtk_view


if __name__ == "__main__":
    app.run_server(debug=False, dev_tools_ui=False)
