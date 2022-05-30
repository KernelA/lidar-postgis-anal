from operator import attrgetter
import numbers

from sqlalchemy import create_engine
from sqlalchemy import sql
from sqlalchemy.orm import sessionmaker
from geoalchemy2 import func as geosql
import geopandas as geo
import numpy as np
import dash_vtk
from dash import Dash, html, dcc, Input, Output, State
from dash.long_callback import DiskcacheLongCallbackManager
import diskcache
import hydra

from db import LazPoints

with hydra.initialize("configs") as config_dir:
    CONFIG = hydra.compose("insert_data")

FILE_DROPDOWN = "file-selection-id"
CHUNK_DROP_DOWN = "chunk-dropdown-id"
SIGN_INFO_TABLE_ID = "sign-info-table"
PLOT_3D_ID = "3d-scatter-id"
EXTERNAL_DROPDOWN = "external-id-field"
PROGRESS_ID = "progress-id"
RADIUS_SLIDER_ID = "selection-radius-range"
PROGRESS_BAR_ID = "progress-bar-id"
BUTTON_ID = "draw-button-id"

CACHE_DIR = "./dashboard-cache"

cache = diskcache.Cache(CACHE_DIR)
long_callback_manager = DiskcacheLongCallbackManager(cache)

app = Dash(__name__)


app.layout = html.Div([
    html.Div(children=[dcc.Link("View control", href="https://dash.plotly.com/vtk/intro#view")]),
    html.Label("File:", id="start"),
    dcc.Dropdown(id=FILE_DROPDOWN),
    html.Label("Chunk id:"),
    dcc.Dropdown(id=CHUNK_DROP_DOWN, multi=True),
    html.Div(children=[html.Button("Query points and draw", id=BUTTON_ID, style={"width": "25%"})]),
    html.Progress(id=PROGRESS_BAR_ID, max=str(100),
                  title="Loading...", style={"visibility": "hidden"}),
    html.Div(id=PLOT_3D_ID, style={"height": "75vh"})
]
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
    Input(FILE_DROPDOWN, "value"),
)
def select_chunk_ids(file):
    engine = create_engine(CONFIG.db.url)
    Session = sessionmaker(engine)

    with Session.begin() as session:
        all_chunk_ids = list(map(attrgetter("chunk_id"), session.query(
            LazPoints.chunk_id).where(LazPoints.file == file)))

    engine.dispose()

    return all_chunk_ids[0], all_chunk_ids


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

    with Session.begin() as session:
        query_points = sql.select(geosql.ST_DumpPoints(LazPoints.points).geom.label(
            geom_col_name)) \
            .filter(LazPoints.file == file_path) \
            .filter(LazPoints.chunk_id.in_(chunk_ids))

        points = geo.read_postgis(
            query_points, session.connection(), geom_col=geom_col_name)

    engine.dispose()

    set_progress([str(50)])

    coords = np.vstack(
        (points[geom_col_name].x, points[geom_col_name].y, points[geom_col_name].z)).T.reshape(-1)

    vtk_view = dash_vtk.View(
        [
            dash_vtk.PointCloudRepresentation(
                xyz=coords,
                property={"pointSize": 2}
            )
        ],
        background=[0, 0, 0]
    )

    set_progress([str(100)])

    return vtk_view


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_ui=True)
