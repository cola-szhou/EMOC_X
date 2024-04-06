from dash_extensions.enrich import (
    DashProxy,
    MultiplexerTransform,
    NoOutputTransform,
    TriggerTransform,
)
import dash_bootstrap_components as dbc

app = DashProxy(
    __name__,
    title="EMOC+X",
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME],
    transforms=[MultiplexerTransform(), NoOutputTransform(), TriggerTransform()],
)
