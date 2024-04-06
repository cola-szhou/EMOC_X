from dash import html, dcc, Input, Output
import dash
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from comparation import ComparationLayout
from stastic import StasticLayout

from init import app

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "15rem",
    "padding": "2rem 1rem",
    # "background-color": "#f8f9fa",
    "background-color": "#444444",
}

CONTENT_STYLE = {
    "margin-left": "15rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


def MainLayout():
    sidebar = html.Div(
        children=[
            dbc.Stack(
                [
                    html.Img(
                        src=dash.get_asset_url("emoc_big.png"),
                        style={
                            "height": "auto",
                            "width": "3rem",
                            # "padding-left": "0.5rem",
                        },
                    ),
                    html.H2(
                        "EMOC+X",
                        # style={"padding-top": "0.8rem"},
                    ),
                ],
                direction="horizontal",
                gap=2,
                style={"align-items": "center", "justifyContent": "center"},
            ),
            html.Hr(),
            dbc.Nav(
                [
                    dbc.NavLink("Statistic", href="/", active="exact"),
                    dbc.NavLink(
                        "Metric Comparation",
                        href="/metric_comparation",
                        active="exact",
                    ),
                    dbc.NavLink(
                        "Scatter Plot",
                        href="/scatter_plot",
                        active="exact",
                    ),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )

    content = html.Div(id="page-content", style=CONTENT_STYLE)
    return html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return StasticLayout()
    elif pathname == "/metric_comparation":
        return ComparationLayout()
    elif pathname == "/scatter_plot":
        return html.P("Oh cool, this is page 2!")
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
    )


if __name__ == "__main__":
    app.layout = MainLayout()
    app.run_server(port=8080, debug=False)
