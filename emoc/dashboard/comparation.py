from dash import html, dcc, Input, Output, ctx
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc

from init import app

# app = dash.Dash(external_stylesheets=[dbc.themes.MINTY, dbc.icons.FONT_AWESOME])

alg_list = ["NSGA-II", "NSGA-III", "MOEA/D"]
prob_list = ["ZDT1", "ZDT2", "ZDT3", "DTLZ1", "DTLZ2"]
metric_list = ["IGD", "GD", "Hypervolume"]


def get_graphs(alg, prob, metric):
    import plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = pd.read_csv(
        "https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv"
    )
    figure = px.histogram(df, x="continent", y="lifeExp", histfunc="avg")
    figures = []
    for i in range(len(metric)):
        figures.append(figure)
    return figures


def set_card(title, figure):
    return dbc.Card(
        [
            dbc.CardHeader(title, style={"color": "white"}),
            dcc.Graph(figure=figure),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Button(
                                [html.I(className="fa-solid fa-expand")],
                                color="primary",
                            )
                        ],
                        width="auto",
                    ),
                    dbc.Col(
                        [
                            dbc.Button(
                                [html.I(className="fa-solid fa-floppy-disk")],
                                color="primary",
                            )
                        ],
                        width="auto",
                    ),
                ],
                justify="end",
                style={
                    "margin-top": "0.4rem",
                    "margin-right": "0.2rem",
                    "margin-bottom": "0.4rem",
                },
            ),
        ]
    )


def ComparationLayout():
    layout = html.Div(
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Details(
                                [
                                    html.Summary("Select Algorithms"),
                                    html.Br(),
                                    dbc.Checkbox(
                                        id="compare_select_all_alg",
                                        label="(Select All)",
                                        value=True,
                                    ),
                                    dbc.Checklist(
                                        options=[
                                            {"label": i, "value": i} for i in alg_list
                                        ],
                                        value=alg_list,
                                        id="compare_check_alg",
                                    ),
                                ]
                            )
                        ],
                        width="auto",
                    ),
                    dbc.Col(
                        [
                            html.Details(
                                [
                                    html.Summary("Select Problems"),
                                    # html.Br(),
                                    dbc.Checkbox(
                                        id="compare_select_all_prob",
                                        label="(Select All)",
                                        value=True,
                                    ),
                                    dbc.Checklist(
                                        options=[
                                            {"label": i, "value": i} for i in prob_list
                                        ],
                                        value=prob_list,
                                        id="compare_check_prob",
                                    ),
                                ]
                            )
                        ],
                        width="auto",
                    ),
                    dbc.Col(
                        [
                            html.Details(
                                [
                                    html.Summary("Select Metrics"),
                                    html.Br(),
                                    dbc.Checkbox(
                                        id="compare_select_all_metric",
                                        label="(Select All)",
                                        value=True,
                                    ),
                                    dbc.Checklist(
                                        options=[
                                            {"label": i, "value": i}
                                            for i in metric_list
                                        ],
                                        value=metric_list,
                                        id="compare_check_metric",
                                    ),
                                ]
                            )
                        ],
                        width="auto",
                    ),
                ]
            ),
            html.Hr(),
            html.Div(id="compare_graphs"),
        ]
    )
    return layout


def create_callback(select_all_name, check_name, list):
    @app.callback(
        Output(select_all_name, "value"),
        Output(check_name, "value"),
        Input(select_all_name, "value"),
        Input(check_name, "value"),
    )
    def check_all(select_all, check):
        input_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if input_id == select_all_name:
            check = list if select_all else []
        else:
            if set(check) == set(list):
                select_all = True
            else:
                select_all = False
        return select_all, check


create_callback("compare_select_all_alg", "compare_check_alg", alg_list)
create_callback("compare_select_all_prob", "compare_check_prob", prob_list)
create_callback("compare_select_all_metric", "compare_check_metric", metric_list)


@app.callback(
    Output("compare_graphs", "children"),
    Input("compare_check_alg", "value"),
    Input("compare_check_prob", "value"),
    Input("compare_check_metric", "value"),
)
def get_graph_layout(alg, prob, metric):
    if len(alg) == 0 or len(prob) == 0 or len(metric) == 0:
        return html.H1("Select at least one algorithm, problem and metric")
    else:
        graphs = get_graphs(alg, prob, metric)
        children = []
        for i in range(int(len(graphs) / 2)):
            children.append(
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                set_card(f"Graph {i*2+1}", graphs[i * 2]),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                set_card(f"Graph {i*2+2}", graphs[i * 2 + 1]),
                            ],
                            width=6,
                        ),
                    ],
                    style={"margin-bottom": "1rem"},
                    justify="center",
                )
            )
        if len(graphs) % 2 == 1:
            children.append(
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                set_card(
                                    f"Graph {len(graphs)}",
                                    graphs[-1],
                                ),
                            ],
                            width=6,
                        ),
                    ],
                )
            )
        return children
