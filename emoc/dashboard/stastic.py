from dash import html, dcc, Input, Output, ctx
import dash
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc

from init import app

alg_list = ["NSGA-II", "NSGA-III", "MOEA/D"]
prob_list = ["dtlz1", "dtlz2", "dtlz3"]
metric_list = ["IGD", "GD", "GD+", "IGD+"]


def show_table(alg, prob, metric, orientation=1):
    # 这里把需要显示的alg, prob, metric传进来，然后需要给我一个dataframe进行展示
    df = pd.read_pickle("emoc/report.pkl")
    df.index.names = ["Problem", "Metric"]
    table = dbc.Table.from_dataframe(
        df, striped=True, bordered=True, hover=True, index=True, color="dark"
    )
    return table


def StasticLayout():
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
                                        id="static_select_all_alg",
                                        label="(Select All)",
                                        value=True,
                                    ),
                                    dbc.Checklist(
                                        options=[
                                            {"label": i, "value": i} for i in alg_list
                                        ],
                                        value=alg_list,
                                        id="static_check_alg",
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
                                        id="static_select_all_prob",
                                        label="(Select All)",
                                        value=True,
                                    ),
                                    dbc.Checklist(
                                        options=[
                                            {"label": i, "value": i} for i in prob_list
                                        ],
                                        value=prob_list,
                                        id="static_check_prob",
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
                                        id="static_select_all_metric",
                                        label="(Select All)",
                                        value=True,
                                    ),
                                    dbc.Checklist(
                                        options=[
                                            {"label": i, "value": i}
                                            for i in metric_list
                                        ],
                                        value=metric_list,
                                        id="static_check_metric",
                                    ),
                                ]
                            )
                        ],
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Stack(
                            [
                                html.P("Test"),
                                dbc.Select(
                                    ["RankSumTest", "WilcoxonTest", "FriedmanTest"],
                                    value="RankSumTest",
                                    style={"margin-top": "-0.7rem"},
                                ),
                            ],
                            direction="horizontal",
                            gap=2,
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        [
                            dbc.Button(
                                [html.I(className="fa-solid fa-floppy-disk")],
                                color="primary",
                                className="me-1",
                            )
                        ],
                        style={"margin-top": "-0.4rem"},
                        width="auto",
                    ),
                ]
            ),
            # html.Hr(),
            dbc.Card(
                [
                    dbc.CardHeader(
                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    label="Orientation 1",
                                    tab_id="tab-1",
                                    active_label_style={
                                        "background-color": "#375a7f",
                                        "color": "white",
                                    },
                                ),
                                dbc.Tab(
                                    label="Orientation 2",
                                    tab_id="tab-2",
                                    active_label_style={
                                        "background-color": "#375a7f",
                                        "color": "white",
                                    },
                                ),
                                dbc.Tab(
                                    label="Orientation 3",
                                    tab_id="tab-3",
                                    active_label_style={
                                        "background-color": "#375a7f",
                                        "color": "white",
                                    },
                                ),
                            ],
                            id="stastic_card_tabs",
                            active_tab="tab-1",
                        )
                    ),
                    dbc.CardBody(id="stastic_card_content"),
                ]
            ),
        ]
    )

    return layout


@app.callback(
    Output("stastic_card_content", "children"),
    Input("stastic_card_tabs", "active_tab"),
    Input("static_check_alg", "value"),
    Input("static_check_prob", "value"),
    Input("static_check_metric", "value"),
)
def switch_tab(at, alg, prob, metric):
    if at == "tab-1":
        return show_table(alg, prob, metric, 1)
    elif at == "tab-2":
        return show_table(alg, prob, metric, 2)
    elif at == "tab-3":
        return show_table(alg, prob, metric, 3)
    return html.P("This shouldn't ever be displayed...")


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


create_callback("static_select_all_alg", "static_check_alg", alg_list)
create_callback("static_select_all_prob", "static_check_prob", prob_list)
create_callback("static_select_all_metric", "static_check_metric", metric_list)


if __name__ == "__main__":
    app.layout = StasticLayout()
    app.run_server(port=8080, debug=True)
