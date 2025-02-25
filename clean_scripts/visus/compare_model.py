import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go

# Charger les données principales
base_path = "./"
#files_to_plot = ["resnet_color_h0_c3_b128", "resnet_color_h0_c3_b256", "resnet_color_h5_c3_b256","resnet_color_h10_c3_b256", "resnet_supervised_b128", "resnet_supervised_b256", "resnet50_baseline", "resnet50_unsup_color"]
#model_names = ["resnet_color_h0_c3_b128", "resnet_color_h0_c3_b256", "resnet_color_h5_c3_b256","resnet_color_h10_c3_b256", "resnet_supervised_b128", "resnet_supervised_b256", "resnet50_basique", "resnet50_color"]
files_to_plot = ["basic_sup", "basic_t07_reg_fine", "basic_t07_reg_coretune", "basic_t01_noreg_coretune",
                "resnet_t01_finetune", "resnet_t01_finecon", "resnet_t01_coretune", "resnet_t07_finetune", "resnet_t07_coretune","resnet_sup", "resultats_basic_baseline"]
model_names = ["basic_supervisé", "basic_t07_regu_finetune", "basic_t07_reg_coretune", "basic_t01_noreg_coretune",
                "resnet_t01_finetune", "resnet_t01_finecon", "resnet_t01_coretune", "resnet_t07_finetune", "resnet_t07_coretune", "resnet_supervisé", "basic_supervised_classic"]


#data_files = [pd.read_csv(base_path + "resultats_" + file + ".csv") for file in files_to_plot]
data_files = [pd.read_csv(base_path + file + ".csv") for file in files_to_plot]

for i, df in enumerate(data_files):
    df["model"] = model_names[i]
df = pd.concat(data_files)

# Charger la distribution des bases
bases_distr = pd.read_csv(base_path + "bases_distributions.csv")
print(bases_distr)

metrics = df["metric"].unique()
finetune_bases = df["finetune_base"].unique()
inference_bases = df["inference_base"].unique()
models = df["model"].unique()

# Initialiser l'application Dash
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Visualisation Interactive"),
    dcc.Dropdown(
        id='metric-dropdown',
        options=[{'label': m, 'value': m} for m in metrics],
        value=metrics[0],
        clearable=False
    ),
    dcc.Dropdown(
        id='finetune-dropdown',
        options=[{'label': f, 'value': f} for f in finetune_bases],
        value=finetune_bases[0],
        clearable=False
    ),
    dcc.Dropdown(
        id='inference-dropdown',
        options=[{'label': i, 'value': i} for i in inference_bases],
        value=inference_bases[0],
        clearable=False
    ),
    dcc.Dropdown(
        id='model-dropdown',
        options=[{'label': m, 'value': m} for m in models],
        value=[models[0]],
        multi=True
    ),
    dcc.Checklist(
        id='toggle-density',
        options=[{'label': 'Afficher la densité', 'value': 'density'}],
        value=[]
    ),
    dcc.Graph(id='line-plot')
])

@app.callback(
    Output('line-plot', 'figure'),
    [
        Input('metric-dropdown', 'value'),
        Input('finetune-dropdown', 'value'),
        Input('inference-dropdown', 'value'),
        Input('model-dropdown', 'value'),
        Input('toggle-density', 'value')
    ]
)
def update_plot(selected_metric, selected_finetune, selected_inference, selected_models, toggle_density):
    if isinstance(selected_models, str):
        selected_models = [selected_models]

    # Filtrer les données
    filtered_df = df[
        (df["metric"] == selected_metric) &
        (df["finetune_base"] == selected_finetune) &
        (df["inference_base"] == selected_inference) &
        (df["model"].isin(selected_models))
    ]

    fig = go.Figure()

    # Ajouter les courbes des modèles (métriques) sur l'axe principal
    for model in selected_models:
        model_data = filtered_df[filtered_df["model"] == model]
        fig.add_trace(go.Scatter(
            x=model_data["plage"],
            y=model_data["value"],
            mode='lines+markers',
            name=f"Métrique - {model}",
            yaxis="y1"
        ))

    # Ajouter les données de bases_distr sur l'axe secondaire
    base_types = ["unsup", "inference", "finetune"]
    colors = ["red", "green", "orange"]  # Assigner des couleurs différentes

    for base_type, color in zip(base_types, colors):
        bases_data = bases_distr[((bases_distr["kind"] == base_type) | (bases_distr["kind"] == base_type+'_'+selected_finetune) ) & (bases_distr["split"]=="uniforme")]
        y_values = bases_data["count"]
        if 'density' in toggle_density:
            y_values = y_values / y_values.sum()
        fig.add_trace(go.Bar(
            x=bases_data["plage_value"],
            y=y_values,
            name=f"Distribution - {base_type}",
            yaxis="y2",
            opacity=0.6,
            marker_color=color
        ))

    # Mise en forme des axes
    fig.update_layout(
        title=f"Métrique: {selected_metric} | Finetune: {selected_finetune} | Inference: {selected_inference}",
        xaxis_title="Plage",
        barmode="group",  # Permet d'afficher les barres côte à côte
        yaxis=dict(
            title="Métrique",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title="Distribution des Bases",
            overlaying="y",
            side="right",
            titlefont=dict(color="red"),
            tickfont=dict(color="red"),
            showgrid=False
        ),
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
