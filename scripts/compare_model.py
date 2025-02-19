import pandas as pd
import plotly.express as px
import plotly.io as pio


base_path = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/"
data_dir = "data/metrics_save/"
files_to_plot = ["resultats_ColHead_Regu_HeadPre0_Clip3.csv"]
model_names = ["Pretrain0"]

# Charger les datasets
data_files = [pd.read_csv(base_path + data_dir + file) for i, file in enumerate(files_to_plot)]
for i, df in enumerate(data_files):
    df["model"] = model_names[i]     #f"Model_{i}"  # Ajouter un identifiant pour chaque modèle
df = pd.concat(data_files)  # Fusionner en un seul DataFrame

# Récupérer les valeurs uniques
metrics = df["metric"].unique()
finetune_bases = df["finetune_base"].unique()
inference_bases = df["inference_base"].unique()
models = df["model"].unique()

# Filtre par défaut
default_metric = metrics[0]
default_finetune = finetune_bases[0]
default_inference = inference_bases[0]
default_models = models  # Tous les modèles sélectionnés par défaut

# Filtrer les données initiales
filtered_df = df[
    (df["metric"] == default_metric) &
    (df["finetune_base"] == default_finetune) &
    (df["inference_base"] == default_inference) &
    (df["model"].isin(default_models))
]

# Créer le graphique initial
fig = px.line(
    filtered_df,
    x="plage",
    y="value",
    color="model",
    markers=True,
    title=f"Métrique: {default_metric} | Finetune: {default_finetune} | Inference: {default_inference}"
)

# Ajouter les menus interactifs
fig.update_layout(
    updatemenus=[
        # Menu pour la métrique
        {
            "buttons": [
                {
                    "label": metric,
                    "method": "update",
                    "args": [
                        {
                            "y": [df[(df["metric"] == metric) & 
                                     (df["finetune_base"] == default_finetune) & 
                                     (df["inference_base"] == default_inference) & 
                                     (df["model"].isin(default_models))]["value"]],
                            "x": [df[(df["metric"] == metric) & 
                                     (df["finetune_base"] == default_finetune) & 
                                     (df["inference_base"] == default_inference) & 
                                     (df["model"].isin(default_models))]["plage"]]
                        },
                        {"title": f"Métrique: {metric} | Finetune: {default_finetune} | Inference: {default_inference}"}
                    ],
                }
                for metric in metrics
            ],
            "direction": "down",
            "showactive": True,
            "x": 0.1,
            "y": 1.15,
            "xanchor": "left",
            "yanchor": "top",
        },
        # Menu pour la base de fine-tuning
        {
            "buttons": [
                {
                    "label": finetune,
                    "method": "update",
                    "args": [
                        {
                            "y": [df[(df["metric"] == default_metric) & 
                                     (df["finetune_base"] == finetune) & 
                                     (df["inference_base"] == default_inference) & 
                                     (df["model"].isin(default_models))]["value"]],
                            "x": [df[(df["metric"] == default_metric) & 
                                     (df["finetune_base"] == finetune) & 
                                     (df["inference_base"] == default_inference) & 
                                     (df["model"].isin(default_models))]["plage"]]
                        },
                        {"title": f"Métrique: {default_metric} | Finetune: {finetune} | Inference: {default_inference}"}
                    ],
                }
                for finetune in finetune_bases
            ],
            "direction": "down",
            "showactive": True,
            "x": 0.3,
            "y": 1.15,
            "xanchor": "left",
            "yanchor": "top",
        },
        # Menu pour la base d'inférence
        {
            "buttons": [
                {
                    "label": inference,
                    "method": "update",
                    "args": [
                        {
                            "y": [df[(df["metric"] == default_metric) & 
                                     (df["finetune_base"] == default_finetune) & 
                                     (df["inference_base"] == inference) & 
                                     (df["model"].isin(default_models))]["value"]],
                            "x": [df[(df["metric"] == default_metric) & 
                                     (df["finetune_base"] == default_finetune) & 
                                     (df["inference_base"] == inference) & 
                                     (df["model"].isin(default_models))]["plage"]]
                        },
                        {"title": f"Métrique: {default_metric} | Finetune: {default_finetune} | Inference: {inference}"}
                    ],
                }
                for inference in inference_bases
            ],
            "direction": "down",
            "showactive": True,
            "x": 0.5,
            "y": 1.15,
            "xanchor": "left",
            "yanchor": "top",
        },
        # Menu pour la sélection multiple des modèles
        {
            "buttons": [
                {
                    "label": "All Models",
                    "method": "update",
                    "args": [
                        {
                            "y": [df[(df["metric"] == default_metric) & 
                                     (df["finetune_base"] == default_finetune) & 
                                     (df["inference_base"] == default_inference) & 
                                     (df["model"].isin(models))]["value"]],
                            "x": [df[(df["metric"] == default_metric) & 
                                     (df["finetune_base"] == default_finetune) & 
                                     (df["inference_base"] == default_inference) & 
                                     (df["model"].isin(models))]["plage"]]
                        },
                        {"title": f"Métrique: {default_metric} | Finetune: {default_finetune} | Inference: {default_inference} | All Models"}
                    ],
                }
            ] + [
                {
                    "label": model,
                    "method": "update",
                    "args": [
                        {
                            "y": [df[(df["metric"] == default_metric) & 
                                     (df["finetune_base"] == default_finetune) & 
                                     (df["inference_base"] == default_inference) & 
                                     (df["model"] == model)]["value"]],
                            "x": [df[(df["metric"] == default_metric) & 
                                     (df["finetune_base"] == default_finetune) & 
                                     (df["inference_base"] == default_inference) & 
                                     (df["model"] == model)]["plage"]]
                        },
                        {"title": f"Métrique: {default_metric} | Finetune: {default_finetune} | Inference: {default_inference} | Model: {model}"}
                    ],
                }
                for model in models
            ],
            "direction": "down",
            "showactive": True,
            "x": 0.7,
            "y": 1.15,
            "xanchor": "left",
            "yanchor": "top",
        },
    ]
)

# Sauvegarder en fichier HTML interactif
pio.write_html(fig, "visualisation_interactive.html")

print("✅ Fichier HTML enregistré : visualisation_interactive.html")
