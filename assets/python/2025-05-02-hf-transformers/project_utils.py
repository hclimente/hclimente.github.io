from typing import List, Tuple

from datasets import load_dataset, concatenate_datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers.pipelines.pt_utils import KeyDataset
import umap


def compute_trainer_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": (preds == p.label_ids).mean()}


def plot_umap(
    embeddings: np.ndarray, labels: List[str], palette: str = "tab10", seed: int = 42
) -> plt.Figure:
    # Compute UMAP
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        metric="cosine",
        random_state=seed,
        low_memory=True,
    )
    umap_embeddings = umap_model.fit_transform(embeddings)

    # Prepare DataFrame
    df = pd.DataFrame(umap_embeddings, columns=["UMAP 1", "UMAP 2"])
    df["Label"] = labels
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    fig = go.Figure()
    unique_labels = df["Label"].unique()

    # Ensure all traces have a name based on the label
    for label in unique_labels:
        df_subset = df[df["Label"] == label]

        fig.add_trace(
            go.Scattergl(
                x=df_subset["UMAP 1"],
                y=df_subset["UMAP 2"],
                mode="markers",
                name=label,  # Explicitly set the name for the legend
                marker=dict(opacity=0.8, size=6, line_width=0),
                hoverinfo="text",
                hovertext=[
                    f"UMAP1: {x:.2f}<br>UMAP2: {y:.2f}<br>Label: {lbl}"
                    for x, y, lbl in zip(
                        df_subset["UMAP 1"], df_subset["UMAP 2"], df_subset["Label"]
                    )
                ],
            )
        )

    return fig


def compute_accuracy(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """
    Compute the accuracy of the predictions.
    :param true_labels: List of true labels
    :param predicted_labels: List of predicted labels
    :return: Accuracy
    """
    assert len(true_labels) == len(predicted_labels), (
        "Length of true and predicted labels must be equal"
    )
    acc = (predicted_labels == true_labels).sum() / len(true_labels)

    return acc


def plot_confusion_matrix(
    true_labels: np.ndarray, predicted_labels: np.ndarray, cmap: str = "Blues"
) -> plt.Figure:
    """
    Plot a styled confusion matrix.
    :param true_labels: Ground truth labels
    :param predicted_labels: Predicted labels
    :param normalize: Whether to normalize the confusion matrix (by true label counts)
    :param cmap: Color map for the plot
    :return: Matplotlib figure object
    """

    # Clean plot style
    sns.set_theme(style="white", context="notebook", font_scale=1.2)

    # Unique sorted labels for consistent axis order
    all_labels = sorted(set(true_labels) | set(predicted_labels))

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
    disp.plot(cmap=cmap, ax=ax, colorbar=False)

    # Improve text and layout
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.xaxis.set_tick_params(rotation=45)
    ax.grid(False)

    plt.tight_layout()
    return fig


def create_dataset(prefix: str, ds_list: List[Tuple[str]]):
    dataset = []

    for label_id, (filename, label_name) in enumerate(ds_list):
        ds = load_dataset("text", data_files=f"{prefix}/{filename}.txt", split="train")
        # give every example in this split the same integer label
        ds = ds.map(lambda ex, idx=label_id: {"labels": idx})
        dataset.append(ds)

    # stitch them all back together
    dataset = concatenate_datasets(dataset)

    return dataset


def extract_embeddings_from_pipeline(pipeline, dataset, **pipeline_kwargs):
    embeddings = []
    predictions = []

    for e in pipeline(KeyDataset(dataset, "text"), **pipeline_kwargs):
        if "embedding" in e:
            embeddings.append(e["embedding"])
        if "logits" in e:
            predictions.append(e["logits"])

    embeddings = np.concatenate(embeddings, axis=0)
    if predictions:
        predictions = np.concatenate(predictions, axis=0)

    return embeddings, predictions
