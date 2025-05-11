from typing import List, Tuple

from datasets import load_dataset, concatenate_datasets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    # Set plot style
    sns.set_theme(style="white", context="notebook", font_scale=1.2)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatterplot
    sns.scatterplot(
        x="UMAP 1",
        y="UMAP 2",
        hue="Label",
        data=df,
        alpha=0.6,
        s=40,
        palette=palette,
        edgecolor="none",
        ax=ax,
    )

    # Move legend outside to bottom-right
    box = ax.get_position()
    ax.set_position(
        [box.x0, box.y0 + 0.1, box.width * 0.85, box.height]
    )  # shrink width a bit

    ax.legend(
        title="",
        loc="lower left",
        bbox_to_anchor=(1.02, 0),  # bottom right, just outside
        borderaxespad=0.0,
        frameon=False,
        fontsize="medium",
        labelspacing=0.4,
        handletextpad=0.5,
    )

    sns.despine(trim=True)
    plt.tight_layout()

    return fig


def compute_accuracy(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """
    Compute the accuracy of the predictions.
    :param true_labels: List of true labels
    :param predicted_labels: List of predicted labels
    :return: Accuracy
    """
    assert len(true_labels) == len(
        predicted_labels
    ), "Length of true and predicted labels must be equal"
    acc = (predicted_labels == true_labels).sum() / len(true_labels)

    return acc


def plot_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> matplotlib.figure.Figure:
    """
    Plot the confusion matrix.
    :param true_labels: List of true labels
    :param predicted_labels: List of predicted labels
    """

    all_labels = list(set(true_labels) | set(predicted_labels))

    cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
    disp.plot(cmap="Blues")
    plt.grid(False)
    plt.xticks(rotation=45)

    return plt.gcf()


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
