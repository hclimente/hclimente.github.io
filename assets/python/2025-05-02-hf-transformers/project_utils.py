from typing import List, Tuple

from datasets import load_dataset, concatenate_datasets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import umap

palette = {
    "H. sapiens": "#1f77b4",  # blue
    "M. musculus": "#6baed6",  # light blue
    "D. melanogaster": "#8c6bb1",  # violet
    "A. thaliana": "#2ca02c",  # green
    "S. cerevisiae": "#d62728",  # red
    "E. coli": "#ff7f0e",  # orange
}


def plot_umap(embeddings: np.ndarray, labels: List[str]) -> matplotlib.figure.Figure:
    # compute UMAP
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        metric="cosine",
        random_state=42,
        low_memory=True,
        verbose=True,
    )
    umap_embeddings = umap_model.fit_transform(embeddings)

    # prepare the data for plotting
    df = pd.DataFrame(umap_embeddings, columns=["x", "y"])
    df["species"] = labels

    ## shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(3, 3))
    sns.scatterplot(
        x="x",
        y="y",
        hue="species",
        data=df,
        alpha=0.5,
        s=10,
        palette=palette,
    )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.gca().legend(loc="center left", bbox_to_anchor=(1, 0.5))

    return plt.gcf()


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
