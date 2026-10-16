import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plots(history, labels, predictions, directory):
    for metric in ("loss", "accuracy"):
        fig, ax = plt.subplots()
        for split in ("train", "validation"):
            ax.plot([r["epoch"] for r in history], [r[f"{split}_{metric}"] for r in history], label=split)
        ax.set(xlabel="Epoch", ylabel=metric.title())
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory/f"{metric}.png", dpi=120)
        plt.close(fig)
    n = int(max(labels.max(), predictions.max()))+1
    matrix = np.zeros((n, n), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)
    fig, ax = plt.subplots()
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > matrix.max()/2 else "black")
    ax.set(xlabel="Predicted label", ylabel="True label", xticks=range(n), yticks=range(n))
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(directory/"confusion_matrix.png", dpi=120)
    plt.close(fig)
