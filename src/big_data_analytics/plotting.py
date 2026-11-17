import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from big_data_analytics.config import FIGURES_DIR  # noqa: E402


PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#6250d6", "#e34948",
]

SENTIMENT_COLORS = {
    "Positive": "#2a78d6",
    "Neutral": "#a3a29b",
    "Negative": "#e34948",
}

sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#c3c2b7",
    "grid.color": "#e1e0d9",
    "figure.dpi": 100,
})


def save_figure(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path
