import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

ALPHA = 0.5
FONT_SIZE = 12
TICK_SIZE = 10
LINEWIDTH = 2
plt.rcParams.update({"font.family": "Times New Roman", "font.size": FONT_SIZE})


LABELS = {
    "magnitude_pruning": "MP",
    "coin_baseline": "COIN",
    "gated": "L0-onie",
    "jpeg": "JPEG",
}
COLORS = {
    "magnitude_pruning": "firebrick",
    "coin_baseline": "#FF8C42",
    "gated": "royalblue",
    "jpeg": "#afd5aa",
}
ORDERS = {
    "magnitude_pruning": 0,
    "coin_baseline": 1,
    "gated": 2,
    "jpeg": 3,
}
