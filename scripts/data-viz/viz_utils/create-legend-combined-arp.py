import math

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

method_colors = {
    "Insert interjection, Insert noise": "#43E0D8",  # light blue
    "Insert noise": "#E76F51",  # dark orange
    "Insert interjection": "#E9C46A",  # yellow
    "Random delete,  Random duplicate,  Insert conjunction,  TAM update": "#aaaaaa",  # gray
    "Delete with exclusions,  Random delete,  Insert conjunction,  TAM update": "#000000", # black
    "Insert conjunction,  TAM update": "#CC7722", #ochre
    "Delete with exclusions,  Random delete,  Random duplicate,  Insert conjunction": "#254653", #dark blue 
    "Delete with exclusions,  Random delete,  Insert conjunction": "#299D8F",  # teal
    "Delete with exclusions,  Random duplicate,  Insert conjunction,  Insert noise": "#F4A261",  # light orange
}

handles = [
    mlines.Line2D(
        [], [], color=color, marker="o", linestyle="None", markersize=8, label=label
    )
    for label, color in method_colors.items()
]
num_columns = math.ceil(len(handles) / 3)
fig_legend, ax_legend = plt.subplots(figsize=(4, 0.5))
legend = ax_legend.legend(
    handles=handles, title="Method", loc="center", ncol=num_columns, frameon=False
)
ax_legend.axis("off")
legend.set_title(None)

fig_legend.canvas.draw()
bbox = legend.get_window_extent().transformed(fig_legend.dpi_scale_trans.inverted())

fig_legend.savefig(
    "legend-combined.pdf",
    format="pdf",
    bbox_inches=bbox,
    pad_inches=0,
)
plt.close(fig_legend)
