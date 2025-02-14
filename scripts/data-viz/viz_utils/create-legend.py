import math

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

method_colors = {
    "Insert noise": "#254653", #dark blue 
    "Insert conjunction/interjection": "#43E0D8",  # light blue
    "Delete": "#F4A261",  # light orange
    "Delete (w/ exclusions)": "#299D8F",  # teal
    "Duplicate": "#E9C46A",  # yellow
    "TAM update": "#E76F51",  # dark orange
    "Permute": "#bbbbbb",  # gray
}
handles = [
    mlines.Line2D(
        [], [], color=color, marker="o", linestyle="None", markersize=8, label=label
    )
    for label, color in method_colors.items()
]
num_columns = math.ceil(len(handles) / 2)
fig_legend, ax_legend = plt.subplots(figsize=(8, 0.5))
legend = ax_legend.legend(
    handles=handles, title="Method", loc="center", ncol=num_columns, frameon=False
)
ax_legend.axis("off")
legend.set_title(None)

fig_legend.canvas.draw()
bbox = legend.get_window_extent().transformed(fig_legend.dpi_scale_trans.inverted())

fig_legend.savefig(
    "legend.pdf",
    format="pdf",
    bbox_inches=bbox,
    pad_inches=0,
)
plt.close(fig_legend)
