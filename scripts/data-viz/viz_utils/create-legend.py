import math

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True


method_colors = {
    r"\textsc{Upd-TAM}": "#E76F51",
    r"\textsc{Del-Excl}": "#299D8F",
    r"\textsc{Ins-Conj, Ins-Intj}": "#43E0D8",
    r"\textsc{Dup}": "#E9C46A",
    r"\textsc{Ins-Noise}": "#254653",
    r"\textsc{Perm}": "#bbbbbb",
    r"\textsc{Del}": "#F4A261",
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
