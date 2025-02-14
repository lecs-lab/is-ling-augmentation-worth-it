import math

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

# method_colors = {
#     "Random duplicate,  Insert noise,  TAM update": "#43E0D8",  # light blue
#     "Delete with exclusions,  Random duplicate,  Insert conjunction,  Insert noise,  TAM update": "#E76F51",  # dark orange
#     "Random delete,  Insert conjunction": "#E9C46A",  # yellow
#     "Random delete,  Random duplicate,  Insert conjunction,  TAM update": "#aaaaaa",  # gray
#     "Delete with exclusions,  Random delete,  Insert conjunction,  TAM update": "#000000", # black
#     "Insert conjunction,  TAM update": "#CC7722", #ochre
#     "Delete with exclusions,  Random delete,  Random duplicate,  Insert conjunction": "#254653", #dark blue
#     "Delete with exclusions,  Random delete,  Insert conjunction": "#299D8F",  # teal
#     "Delete with exclusions,  Random duplicate,  Insert conjunction,  Insert noise": "#F4A261",  # light orange
# }
#
plt.rcParams["text.usetex"] = True


method_colors = {
    r"\textsc{Del-Excl + Dup + Ins-Conj}": "#254653", #dark blue
    r"\textsc{Del-Excl + Dup + Ins-Conj + Ins-Noise}": "#299D8F",  # teal
    r"\textsc{Dup + Ins-Conj + Ins-Noise}": "#F4A261",  # light orange
    r"\textsc{Del + Dup + Ins-Conj + Upd-TAM}": "#43E0D8",  # light blue
    r"\textsc{Ins-Conj + Upd-TAM}": "#E76F51",  # dark orange
    r"\textsc{Del-Excl + Del + Ins-Conj + Upd-TAM}": "#E9C46A",  # yellow
    r"\textsc{Del-Excl + Dup + Ins-Conj + Ins-Noise + Upd-TAM}": "#aaaaaa",  # gray
    r"\textsc{Del + Dup + Ins-Conj + Ins-Noise}": "#000000", # black
    r"\textsc{Del-Excl + Del + Ins-Conj}": "#CC7722", #ochre
}


handles = [
    mlines.Line2D(
        [], [], color=color, marker="o", linestyle="None", markersize=8, label=label
    )
    for label, color in method_colors.items()
]
num_columns = math.ceil(len(handles) / 5)
fig_legend, ax_legend = plt.subplots(figsize=(4, 0.5))
legend = ax_legend.legend(
    handles=handles, title="Method", loc="center", ncol=num_columns, frameon=False
)
ax_legend.axis("off")
legend.set_title(None)

fig_legend.canvas.draw()
bbox = legend.get_window_extent().transformed(fig_legend.dpi_scale_trans.inverted())

fig_legend.savefig(
    "legend-combined-usp.pdf",
    format="pdf",
    bbox_inches=bbox,
    pad_inches=0,
)
plt.close(fig_legend)
