method_colors = {
    "Ins-Noise": "#254653",  # blue
    "Del-Excl": "#F4A261",  # teal
    "Del": "#F4A261",  # light orange
    "Ins-Conj": "#254653",  # light blue
    "Upd-TAM": "#E76F51",  # dark orange
    "Dup": "#E9C46A",  # yellow
    "Ins-Intj": "#254653",  # light blue
    "Perm": "#bbbbbb",  # gray
}

linguistic_strategies = ["Ins-Conj", "Ins-Intj", "Del-Excl", "Upd-TAM", "Perm"]
non_linguistic_strategies = ["Ins-Noise", "Del", "Dup"]

method_dashes = {
    **{method: "" for method in linguistic_strategies},
    **{method: (2, 2) for method in non_linguistic_strategies},
}
