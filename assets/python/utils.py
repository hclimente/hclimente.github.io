import matplotlib.pyplot as plt


def save_fig(fig, basename):
    fig.savefig(f"img/{basename}.webp", bbox_inches="tight", dpi=300)
    dpi_800 = 800 / fig.get_size_inches()[0]
    fig.savefig(f"img/{basename}-800.webp", bbox_inches="tight", dpi=dpi_800)
    dpi_1600 = 1600 / fig.get_size_inches()[0]
    fig.savefig(f"img/{basename}-1600.webp", bbox_inches="tight", dpi=dpi_1600)
    plt.show()
    plt.close(fig)
