import matplotlib.pyplot as plt
import plotly


def save_fig(fig, basename):
    fig.savefig(f"img/{basename}.webp", bbox_inches="tight", dpi=300)
    dpi_800 = 800 / fig.get_size_inches()[0]
    fig.savefig(f"img/{basename}-800.webp", bbox_inches="tight", dpi=dpi_800)
    dpi_1600 = 1600 / fig.get_size_inches()[0]
    fig.savefig(f"img/{basename}-1600.webp", bbox_inches="tight", dpi=dpi_1600)
    plt.show()
    plt.close(fig)


PLOTLY_AXIS_ATTR_DICT = {
    "showgrid": False,  # Hide grid lines
    "zeroline": False,  # Hide zero line
    "showline": True,  # Hide axis line
    "ticks": "",  # Hide ticks
    "showticklabels": False,  # Hide tick labels if you want extreme simplicity
    "linecolor": "black",  # Line color
    "linewidth": 1,  # Line width
}

# Position the legend outside the plot area (right and bottom)
PLOTLY_LEGEND_ATTR_DICT = {
    "orientation": "v",  # Vertical legend
    "yanchor": "middle",  # Anchor point is the bottom of the legend box
    "y": 0.5,  # Position the bottom of the legend box at "y": 0 (bottom of plot area)
    "xanchor": "left",  # Anchor point is the left of the legend box
    "x": 1.02,  # Position the left of the legend box just outside the plot area on the right
}


def save_plotly(
    fig,
    basename: str,
    xaxis_attr_dict: dict,
    yaxis_attr_dict: dict,
    legend_attr_dict: dict,
    colorway=plotly.colors.qualitative.T10,
    **kwargs,
):
    fig.update_layout(
        xaxis=xaxis_attr_dict,
        yaxis=yaxis_attr_dict,
        template="plotly_white",  # Clean base
        margin=dict(
            l=50, r=180, t=50, b=50
        ),  # Increase right margin to make space for legend
        legend=legend_attr_dict,
        colorway=colorway,
    )

    fig.show()
    fig.write_html(f"plotly/{basename}.html", include_plotlyjs="cdn", **kwargs)
