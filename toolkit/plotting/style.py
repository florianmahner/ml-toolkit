import os
import matplotlib as mpl
import seaborn as sns
from typing import Any


def get_style_path(context: str) -> str:
    return os.path.join(os.path.dirname(__file__), f"{context}.mplstyle")


def set_plotting_style(
    context: str | None = "paper",
    despine: bool = False,
    tight_layout: bool = False,
    **rc_params: dict[str, Any],
):
    """
    Set plotting context and optionally despine the plots.

    Parameters:
    context (str): The name of the context (e.g., 'default', 'paper').
    despine (bool): Whether to remove the top and right spines.
    """

    if context:
        mplstyle_params = mpl.rc_params_from_file(get_style_path(context))

    if tight_layout:
        mplstyle_params["figure.constrained_layout.use"] = True

    if despine:
        mplstyle_params["axes.spines.top"] = False
        mplstyle_params["axes.spines.right"] = False

    mplstyle_params.update(rc_params)

    current_style = sns.axes_style()
    sns.set_theme(
        # context="notebook",
        style=current_style,
        rc=mplstyle_params,
    )
