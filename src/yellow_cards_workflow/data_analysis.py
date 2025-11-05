from typing import Any, Dict, List, Literal, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from scipy.stats import linregress

from yellow_cards_workflow import (
    BASE_DIR,
    DEFAULT_DATA_OFFSET,
    DEFAULT_DATA_OFFSETS,
    DEFAULT_THRESHOLD,
    DEFAULT_THRESHOLDS,
    MODEL_NAMES,
)
from yellow_cards_workflow.calculations import (
    calc_classification_metrics,
    calc_basic_stats,
    calc_groups,
    calc_models_diffs,
    get_modified_entries,
    calc_values_diffs,
    calc_diffs_corrs,
)

colorblind = sns.color_palette("colorblind")
new_order = [colorblind[i] for i in [0, 3, 7, 2, 5, 4, 6, 8, 1, 9]]
sns.palettes.SEABORN_PALETTES["colorblind_reordered"] = new_order
PALETTE = "colorblind_reordered"
HATCH_COEFF = 5
HATCHES = [
    "/" * HATCH_COEFF,
    "\\" * HATCH_COEFF,
    "x" * HATCH_COEFF,
    "o" * HATCH_COEFF,
    "O" * HATCH_COEFF,
    "." * HATCH_COEFF,
    "*" * HATCH_COEFF,
    "|" * HATCH_COEFF,
    "-" * HATCH_COEFF,
    "+" * HATCH_COEFF,
]
HATCH_DICT = dict(zip(new_order, HATCHES))


def plot_distribution_deltas(
    dataframe: pd.DataFrame,
    ref_data: pd.Series | NDArray[np.float64] | pd.DataFrame,
    value_name: str = "values",
    col: str = "modified_%",
    row: str = "var_add",
    height: float = 3,
    aspect: float = 1,
    margin_titles: bool = False,
    **kwargs,
) -> sns.FacetGrid:
    """
    Plot distribution deltas using histogram visualization.

    This function calculates differences between values in a dataframe and reference data,
    then creates a histogram plot to visualize the distribution of these differences.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The main dataframe containing the data to analyze.
    ref_data : pd.Series, np.ndarray, or pd.DataFrame
        Reference data for calculating differences.
    value_name : str, default "values"
        Name of the column containing the values to plot.
    col : str, default "modified_%"
        Column name for faceting the plot horizontally.
    row : str, default "var_add"
        Column name for faceting the plot vertically.
    height : float, default 3.0
        Height of each facet in the plot.
    aspect : float, default 1.0
        Aspect ratio of each facet.
    margin_titles : bool, default False
        Whether to draw titles on the marginal axes.
    **kwargs
        Additional keyword arguments passed to the plotting function.

    Returns
    -------
    sns.FacetGrid
        The matplotlib figure containing the histogram plot.
    """
    plot_df = calc_values_diffs(
        dataframe=dataframe,
        ref_data=ref_data,
        value_name=value_name,
    )
    plot_df = plot_df.reset_index()
    plot_df = plot_df[plot_df.loc[:, value_name].abs() > 1e-8]
    fg = sns.displot(
        data=plot_df,
        x=value_name,
        col=col if col in plot_df.columns else None,
        row=row if row in plot_df.columns else None,
        kind="hist",
        height=height,
        aspect=aspect,
        edgecolor=None,
        lw=0,
        facet_kws={"sharey": False, "margin_titles": margin_titles},
    )
    return fg


# NOT USED FOR ARTICLE
def plot_binned_values(
    dataframe: pd.DataFrame,
    bins: NDArray[np.float64],
    model_names: str | List[str] = MODEL_NAMES,
    title: str = "",
    value_name: str = "values",
):
    plot_df = dataframe[[value_name]].copy()
    plot_df["bin_#"] = np.digitize(dataframe[value_name], bins)
    plot_df["bin_val"] = [
        (
            round(bins[x], 3)
            if 0 < x < len(bins)
            else (
                round(min(bins), 3)
                if x == 0
                else round(max(bins), 3) if x == len(bins) else None
            )
        )
        for x in plot_df["bin_#"]
    ]
    plot_df = pd.concat(
        [
            plot_df,
            calc_basic_stats(dataframe, model_names=model_names, value_name=value_name),
        ],
        axis=1,
    )
    fg = sns.catplot(
        data=plot_df,
        x="bin_val",
        y="d_median_a",
        kind="boxen",
        height=4,
        aspect=1.75,
    )
    fg.ax.set_ylim(-1e-8, plot_df["d_median_a"].median() * 5)
    regress_line = (
        (plot_df.groupby("bin_val").size() / len(plot_df)).reset_index().to_numpy()
    )
    twinx = fg.ax.twinx()
    twinx.set_ylabel("density")
    twinx.plot(regress_line[:, 1], lw=2, ls="--", c="orange")
    fg.ax.set_xticklabels(fg.ax.get_xticklabels(), rotation=-90)
    fg.ax.set_xticklabels(fg.ax.get_xticklabels(), rotation=-90)
    reg_result = linregress(
        plot_df[value_name],
        plot_df["d_median_a"],
    )
    binned_data = (
        plot_df.groupby("bin_val", as_index=False)["d_median_a"].median().to_numpy()
    )
    binned_reg = linregress(
        binned_data[1:-1, 0],
        binned_data[1:-1, 1],
    )
    fg.figure.text(
        0.5,
        1.0,
        f"Non-binned | Medians\n{reg_result.slope:.4f}*x + {binned_reg.intercept:.4f} | {binned_reg.slope:.4f}*x + {binned_reg.intercept:.4f}\nR^2={reg_result.rvalue**2:.3e} | R^2={binned_reg.rvalue**2:.3e} \n p={reg_result.pvalue:.3e} | p={binned_reg.pvalue:.3e}"
        "",
        ha="center",
    )
    fg.figure.suptitle(title)
    fg.tight_layout()
    return fg


def plot_delta_corr(
    dataframe: pd.DataFrame,
    ref_data: Optional[
        List[float] | pd.Series | NDArray[np.float64] | pd.DataFrame
    ] = None,
    model_names: str | List[str] = MODEL_NAMES,
    data_offset: float = DEFAULT_DATA_OFFSET,
    value_name: str = "values",
    **kwargs,
):
    """
    Plot correlation heatmaps for delta correlations across different model variations.

    This function calculates correlations between model outputs and visualizes them
    using heatmap plots. It supports both single and multi-level index structures
    in the correlation data, generating appropriate visualizations for each case.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataframe containing model data for correlation calculation.
    ref_data : array-like, optional
        Reference data for correlation calculation. If None, uses default reference.
    model_names : str or list of str, default MODEL_NAMES
        Name(s) of models to include in analysis. Can be a single string or list.
    data_offset : float, default DEFAULT_DATA_OFFSET
        Offset value to apply to data during correlation calculation.
    value_name : str, default "values"
        Name of the column containing values to analyze in the dataframe.
    **kwargs
        Additional keyword arguments passed to underlying plotting functions.

    Returns
    -------
    tuple
        Figure objects and axes for the generated plots. Return type varies based on
        the structure of the correlation data:
        - For single-level index: (fig, fig2, axs, ax2)
        - For two-level index: (fig, fig2, axs, ax2)
        - For single plot: (fig, axs)
    """
    correlations = calc_diffs_corrs(
        dataframe=dataframe,
        ref_data=ref_data,
        model_names=model_names,
        data_offset=data_offset,
        value_name=value_name,
    )
    idx_names = ["var_add", "modified_%"]
    idx_names = [x for x in idx_names if x in correlations.index.names]
    if len(idx_names) == 1:
        xs = correlations.index.get_level_values(idx_names[0]).unique()
        fig, axs = plt.subplots(
            ncols=len(xs), nrows=1, figsize=(len(xs) * 5, 4), dpi=100
        )
        arr_correlations = []
        for ix, x in enumerate(xs):
            correlation = correlations.loc[x, :]
            arr_correlations.append(correlation.to_numpy())
            sns.heatmap(
                data=correlations.loc[x, :],
                annot=True,
                cmap="vlag",
                vmax=1,
                vmin=-1,
                ax=axs[ix],
            )
        arr_correlations = np.stack(arr_correlations)
        arr_min = arr_correlations.min(axis=0).round(2)
        arr_max = arr_correlations.max(axis=0).round(2)
        # annots = np.char.add(
        #     np.char.add(arr_max.astype(str), "\n"), arr_min.astype(str)
        # )
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
        sns.heatmap(
            data=pd.DataFrame(
                arr_min,
                index=correlation.index,
                columns=correlation.columns,
            ),
            # annot=annots,
            annot=True,
            mask=np.triu(np.ones_like(arr_min, dtype=bool), k=1),
            cmap="vlag",
            vmax=1,
            vmin=-1,
            ax=ax2,
            # fmt="s",
        )
        sns.heatmap(
            data=pd.DataFrame(
                arr_max,
                index=correlation.index,
                columns=correlation.columns,
            ),
            # annot=annots,
            annot=True,
            mask=np.tril(np.ones_like(arr_min, dtype=bool), k=-1),
            cmap="vlag",
            vmax=1,
            vmin=-1,
            ax=ax2,
            cbar=False,
            # fmt="s",
        )
        return fig, fig2, axs, ax2
    elif len(idx_names) == 2:
        xs = correlations.index.get_level_values(idx_names[0]).unique()
        ys = correlations.index.get_level_values(idx_names[1]).unique()

        fig, axs = plt.subplots(
            ncols=len(xs), nrows=len(ys), figsize=(len(xs) * 5, len(ys) * 4), dpi=100
        )
        arr_correlations = []
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                correlation = correlations.loc[(y, x), :]
                arr_correlations.append(correlation.to_numpy())
                sns.heatmap(
                    data=correlation,
                    annot=True,
                    cmap="vlag",
                    vmax=1,
                    vmin=-1,
                    ax=axs[iy, ix],
                )
        arr_correlations = np.stack(arr_correlations)
        arr_min = arr_correlations.min(axis=0).round(2)
        arr_max = arr_correlations.max(axis=0).round(2)
        # annots = np.char.add(
        #     np.char.add(arr_max.astype(str), "\n"), arr_min.astype(str)
        # )
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
        sns.heatmap(
            data=pd.DataFrame(
                arr_min,
                index=correlation.index,
                columns=correlation.columns,
            ),
            # annot=annots,
            annot=True,
            mask=np.triu(np.ones_like(arr_min, dtype=bool), k=1),
            cmap="vlag",
            vmax=1,
            vmin=-1,
            ax=ax2,
            # fmt="s",
        )
        sns.heatmap(
            data=pd.DataFrame(
                arr_max,
                index=correlation.index,
                columns=correlation.columns,
            ),
            # annot=annots,
            annot=True,
            mask=np.tril(np.ones_like(arr_min, dtype=bool), k=-1),
            cmap="vlag",
            vmax=1,
            vmin=-1,
            ax=ax2,
            cbar=False,
            # fmt="s",
        )
        return fig, fig2, axs, ax2
    else:
        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(5, 4), dpi=100)
        sns.heatmap(correlations, annot=True, cmap="vlag", vmax=1, vmin=-1, ax=axs)
        return fig, axs


def plot_delta_correlations(
    dataframe: pd.DataFrame,
    ref_data: Optional[
        List[float] | pd.Series | NDArray[np.float64] | pd.DataFrame
    ] = None,
    model_names: str | List[str] = MODEL_NAMES,
    data_offset: float = DEFAULT_DATA_OFFSET,
    value_name: str = "values",
    x="modified_%",
    hue=None,
    col="var_add",
    s: float = 5,
    height: float = 2,
    aspect: float = 1,
    col_wrap: None | int = None,
    **kwargs,
) -> sns.FacetGrid:
    """
    Plot prediction differences correlations across the models.

    This function calculates prediction differences correlations between models,
    then visualizes the results using swarm plots with overlaid violin plots.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe containing model data
    ref_data : array-like, optional
        Reference data for correlation calculations
    model_names : str or list of str, default MODEL_NAMES
        Names of models to include in analysis
    data_offset : float, default DEFAULT_DATA_OFFSET
        Offset value for data processing
    value_name : str, default "values"
        Name of the value column in the melted dataframe
    x : str, default "modified_%"
        Column name to use for x-axis
    hue : str, optional
        Column name to use for hue encoding
    col : str, default "var_add"
        Column name to use for facet grid column grouping
    s : float, default 5
        Size of swarm plot points
    height : float, default 2
        Height of each facet in the grid
    aspect : float, default 1
        Aspect ratio of each facet
    col_wrap : int or None, default None
        Number of columns in facet grid
    **kwargs
        Additional keyword arguments passed to underlying plotting functions

    Returns
    -------
    sns.FacetGrid
        The matplotlib figure with the plotted data
    """
    plot_df = calc_diffs_corrs(
        dataframe,
        ref_data=ref_data,
        model_names=model_names,
        data_offset=data_offset,
        value_name=value_name,
    )

    plot_df.columns.name = "model_name_2"
    plot_df = plot_df.melt(ignore_index=False).reset_index(
        # level=-1
    )
    plot_df = plot_df[plot_df["model_name"] != plot_df["model_name_2"]]
    plot_df["pair"] = plot_df[["model_name", "model_name_2"]].apply(
        lambda x: "; ".join(sorted(x)), axis=1
    )
    plot_df = plot_df.sort_values("pair")
    id_vars = ["var_add", "modified_%", "pair"]
    id_vars = [x for x in id_vars if x in plot_df.columns]
    plot_df = plot_df.drop_duplicates(id_vars)

    fg = sns.FacetGrid(
        plot_df,
        col=col if col in plot_df.columns else None,
        # hue="pair",
        # palette=PALETTE,
        height=height,
        aspect=aspect,
        col_wrap=col_wrap,
    )
    fg.map_dataframe(sns.swarmplot, x=x, y="value", hue="pair", palette=PALETTE, s=s)
    fg.map_dataframe(
        sns.violinplot,
        x=x,
        y="value",
        hue=None,
        inner=None,
        fill=False,
        density_norm="width",
    )
    fg.add_legend()
    fg.tight_layout()
    return fg


STATS_MODELS_VARS = Literal[
    "modified_%",
    "var_add",
    "delta_type",
    "agg",
    "model_name",
    "ref_type",
    None,
]


# TODO: needs a bit more polish
def plot_stats_models(
    dataframe: pd.DataFrame,
    ref_data: Optional[pd.Series | NDArray[np.float64] | pd.DataFrame] = None,
    original_data: Optional[pd.Series | NDArray[np.float64] | pd.DataFrame] = None,
    model_names: str | List[str] = MODEL_NAMES,
    data_offset: float = DEFAULT_DATA_OFFSET,
    value_name: str = "values",
    agg: List[str] | str = ["mean", "median"],
    x_var: STATS_MODELS_VARS = "modified_%",
    hue: STATS_MODELS_VARS = "model_name",
    style: STATS_MODELS_VARS = "agg",
    col: STATS_MODELS_VARS = "var_add",
    row: STATS_MODELS_VARS = "delta_type",
    plot_type: Optional[Literal["catplot", "relplot"]] = "relplot",
    delta_type: Optional[Literal["delta_abs", "delta_rel"]] = None,
    facet_kws: Optional[Dict] = None,
    split_modified: bool = False,
    ignore_mod_df: bool = False,
    height: int = 3,
    aspect: float = 1.0,
    **kwargs,
) -> sns.FacetGrid:
    """
    Plot models stats using seaborn visualization tools.

    This function generates plots comparing model performance metrics across
    different configurations and data sets, supporting both relational and
    categorical plot types.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame containing model data to visualize.
    ref_data : Optional[pd.Series | NDArray[np.float64] | pd.DataFrame], optional
        Reference data for comparison, by default None
    original_data : Optional[pd.Series | NDArray[np.float64] | pd.DataFrame], optional
        Original data for baseline comparison, by default None
    model_names : str | List[str], optional
        Names of models to include in analysis, by default MODEL_NAMES
    data_offset : float, optional
        Offset value for data processing, by default DEFAULT_DATA_OFFSET
    value_name : str, optional
        Name of the value column in the data, by default "values"
    agg : List[str] | str, optional
        Aggregation methods for grouping data, by default ["mean", "median"]
    x_var : STATS_MODELS_VARS, optional
        Variable to use for x-axis, by default "modified_%"
    hue : STATS_MODELS_VARS, optional
        Variable to map to color hue, by default "model_name"
    style : STATS_MODELS_VARS, optional
        Variable to map to marker style, by default "agg"
    col : STATS_MODELS_VARS, optional
        Variable to map to columns in facets, by default "var_add"
    row : STATS_MODELS_VARS, optional
        Variable to map to rows in facets, by default "delta_type"
    plot_type : Optional[Literal["catplot", "relplot"]], optional
        Type of plot to generate, by default "relplot"
    delta_type : Optional[Literal["delta_abs", "delta_rel"]], optional
        Specific delta type to filter results, by default None
    facet_kws : Optional[Dict], optional
        Additional keyword arguments for facets, by default None
    split_modified : bool, optional
        Whether to split modified entries, by default False
    ignore_mod_df : bool, optional
        Whether to ignore modified data frame, by default False
    height : int, optional
        Height of each facet, by default 3
    aspect : float, optional
        Aspect ratio of each facet, by default 1.0
    **kwargs
        Additional keyword arguments passed to seaborn plotting functions.

    Returns
    -------
    sns.FacetGrid
        The generated seaborn FacetGrid object containing the plot.
    """
    if isinstance(agg, str):
        agg = [agg]
    if ref_data is None and original_data is None:
        plot_df = calc_models_diffs(
            dataframe=dataframe,
            model_names=model_names,
            ref_data=ref_data,
            value_name=value_name,
            data_offset=data_offset,
        )
    elif ref_data is not None:
        plot_data = {}
        if not ignore_mod_df:
            mod_df = calc_models_diffs(
                dataframe=dataframe,
                model_names=model_names,
                value_name=value_name,
                data_offset=data_offset,
            )
            mod_df["modified"] = get_modified_entries(dataframe, ref_data, value_name)
            plot_data["mod"] = mod_df

        noisy_df = calc_models_diffs(
            dataframe=dataframe,
            model_names=model_names,
            ref_data=ref_data,
            value_name=value_name,
            data_offset=data_offset,
        )
        noisy_df["modified"] = get_modified_entries(dataframe, ref_data, value_name)
        plot_data["noisy"] = noisy_df

        if original_data is not None:
            base_df = calc_models_diffs(
                dataframe=dataframe,
                model_names=model_names,
                ref_data=original_data,
                value_name=value_name,
                data_offset=data_offset,
            )
            base_df["modified"] = get_modified_entries(dataframe, ref_data, value_name)
            plot_data["base"] = base_df
        plot_df = pd.concat(plot_data, names=["ref_type"])

    plot_df = plot_df.reset_index().drop(columns=["#", "delta"])
    id_vars = ["var_add", "modified_%", "model_name", "ref_type"]
    if split_modified:
        id_vars.append("modified")
    id_vars = [x for x in id_vars if x in plot_df.columns]
    if plot_type == "relplot":
        plot_df = (
            plot_df.groupby(by=id_vars)
            .agg(agg)
            .stack(0, future_stack=True)
            .reset_index(names=id_vars + ["delta_type"])
            .melt(id_vars=id_vars + ["delta_type"], var_name="agg")
        )
        # return plot_df
        if delta_type is not None:
            plot_df = plot_df[plot_df["delta_type"] == delta_type]
        fg = sns.relplot(
            data=plot_df,
            x=x_var,
            y="value",
            hue=hue,
            style=style,
            col=col if col in plot_df.columns else None,
            row=row if row in plot_df.columns else None,
            kind="line",
            markers=True,
            facet_kws={"sharey": "row"} if facet_kws is None else facet_kws,
            height=height,
            aspect=aspect,
            palette=PALETTE,
        )
    else:
        plot_df = plot_df.melt(id_vars=id_vars, var_name="delta_type")
        fg = sns.catplot(
            data=plot_df,
            x=x_var,
            y="value",
            hue=hue,
            col=col if col in plot_df.columns else None,
            row=row if row in plot_df.columns else None,
            kind="boxen",
            showfliers=False,
            height=3,
            palette=PALETTE,
            facet_kws=facet_kws,
        )
    return fg


def plot_group_sizes(
    dataframe: pd.DataFrame,
    model_names: str | List[str] = MODEL_NAMES,
    ref_data: Optional[
        List[float] | pd.Series | NDArray[np.float64] | pd.DataFrame
    ] = None,
    data_offset: float = DEFAULT_DATA_OFFSET,
    thresholds: float | List[float] | NDArray[np.float64] = DEFAULT_THRESHOLD,
    groups: int | List[int] | NDArray[np.int_] = np.arange(0, len(MODEL_NAMES) + 1),
    group_stat: Literal["abs", "rel", "both"] = "abs",
    value_name: str = "values",
    hue: str = "modified_%",
    col: str = "var_add",
    row: str = "modified",
    height: float = 4,
    aspect: float = 1.2,
    add_catplot: bool = True,
    plot_type: Optional[Literal["relplot", "barplot", "histplot"]] = None,
    margin_titles: bool = False,
    **kwargs,
) -> sns.FacetGrid:
    """
    Plot group sizes based on calculated group statistics.

    This function visualizes the distribution of data points across different groups
    derived from model thresholds and data offsets. It supports multiple plot types
    including relational, bar, and histogram plots.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame containing the data to analyze.
    model_names : str | List[str], default MODEL_NAMES
        Name or list of names of models to consider.
    ref_data : Optional[List[float] | pd.Series | NDArray[np.float64] | pd.DataFrame], default None
        Reference data for comparison. If provided, ground truth is calculated.
    data_offset : float, default DEFAULT_DATA_OFFSET
        Offset value for data processing.
    thresholds : float | List[float] | NDArray[np.float64], default DEFAULT_THRESHOLD
        Threshold values for grouping.
    groups : int | List[int] | NDArray[np.int_], default np.arange(0, len(MODEL_NAMES) + 1)
        Group identifiers to include in the plot.
    group_stat : Literal["abs", "rel", "both"], default "abs"
        Type of group statistics to use: absolute, relative, or both.
    value_name : str, default "values"
        Name of the column containing values to analyze.
    hue : str, default "modified_%"
        Column name for hue encoding in the plot.
    col : str, default "var_add"
        Column name for column facetting.
    row : str, default "modified"
        Column name for row facetting.
    height : float, default 4
        Height of each facet in the plot.
    aspect : float, default 1.2
        Aspect ratio of each facet.
    add_catplot : bool, default True
        Whether to add a categorical plot overlay for non-relplot types.
    plot_type : Optional[Literal["relplot", "barplot", "histplot"]], default None
        Type of plot to generate. If None, defaults to "relplot".
    margin_titles : bool, default False
        Whether to add titles to margin areas of facets.
    **kwargs
        Additional keyword arguments passed to the plotting functions.

    Returns
    -------
    sns.FacetGrid
        The resulting plot grid object.
    """
    stat_cols = {
        "abs": f"s_{len(model_names)}_a",
        "rel": f"s_{len(model_names)}_r",
        "both": f"s_{len(model_names)}_b",
    }
    group = stat_cols.get(group_stat, f"s_{len(model_names)}_a")
    group_stats = calc_groups(
        dataframe=dataframe,
        model_names=model_names,
        thresholds=thresholds,
        data_offsets=data_offset,
        value_name=value_name,
    )
    if ref_data is not None:
        ground_truth = get_modified_entries(
            data=dataframe, ref_data=ref_data, value_name=value_name
        )
        plot_df = group_stats.join(ground_truth)
    else:
        plot_df = group_stats
    plot_df = plot_df.reset_index()
    plot_df = plot_df[np.isin(plot_df[group], groups)]
    id_vars = ["var_add", "modified_%", "modified"]
    id_vars = [x for x in id_vars if x in plot_df.columns]
    id_vars.append(group)

    if plot_type is None or plot_type == "relplot":
        plot_df[group] = plot_df[group].astype(str)

        relplot_df = (
            plot_df.groupby(id_vars)
            .agg(
                "size",
            )
            .to_frame("size")
        ).reset_index()
        fg = sns.relplot(
            data=relplot_df,
            x=group,
            y="size",
            hue=hue if hue in relplot_df.columns else None,
            style="modified" if "modified" in relplot_df.columns else None,
            col=col if col in relplot_df.columns else None,
            row=row if row in relplot_df.columns else None,
            markers=True,
            palette=PALETTE,
            kind="line",
            height=height,
            aspect=aspect,
        )
        if col not in plot_df.columns and row not in plot_df.columns and add_catplot:
            barplot_df = (
                plot_df.drop(columns="modified") if "modified" in plot_df else plot_df
            )
            bar_id_vars = [x for x in id_vars if x in barplot_df.columns]
            barplot_df = (
                barplot_df.groupby(by=bar_id_vars).size().to_frame("size").reset_index()
            )
            sns.barplot(
                data=barplot_df,
                x=group,
                y="size",
                alpha=0.3,
                err_kws=dict(alpha=0.3),
                ax=fg.ax,
                legend=False,
            )
    elif plot_type == "barplot":
        plot_df[group] = plot_df[group].astype(str)

        relplot_df = (
            plot_df.groupby(id_vars)
            .agg(
                "size",
            )
            .to_frame("size")
        ).reset_index()
        # relplot_df["size"] = np.sqrt(relplot_df["size"] / (relplot_df["size"].max() - relplot_df["size"].min()))
        fg = sns.catplot(
            data=relplot_df,
            x=group,
            y="size",
            hue=hue if hue in relplot_df.columns else None,
            col=col if col in relplot_df.columns else None,
            row=row if row in relplot_df.columns else None,
            palette=PALETTE if hue is not None else None,
            color=sns.color_palette(PALETTE, 1)[0] if hue is None else None,
            kind="bar",
            height=height,
            aspect=aspect,
            fill=False,
        )
    elif plot_type == "histplot":
        if groups is not None:
            bins = groups
        else:
            bins = np.array(sorted(plot_df[group].unique().astype(int)))
        # bins = bins[bins>0]
        fg = sns.displot(
            data=plot_df,
            x=group,
            bins=bins,
            hue=hue if hue in plot_df.columns else None,
            col=col if col in plot_df.columns else None,
            row=row if row in plot_df.columns else None,
            palette=PALETTE if hue is not None else None,
            kind="hist",
            fill=False,
            shrink=0.8,
            element="bars",
            multiple="stack",
            discrete=True,
            height=height,
            aspect=aspect,
            facet_kws={"margin_titles": margin_titles},
        )
        for ax in fg.axes.flatten():
            for patch in ax.patches:
                patch.set_hatch(HATCH_DICT[patch.get_edgecolor()[:-1]])
                patch.set_hatch_linewidth(0.5)
            ax.set_xticks(bins)

        handles = fg.legend.legend_handles
        for handle in handles:
            handle.set_hatch(HATCH_DICT[handle.get_edgecolor()[:-1]])
            handle.set_hatch_linewidth(0.5)
        if row in plot_df.columns:
            fg.set_titles(template="{col_var} = {col_name}\n{row_var} = {row_name}")
    fg.tight_layout()
    return fg


def add_size_annotations(
    fg: sns.FacetGrid,
    plot_stats: pd.DataFrame,
    col: Optional[str],
    row: Optional[str],
    split_modified: bool | str,
):
    """
    Add size annotations to the plot showing group counts.

    Parameters
    ----------
    fg : sns.FacetGrid
        The FacetGrid object to add annotations to.
    plot_stats : pd.DataFrame
        DataFrame containing statistics for plotting.
    col : Optional[str]
        Column name to group by for columns in the grid.
    row : Optional[str]
        Column name to group by for rows in the grid.
    split_modified : bool | str
        If True, split by both 'group' and 'modified' columns. If False or string,
        only split by 'group' column.

    Returns
    -------
    None
        This function modifies the FacetGrid in-place by adding text annotations.
    """

    cols_unique = plot_stats[col].unique() if col in plot_stats.columns else None
    rows_unique = plot_stats[row].unique() if row in plot_stats.columns else None
    if cols_unique is not None or rows_unique is not None:
        x = 0.5
        ha = "center"
        alpha = 0.5
    else:
        x = -0.3
        ha = "right"
        alpha = 0.75
    y = 0.99
    va = "top"
    if split_modified is True:
        count_vars = ["group", "modified"]
    else:
        count_vars = ["group"]

    if cols_unique is not None and rows_unique is not None:
        for r, row_name in enumerate(rows_unique):
            for c, col_name in enumerate(cols_unique):
                stat_text = (
                    plot_stats.loc[
                        (plot_stats[row] == row_name) & (plot_stats[col] == col_name),
                        count_vars,
                    ]
                    .value_counts(sort=False)
                    .sort_index()
                    .to_string()
                )
                if stat_text:
                    fg.axes[r, c].text(
                        s=stat_text,
                        x=x,
                        y=y,
                        transform=fg.axes[r, c].transAxes,
                        ha=ha,
                        va=va,
                        alpha=alpha,
                    )
                    fg.axes[r, c].set_title(
                        f"var_add = {row_name} | modified_% = {col_name}"
                    )
    elif cols_unique is not None:
        for c, col_name in enumerate(cols_unique):
            text = (
                plot_stats.loc[plot_stats[col] == col_name, count_vars]
                .value_counts(sort=False)
                .sort_index()
                .to_string()
            )
            fg.axes[0][c].text(
                s=text,
                x=x,
                y=y,
                transform=fg.axes[0][c].transAxes,
                ha=ha,
                va=va,
                alpha=alpha,
            )
    elif rows_unique is not None:
        for r, row_name in enumerate(rows_unique):
            text = (
                plot_stats.loc[plot_stats[row] == row_name, count_vars]
                .value_counts(sort=False)
                .sort_index()
                .to_string()
            )
            fg.axes[0][r].text(
                s=text,
                x=x,
                y=y,
                transform=fg.axes[0][r].transAxes,
                ha=ha,
                va=va,
                alpha=alpha,
            )
    else:
        text = plot_stats[count_vars].value_counts(sort=False).sort_index().to_string()
        fg.ax.text(
            s=text,
            x=x,
            y=y,
            transform=fg.ax.transAxes,
            ha=ha,
            va=va,
            alpha=alpha,
        )


def add_titles(
    fg: sns.FacetGrid,
    plot_stats: pd.DataFrame,
    col: Optional[str],
    row: Optional[str],
):
    """
    Add size annotations to the plot showing group counts.

    Parameters
    ----------
    fg : sns.FacetGrid
        The FacetGrid object to add titles to
    plot_stats : pd.DataFrame
        DataFrame containing statistics for plotting
    col : str, optional
        Column name to use for grouping in columns
    row : str, optional
        Column name to use for grouping in rows

    Returns
    -------
    None
        Modifies the FacetGrid object in-place
    """

    cols_unique = plot_stats[col].unique() if col in plot_stats.columns else None
    rows_unique = plot_stats[row].unique() if row in plot_stats.columns else None

    if cols_unique is not None and rows_unique is not None:
        for r, row_name in enumerate(rows_unique):
            for c, col_name in enumerate(cols_unique):
                fg.axes[r, c].set_title(f"{row} = {row_name} | {col} = {col_name}")
    elif cols_unique is not None:
        for c, col_name in enumerate(cols_unique):
            fg.axes[0, c].set_title(f" {col} = {col_name}")
    elif rows_unique is not None:
        for r, row_name in enumerate(rows_unique):
            fg.axes[r, 0].set_title(f"{row} = {row_name}")
    else:
        pass


def plot_stats_group(
    dataframe: pd.DataFrame,
    model_names: str | List[str] = MODEL_NAMES,
    ref_data: Optional[
        List[float] | pd.Series | NDArray[np.float64] | pd.DataFrame
    ] = None,
    mod_df_reference: bool = False,
    split_modified: bool | Literal["last"] = False,
    data_offset: float = DEFAULT_DATA_OFFSET,
    thresholds: float | List[float] | NDArray[np.float64] = DEFAULT_THRESHOLD,
    groups: int | List[int] | NDArray[np.int_] = [0, 5],
    group_stat: Literal["abs", "rel", "both"] = "abs",
    stats: (
        Literal["p_mean", "p_median", "p_std", "d_median_a", "d_mean_a", MODEL_NAMES]
        | List
        | Tuple
    ) = ("d_mean_a", "d_median_a"),
    kind: Literal["catplot", "relplot", "histplot", "violinplot"] = "catplot",
    relplot_agg: bool = False,
    x_var: str = "group",
    y_var: str = "value",
    hue: str = "group",
    col: Optional[str] = "modified_%",
    row: Optional[str] = "var_add",
    annot_sizes: bool = True,
    facet_kws: Optional[Dict[Any, Any]] = None,
    value_name: str = "values",
    margin_titles: bool = False,
    height: float = 3,
    aspect: float = 1,
    dodge: bool = True,
    **kwargs,
) -> sns.FacetGrid:
    """
    Plot statistics across the groups.

    This function generates various types of plots (e.g., boxplots, histograms,
    relational plots) based on statistical calculations of model differences and
    groupings. It supports both absolute and relative statistics, and can handle
    modified/unmodified data subsets.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input data containing model and statistical information.
    model_names : str or List[str], optional
        Names of models to analyze. Defaults to MODEL_NAMES.
    ref_data : array-like, optional
        Reference data for calculating differences. If None, uses default.
    mod_df_reference : bool, default False
        If True, uses mod_df as reference data.
    split_modified : bool or Literal["last"], default False
        Whether to split data by modification status.
    data_offset : float, default DEFAULT_DATA_OFFSET
        Offset value for data adjustment.
    thresholds : float or List[float] or NDArray[np.float64], default DEFAULT_THRESHOLD
        Threshold values for grouping.
    groups : int or List[int] or NDArray[np.int_], default [0, 5]
        Group identifiers to filter data.
    group_stat : Literal["abs", "rel", "both"], default "abs"
        Type of statistics to use for grouping.
    stats : str or List[str], default ("d_mean_a", "d_median_a")
        Statistical measures to plot.
    kind : Literal["catplot", "relplot", "histplot", "violinplot"], default "catplot"
        Type of plot to generate.
    relplot_agg : bool, default False
        Whether to aggregate data for relational plots.
    x_var : str, default "group"
        Variable to plot on x-axis.
    y_var : str, default "value"
        Variable to plot on y-axis.
    hue : str, default "group"
        Variable for color encoding.
    col : str, optional
        Column to facet by.
    row : str, optional
        Row to facet by.
    annot_sizes : bool, default True
        Whether to add size annotations.
    facet_kws : dict, optional
        Additional keyword arguments for faceting.
    value_name : str, default "values"
        Name for the value column in melted data.
    margin_titles : bool, default False
        Whether to show margin titles in facets.
    height : float, default 3
        Height of each facet.
    aspect : float, default 1
        Aspect ratio of each facet.
    dodge : bool, default True
        Whether to dodge bars in bar plots.
    **kwargs
        Additional keyword arguments passed to underlying plotting functions.

    Returns
    -------
    sns.FacetGrid
        The generated plot as a seaborn FacetGrid object.
    """
    stat_cols = {
        "abs": f"s_{len(model_names)}_a",
        "rel": f"s_{len(model_names)}_r",
        "both": f"s_{len(model_names)}_b",
    }
    group = stat_cols.get(group_stat, f"s_{len(model_names)}_a")
    basic_stats = calc_basic_stats(
        dataframe=dataframe,
        model_names=model_names,
        data_offset=data_offset,
        ref_data=ref_data if not mod_df_reference else None,
        value_name=value_name,
    )
    group_stats = calc_groups(
        dataframe=dataframe,
        model_names=model_names,
        thresholds=thresholds,
        data_offsets=data_offset,
        value_name=value_name,
    )
    models_stats = calc_models_diffs(
        dataframe=dataframe,
        ref_data=ref_data if not mod_df_reference else None,
        model_names=model_names,
        data_offset=data_offset,
        value_name=value_name,
    )
    models_stats = models_stats.pivot(columns="model_name", values="delta_abs")
    plot_stats = group_stats.join(basic_stats).join(models_stats)
    if ref_data is not None:
        ground_truth = get_modified_entries(
            data=dataframe, ref_data=ref_data, value_name=value_name
        )
        plot_stats = plot_stats.join(ground_truth)
    plot_stats = plot_stats.reset_index()
    plot_stats = plot_stats.loc[np.isin(plot_stats[group], groups), :]
    plot_stats["group"] = plot_stats[group].round(0).astype(str)
    plot_stats.sort_values("group", inplace=True)
    id_vars = ["var_add", "modified_%", "#", "group", "threshold_%"]
    if split_modified == "last":
        plot_stats.loc[plot_stats[group] == len(model_names), "group"] = np.where(
            plot_stats.loc[plot_stats[group] == len(model_names), "modified"],
            plot_stats.loc[plot_stats[group] == len(model_names), "group"]
            + "\n"
            + "Mod",
            plot_stats.loc[plot_stats[group] == len(model_names), "group"]
            + "\n"
            + "Unmod",
        )
    elif split_modified is True:
        id_vars.append("modified")

    id_vars = [x for x in id_vars if x in plot_stats.columns]
    if isinstance(stats, str):
        stats = [stats]

    # col = col if col in plot_stats.columns else None
    # row = row if row in plot_stats.columns else None
    if kind == "catplot":
        data = plot_stats.melt(
            id_vars=id_vars,
            var_name="stat",
            value_vars=stats,
        ).sort_values("group")
        fg = sns.catplot(
            data=data,
            x=x_var,
            y="value",
            hue=hue,
            col=col if col in plot_stats.columns else None,
            row=row if row in plot_stats.columns else None,
            kind="boxen",
            height=height,
            aspect=aspect,
            palette=PALETTE if hue is not None else None,
            showfliers=False,
            fill=False,
            # dodge=True,
            # gap=0.5,
            margin_titles=margin_titles,
        )
    elif kind == "barplot":
        all_df = plot_stats.copy()
        if "modified" in plot_stats.columns:
            mod_df = plot_stats.loc[plot_stats["modified"], :]
            unmod_df = plot_stats.loc[np.invert(plot_stats["modified"]), :]
            plot_df = pd.concat(
                {
                    "All": all_df,
                    "Modified": mod_df,
                    "Unmodified": unmod_df,
                },
                names=["Type"],
            ).reset_index()
        else:
            plot_df = plot_stats
            plot_df["Type"] = "All"
        fg = sns.catplot(
            data=plot_df,
            x=x_var,
            y=y_var,
            hue=hue,
            kind="bar",
            col=col,
            row=row,
            fill=False,
            native_scale=True,
            dodge=dodge,
            gap=0.4 if dodge else None,
            palette=PALETTE if hue is not None else None,
            color=sns.color_palette(PALETTE)[0] if hue is None else None,
            linewidth=1,
            err_kws={"lw": 0.5},
            height=height,
            aspect=aspect,
            sharey=False,
            margin_titles=margin_titles,
        )
        for ax in fg.axes.flatten():
            for patch in ax.patches:
                patch.set_hatch(HATCH_DICT[patch.get_edgecolor()[:-1]])
                patch.set_hatch_linewidth(0.5)

        handles = fg.legend.legend_handles
        for handle in handles:
            handle.set_hatch(HATCH_DICT[handle.get_edgecolor()[:-1]])
            handle.set_hatch_linewidth(0.5)

    elif kind == "violinplot" or kind == "boxenplot":
        data = plot_stats.melt(
            id_vars=id_vars,
            var_name="stat",
            value_vars=stats,
        ).sort_values("group")
        fg = sns.catplot(
            data=data,
            x=x_var,
            y=y_var,
            hue=hue,
            col=col if col in plot_stats.columns else None,
            row=row if row in plot_stats.columns else None,
            kind="violin" if kind == "violinplot" else "boxen",
            height=height,
            aspect=1.2 if not annot_sizes or col or row else 1.75,
            palette=PALETTE if hue is not None else None,
            fill=False,
            # inner="quart" if kind == "violinplot" else None,
        )
    elif kind == "relplot":
        if relplot_agg:
            id_vars = [x for x in id_vars if x != "#"]
            data = (
                plot_stats.groupby(id_vars)[stats]
                .agg(["mean", "median"])
                .stack(future_stack=True)
            )
            data.index.names = data.index.names[:-1] + ["agg"]
            data = data.melt(var_name="stat", value_vars=stats, ignore_index=False)
            data = data.reset_index()
            fg = sns.relplot(
                data=data,
                x=x_var,
                y="value",
                hue="agg" if len(stats) == 1 else "stat",
                col=col if col in plot_stats.columns else None,
                row=row if row in plot_stats.columns else None,
                kind="line",
                markers=True,
                style="modified" if split_modified else "stat",
                height=height,
                aspect=1.2 if not annot_sizes or col or row else 1.75,
                palette=PALETTE,
                facet_kws=facet_kws,
            )
        else:
            data = plot_stats
            data = data.melt(
                id_vars=id_vars,
                var_name="stat",
                value_vars=stats,
            ).sort_values("group")
            fg = sns.relplot(
                data=data,
                x=x_var,
                y="value",
                hue="stat",
                col=col if col in plot_stats.columns else None,
                row=row if row in plot_stats.columns else None,
                kind="line",
                markers=True,
                style="modified" if split_modified else "stat",
                height=height,
                aspect=1.2 if not annot_sizes or col or row else 1.75,
                palette=PALETTE,
                facet_kws=facet_kws,
            )
    elif kind == "histplot":
        data = plot_stats.melt(
            id_vars=id_vars,
            var_name="stat",
            value_vars=stats,
        ).sort_values("group")
        fg = sns.displot(
            data=data,
            x="value",
            hue="group",
            col=col if col in plot_stats.columns else None,
            row=row if row in plot_stats.columns else None,
            kind="hist",
            height=height,
            aspect=1.2 if not annot_sizes or col or row else 1.75,
            palette=PALETTE,
            bins=50,
            binrange=(0, data["value"].quantile(0.99)),
            common_norm=False,
            element="step",
            stat="density",
            facet_kws=dict(sharey=False),
        )
    else:
        raise ValueError("Unknown plot type. Use 'relplot','catplot' or 'histplot'.")

    if annot_sizes:
        add_size_annotations(
            fg=fg,
            plot_stats=plot_stats,
            col=col,
            row=row,
            split_modified=split_modified,
        )
        fg.tight_layout()
    return fg


def plot_stats_class_curve_v_shifted(
    dataframe: pd.DataFrame,
    shifted_dataframe: pd.DataFrame,
    ref_dataframe: pd.DataFrame,
    shifted_ref_dataframe: pd.DataFrame,
    model_names: List[str] = MODEL_NAMES,
    value_name: str = "values",
    **kwargs,
) -> sns.FacetGrid:
    """
    Plot classification curves with data_offset compared to shifted data.

    This function calculates classification metrics for both original and shifted
    datasets, then visualizes the precision-recall curves with different data offsets
    and dataset types.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Original input data for analysis.
    shifted_dataframe : pd.DataFrame
        Shifted version of the input data.
    ref_dataframe : pd.DataFrame
        Reference data for comparison.
    shifted_ref_dataframe : pd.DataFrame
        Reference data for the shifted dataset.
    model_names : List[str], optional
        Names of models to analyze (default: MODEL_NAMES).
    value_name : str, optional
        Name of the value column (default: "values").
    **kwargs
        Additional keyword arguments passed to calculation functions.

    Returns
    -------
    sns.FacetGrid
        The seaborn FacetGrid object containing the plot.
    """
    data_offsets = np.array([0, 1.5, 10])
    plot_df = calc_classification_metrics(
        dataframe=dataframe,
        ref_dataframe=ref_dataframe,
        model_names=model_names,
        data_offsets=data_offsets,
        thresholds=DEFAULT_THRESHOLDS,
        value_name=value_name,
        group_stat="both",
        **kwargs,
    )
    if isinstance(dataframe, pd.DataFrame):
        plot_df = plot_df.to_frame("value").reset_index(
            names=["group_#", "threshold_%", "data_offset", "stat"]
        )
        plot_df["modified_%"] = None
    else:
        plot_df = plot_df.to_frame("value").reset_index(
            names=["modified_%", "group_#", "threshold_%", "data_offset", "stat"]
        )
    shifted_df = calc_classification_metrics(
        dataframe=shifted_dataframe,
        ref_dataframe=shifted_ref_dataframe,
        model_names=model_names,
        data_offsets=data_offsets - 1.5,
        thresholds=DEFAULT_THRESHOLDS,
        value_name=value_name,
        group_stat="both",
        **kwargs,
    )
    if isinstance(dataframe, pd.DataFrame):
        shifted_df = shifted_df.to_frame("value").reset_index(
            names=["group_#", "threshold_%", "data_offset", "stat"]
        )
        shifted_df["modified_%"] = None
    else:
        shifted_df = shifted_df.to_frame("value").reset_index(
            names=["modified_%", "group_#", "threshold_%", "data_offset", "stat"]
        )
    plot_df["df_type"] = "orig"
    shifted_df["data_offset"] += 1.5
    shifted_df["df_type"] = "shifted"
    plot_df = pd.concat([plot_df, shifted_df])
    plot_df = plot_df[plot_df["group_#"] == len(model_names)].drop(columns="group_#")
    plot_df = plot_df.pivot(
        index=["modified_%", "threshold_%", "data_offset", "df_type"],
        columns="stat",
        values="value",
    ).reset_index()
    fg1 = sns.relplot(
        data=plot_df,
        x="recall",
        y="precision",
        col="modified_%" if isinstance(dataframe, dict) else None,
        hue="data_offset",
        style="df_type",
        markers=True,
        kind="line",
        height=3 if isinstance(dataframe, dict) else 4,
        palette=PALETTE,
    )
    for ax in fg1.axes.flatten():
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
    return fg1


STATS_CLASS_VARS = Literal[
    "precision",
    "recall",
    "f1",
    "threshold_%",
    "group_len",
    "group_ratio",
    "group_true_ratio",
    "data_offset",
    "modified_%",
    "var_add",
    None,
]


def plot_stats_classification(
    dataframe: pd.DataFrame,
    ref_dataframe: pd.DataFrame,
    model_names: str | List[str] = MODEL_NAMES,
    kind: Literal["catplot", "relplot"] = "relplot",
    agg: Optional[str] = None,
    add_recursive: bool = False,
    precision_stop: Optional[float] = None,
    x_var: STATS_CLASS_VARS = "recall",
    y_var: STATS_CLASS_VARS | List[STATS_CLASS_VARS] = "precision",
    hue: STATS_CLASS_VARS = "data_offset",
    style: Optional[STATS_CLASS_VARS] = None,
    col: STATS_CLASS_VARS = "modified_%",
    row: STATS_CLASS_VARS = "var_add",
    group_stat: Literal["abs", "rel", "both", "norm"] = "abs",
    thresholds: float | List[float] | NDArray[np.float64] = DEFAULT_THRESHOLDS,
    data_offsets: float | List[float] | NDArray[np.float64] = DEFAULT_DATA_OFFSET,
    value_name: str = "values",
    facet_kws: Optional[dict] = None,
    group_no: Optional[int] = None,
    **kwargs,
) -> sns.FacetGrid:
    """
    Plot classification statistics using seaborn's relplot or catplot.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The main dataframe containing the data to be plotted.
    ref_dataframe : pd.DataFrame
        Reference dataframe for comparison.
    model_names : str or List[str], optional
        Names of models to include in the analysis. Default is MODEL_NAMES.
    kind : Literal["catplot", "relplot"], default "relplot"
        Type of plot to generate. Either 'relplot' for line plots or 'catplot' for box plots.
    agg : str, optional
        Aggregation function to apply to the data before plotting.
    add_recursive : bool, default False
        Whether to include recursive calculations in the plot.
    precision_stop : float, optional
        Threshold for stopping precision calculations.
    x_var : STATS_CLASS_VARS, default "recall"
        Variable to plot on the x-axis.
    y_var : STATS_CLASS_VARS or List[STATS_CLASS_VARS], default "precision"
        Variable(s) to plot on the y-axis.
    hue : STATS_CLASS_VARS, default "data_offset"
        Variable to map to the color hue.
    style : STATS_CLASS_VARS, optional
        Variable to map to the marker style.
    col : STATS_CLASS_VARS, default "modified_%"
        Variable to facet the plot by column.
    row : STATS_CLASS_VARS, default "var_add"
        Variable to facet the plot by row.
    group_stat : Literal["abs", "rel", "both", "norm"], default "abs"
        Group statistics calculation method.
    thresholds : float, List[float], or NDArray[np.float64], default DEFAULT_THRESHOLDS
        Threshold values for analysis.
    data_offsets : float, List[float], or NDArray[np.float64], default DEFAULT_DATA_OFFSET
        Data offset values for analysis.
    value_name : str, default "values"
        Name for the values column in the melted dataframe.
    facet_kws : dict, optional
        Additional keyword arguments for facetting.
    group_no : int, optional
        Group number to filter data by.
    **kwargs
        Additional keyword arguments passed to underlying plotting functions.

    Returns
    -------
    sns.FacetGrid
        The generated plot as a seaborn FacetGrid object.
    """
    if group_no is None:
        group_no = len(model_names)
    plot_df = calc_classification_metrics(
        dataframe=dataframe,
        ref_dataframe=ref_dataframe,
        model_names=model_names,
        data_offsets=data_offsets,
        thresholds=thresholds,
        value_name=value_name,
        group_stat=group_stat,
        **kwargs,
    )
    plot_df = plot_df.reset_index()

    if add_recursive:
        plot_df["recursive"] = False
        recursive_df = calc_classification_metrics(
            dataframe=dataframe,
            ref_dataframe=ref_dataframe,
            model_names=model_names,
            thresholds=thresholds,
            data_offsets=data_offsets,
            value_name=value_name,
            group_stat=group_stat,
            recursive=True,
            **kwargs,
        )
        recursive_df = recursive_df.reset_index()
        recursive_df["recursive"] = True
        style = "recursive"
        plot_df = pd.concat([plot_df, recursive_df])
    plot_df = plot_df[plot_df["group_#"] == group_no].drop(columns="group_#")
    id_vars = ["var_add", "modified_%", "recursive"]
    id_vars = [x for x in id_vars if x in plot_df.columns]
    if agg is not None:
        plot_df = (
            plot_df.groupby(id_vars)
            .agg(agg)
            .stack(0, future_stack=True)
            .to_frame(agg)
            .reset_index(names=id_vars + ["stat"])
        )
        if isinstance(y_var, str):
            y_var = [y_var]
        plot_df = plot_df[np.isin(plot_df["stat"], y_var)]
        y_var = agg
        hue = "stat" if hue is None else hue

    if kind == "relplot":
        fg1 = sns.relplot(
            data=plot_df,
            x=x_var,
            y=y_var,
            hue=hue if hue in plot_df.columns else None,
            style=style,
            col=col if col in plot_df.columns else None,
            row=row if row in plot_df.columns else None,
            markers=True if style is not None else False,
            kind="line",
            height=4 if col in plot_df.columns or row in plot_df.columns else 5,
            palette=PALETTE,
            facet_kws=facet_kws,
        )
    elif kind == "catplot":
        fg1 = sns.catplot(
            data=plot_df,
            x=x_var,
            y=y_var,
            hue=hue if hue in plot_df.columns else None,
            col=col if col in plot_df.columns else None,
            row=row if row in plot_df.columns else None,
            kind="box",
            height=4 if col in plot_df.columns or row in plot_df.columns else 5,
            palette=PALETTE,
            facet_kws=facet_kws,
        )
    else:
        raise ValueError("Wrong plot type. Available types are relplot and catplot.")
    return fg1


def plot_group_size_ratio(
    dataframe: pd.DataFrame,
    ref_dataframe: pd.DataFrame,
    model_names: str | List[str] = MODEL_NAMES,
    x_var: STATS_CLASS_VARS = "threshold_%",
    y_var: STATS_CLASS_VARS | List[STATS_CLASS_VARS] = "group_len",
    hue: STATS_CLASS_VARS = "var_add",
    style: Optional[STATS_CLASS_VARS] = None,
    col: STATS_CLASS_VARS = "modified_%",
    row: STATS_CLASS_VARS = "var_add",
    group_stat: Literal["abs", "rel", "both"] = "abs",
    thresholds: float | List[float] | NDArray[np.float64] = DEFAULT_THRESHOLDS,
    data_offsets: float | List[float] | NDArray[np.float64] = DEFAULT_DATA_OFFSET,
    value_name: str = "values",
    facet_kws: Optional[dict] = None,
    **kwargs,
) -> sns.FacetGrid:
    """
    Plot the logarithmic ratio of group sizes between two datasets.

    This function calculates classification metrics and visualizes the
    logarithmic ratio of group lengths between reference and current data,
    grouped by various parameters and thresholds.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The main dataframe containing the data to analyze.
    ref_dataframe : pd.DataFrame
        The reference dataframe for comparison.
    model_names : str or List[str], optional
        Names of models to include. Default is MODEL_NAMES.
    x_var : STATS_CLASS_VARS, default "threshold_%"
        Variable to use for x-axis.
    y_var : STATS_CLASS_VARS or List[STATS_CLASS_VARS], default "group_len"
        Variable(s) to use for y-axis.
    hue : STATS_CLASS_VARS, default "var_add"
        Variable to map to color hue.
    style : STATS_CLASS_VARS, optional
        Variable to map to line style.
    col : STATS_CLASS_VARS, default "modified_%"
        Variable to map to columns in facet grid.
    row : STATS_CLASS_VARS, default "var_add"
        Variable to map to rows in facet grid.
    group_stat : Literal["abs", "rel", "both"], default "abs"
        Type of group statistics to calculate.
    thresholds : float or List[float] or NDArray[np.float64], default DEFAULT_THRESHOLDS
        Threshold values for analysis.
    data_offsets : float or List[float] or NDArray[np.float64], default DEFAULT_DATA_OFFSET
        Data offset values for analysis.
    value_name : str, default "values"
        Name for the value column in the output dataframe.
    facet_kws : dict, optional
        Additional keyword arguments for facet grid.
    **kwargs : dict
        Additional keyword arguments for plotting.

    Returns
    -------
    sns.FacetGrid
        The matplotlib figure with the plotted data.
    """
    stats = calc_classification_metrics(
        dataframe=dataframe,
        ref_dataframe=ref_dataframe,
        model_names=model_names,
        thresholds=thresholds,
        data_offsets=data_offsets,
        value_name=value_name,
        group_stat=group_stat,
    )
    idx = pd.IndexSlice
    plot_df = np.log(
        stats.loc[idx[:, :, :, 5], "group_len"].droplevel("group_#")
        / stats.loc[idx[:, :, :, 4], "group_len"].droplevel("group_#")
    )
    fg = sns.relplot(
        data=plot_df.to_frame().reset_index(),
        x="threshold_%",
        y="group_len",
        col="modified_%",
        row="var_add",
        hue="var_add",
        kind="line",
        palette=PALETTE,
        facet_kws={"sharey": False},
    )

    for ax in fg.axes.flatten():
        ax.axhline(0, color="red", ls="dashed")
    return fg
