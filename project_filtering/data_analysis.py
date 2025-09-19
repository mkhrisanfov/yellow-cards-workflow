from typing import Any, Dict, List, Literal, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from scipy.stats import linregress

from project_filtering import (
    BASE_DIR,
    DEFAULT_DATA_OFFSET,
    DEFAULT_DATA_OFFSETS,
    DEFAULT_THRESHOLD,
    DEFAULT_THRESHOLDS,
    MODEL_NAMES,
)
from project_filtering.calculations import (
    calc_stats_classification,
    calc_stats_basic_prediction,
    calc_stats_groups,
    calc_stats_models,
    get_ground_truth,
    get_values,
    get_deltas,
    calc_delta_corr,
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
    dataframes: pd.DataFrame | Dict[Any, pd.DataFrame],
    clean_data: pd.Series | NDArray[np.float_] | pd.DataFrame,
    value_name: str = "values",
    col: str = "modified_%",
    row: str = "var_add",
    height: float = 3,
    aspect: float = 1,
    margin_titles: bool = False,
    **kwargs,
):
    plot_df = get_deltas(
        dataframes=dataframes,
        clean_data=clean_data,
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


def plot_binned_values(
    dataframe: pd.DataFrame,
    bins: NDArray[np.float_],
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
            calc_stats_basic_prediction(
                dataframe, model_names=model_names, value_name=value_name
            ),
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
    dataframes: (
        pd.DataFrame | Dict[Any, pd.DataFrame] | Dict[Any, Dict[Any, pd.DataFrame]]
    ),
    clean_data: Optional[
        List[float] | pd.Series | NDArray[np.float_] | pd.DataFrame
    ] = None,
    model_names: str | List[str] = MODEL_NAMES,
    data_offset: float = DEFAULT_DATA_OFFSET,
    value_name: str = "values",
    **kwargs,
):
    correlations = calc_delta_corr(
        dataframes=dataframes,
        clean_data=clean_data,
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
    dataframes: (
        pd.DataFrame | Dict[Any, pd.DataFrame] | Dict[Any, Dict[Any, pd.DataFrame]]
    ),
    clean_data: Optional[
        List[float] | pd.Series | NDArray[np.float_] | pd.DataFrame
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
    col_wrap: None | float = None,
    **kwargs,
):
    plot_df = calc_delta_corr(
        dataframes,
        clean_data=clean_data,
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


def plot_stats_models(
    dataframes: pd.DataFrame | Dict[Any, pd.DataFrame],
    clean_data: Optional[pd.Series | NDArray[np.float_] | pd.DataFrame] = None,
    original_data: Optional[pd.Series | NDArray[np.float_] | pd.DataFrame] = None,
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
):
    if isinstance(agg, str):
        agg = [agg]
    if clean_data is not None or original_data is not None:
        data = []
        mod_df = calc_stats_models(
            dataframes=dataframes,
            model_names=model_names,
            value_name=value_name,
            data_offset=data_offset,
        )
        mod_df["modified"] = get_ground_truth(dataframes, clean_data, value_name)
        mod_df["ref_type"] = "mod"
        if not ignore_mod_df:
            data.append(mod_df)
        if clean_data is not None:
            clean_df = calc_stats_models(
                dataframes=dataframes,
                model_names=model_names,
                clean_data=clean_data,
                value_name=value_name,
                data_offset=data_offset,
            )
            clean_df["modified"] = get_ground_truth(dataframes, clean_data, value_name)
            clean_df["ref_type"] = "noise"
            data.append(clean_df)

        if original_data is not None:
            orig_df = calc_stats_models(
                dataframes=dataframes,
                model_names=model_names,
                clean_data=original_data,
                value_name=value_name,
                data_offset=data_offset,
            )
            orig_df["modified"] = get_ground_truth(dataframes, clean_data, value_name)
            orig_df["ref_type"] = "clean"
            data.append(orig_df)
        plot_df = pd.concat(data)
    else:
        plot_df = calc_stats_models(
            dataframes=dataframes,
            model_names=model_names,
            clean_data=clean_data,
            value_name=value_name,
            data_offset=data_offset,
        )
    plot_df = plot_df.reset_index().drop(columns=["compound_#", "delta"])
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
    dataframes: pd.DataFrame | Dict[Any, pd.DataFrame],
    model_names: str | List[str] = MODEL_NAMES,
    clean_data: Optional[
        List[float] | pd.Series | NDArray[np.float_] | pd.DataFrame
    ] = None,
    data_offset: float = DEFAULT_DATA_OFFSET,
    thresholds: float | List[float] | NDArray[np.float_] = DEFAULT_THRESHOLD,
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
):
    stat_cols = {
        "abs": f"s_{len(model_names)}_a",
        "rel": f"s_{len(model_names)}_r",
        "both": f"s_{len(model_names)}_b",
    }
    group = stat_cols.get(group_stat, f"s_{len(model_names)}_a")
    group_stats = calc_stats_groups(
        dataframes=dataframes,
        model_names=model_names,
        thresholds=thresholds,
        data_offsets=data_offset,
        value_name=value_name,
    )
    if clean_data is not None:
        ground_truth = get_ground_truth(
            dataframes=dataframes, clean_dataframe=clean_data, value_name=value_name
        )
        plot_df = pd.concat([group_stats, ground_truth], axis=1)
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
    """Add size annotations to the plot showing group counts."""

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
    """Add size annotations to the plot showing group counts."""

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
    dataframes: pd.DataFrame | Dict[Any, pd.DataFrame],
    model_names: str | List[str] = MODEL_NAMES,
    clean_data: Optional[
        List[float] | pd.Series | NDArray[np.float_] | pd.DataFrame
    ] = None,
    mod_df_reference: bool = False,
    split_modified: bool | Literal["last"] = False,
    data_offset: float = DEFAULT_DATA_OFFSET,
    thresholds: float | List[float] | NDArray[np.float_] = DEFAULT_THRESHOLD,
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
):
    stat_cols = {
        "abs": f"s_{len(model_names)}_a",
        "rel": f"s_{len(model_names)}_r",
        "both": f"s_{len(model_names)}_b",
    }
    group = stat_cols.get(group_stat, f"s_{len(model_names)}_a")
    basic_stats = calc_stats_basic_prediction(
        dataframes=dataframes,
        model_names=model_names,
        data_offset=data_offset,
        clean_data=clean_data if not mod_df_reference else None,
        value_name=value_name,
    )
    group_stats = calc_stats_groups(
        dataframes=dataframes,
        model_names=model_names,
        thresholds=thresholds,
        data_offsets=data_offset,
        value_name=value_name,
    )
    models_stats = calc_stats_models(
        dataframes=dataframes,
        clean_data=clean_data if not mod_df_reference else None,
        model_names=model_names,
        data_offset=data_offset,
        value_name=value_name,
    )
    models_stats = models_stats.pivot(columns="model_name", values="delta_abs")
    if isinstance(thresholds, (list, tuple, np.ndarray)) and len(thresholds) > 1:
        basic_stats = pd.concat({t: basic_stats for t in thresholds}, axis=0)
        basic_stats.index.names = basic_stats.index.names[1:] + ["threshold_%"]
        models_stats = pd.concat({t: models_stats for t in thresholds}, axis=0)
        models_stats.index.names = models_stats.index.names[1:] + ["threshold_%"]
    plot_stats = pd.concat(
        [basic_stats, group_stats, models_stats],
        axis=1,
    )
    if clean_data is not None:
        ground_truth = get_ground_truth(
            dataframes=dataframes, clean_dataframe=clean_data, value_name=value_name
        )
        if isinstance(thresholds, (list, tuple, np.ndarray)) and len(thresholds) > 1:
            ground_truth = pd.concat({t: ground_truth for t in thresholds}, axis=0)
            ground_truth.index.names = ground_truth.index.names[1:] + ["threshold_%"]

        plot_stats = pd.concat(
            [plot_stats, ground_truth],
            axis=1,
        )

    plot_stats = plot_stats.reset_index()
    plot_stats = plot_stats.loc[np.isin(plot_stats[group], groups), :]
    plot_stats["group"] = plot_stats[group].round(0).astype(str)
    plot_stats.sort_values("group", inplace=True)

    id_vars = ["var_add", "modified_%", "compound_#", "group", "threshold_%"]
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
            id_vars = [x for x in id_vars if x != "compound_#"]
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


# def plot_stats_class_curve_v_mpe(
#     dataframes: pd.DataFrame | Dict[Any, pd.DataFrame],
#     clean_dataframe: pd.DataFrame,
#     model_names: List[str],
#     value_name: str = "values",
#     **kwargs,
# ):
#     plot_df = calc_stats_classification(
#         dataframes=dataframes,
#         clean_dataframe=clean_dataframe,
#         model_names=model_names,
#         data_offsets=[0, 1.5],
#         thresholds=DEFAULT_THRESHOLDS,
#         value_name=value_name,
#         group_stat="both",
#         **kwargs,
#     )
#     abs_df = calc_stats_classification(
#         dataframes=dataframes,
#         clean_dataframe=clean_dataframe,
#         model_names=model_names,
#         data_offsets=DEFAULT_DATA_OFFSET,
#         thresholds=DEFAULT_THRESHOLDS,
#         value_name=value_name,
#         group_stat="abs",
#         **kwargs,
#     )
#     plot_df = plot_df.reset_index()
#     abs_df = abs_df.reset_index()
#     abs_df["data_offset"] = "abs"
#     plot_df = pd.concat([plot_df, abs_df])
#     plot_df = plot_df[plot_df["group_#"] == len(model_names)].drop(columns="group_#")
#     fg1 = sns.relplot(
#         data=plot_df,
#         x="recall",
#         y="precision",
#         hue="data_offset",
#         style="data_offset",
#         col="modified_%" if isinstance(dataframes, dict) else None,
#         row="var_add" if "var_add" in plot_df.columns else None,
#         markers=True,
#         kind="line",
#         height=3 if isinstance(dataframes, dict) else 4,
#         palette=PALETTE,
#     )
#     for ax in fg1.axes.flatten():
#         ax.set_xlim(0, 1.05)
#         ax.set_ylim(0, 1.05)
#     return fg1


def plot_stats_class_curve_v_shifted(
    dataframes: pd.DataFrame | Dict[Any, pd.DataFrame],
    shifted_dataframes: pd.DataFrame | Dict[Any, pd.DataFrame],
    clean_dataframe: pd.DataFrame,
    shifted_clean_dataframe: pd.DataFrame,
    model_names: List[str] = MODEL_NAMES,
    value_name: str = "values",
    **kwargs,
):
    data_offsets = np.array([0, 1.5, 10])
    plot_df = calc_stats_classification(
        dataframes=dataframes,
        clean_dataframe=clean_dataframe,
        model_names=model_names,
        data_offsets=data_offsets,
        thresholds=DEFAULT_THRESHOLDS,
        value_name=value_name,
        group_stat="both",
        **kwargs,
    )
    if isinstance(dataframes, pd.DataFrame):
        plot_df = plot_df.to_frame("value").reset_index(
            names=["group_#", "threshold_%", "data_offset", "stat"]
        )
        plot_df["modified_%"] = None
    else:
        plot_df = plot_df.to_frame("value").reset_index(
            names=["modified_%", "group_#", "threshold_%", "data_offset", "stat"]
        )
    shifted_df = calc_stats_classification(
        dataframes=shifted_dataframes,
        clean_dataframe=shifted_clean_dataframe,
        model_names=model_names,
        data_offsets=data_offsets - 1.5,
        thresholds=DEFAULT_THRESHOLDS,
        value_name=value_name,
        group_stat="both",
        **kwargs,
    )
    if isinstance(dataframes, pd.DataFrame):
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
        col="modified_%" if isinstance(dataframes, dict) else None,
        hue="data_offset",
        style="df_type",
        markers=True,
        kind="line",
        height=3 if isinstance(dataframes, dict) else 4,
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
    dataframes: pd.DataFrame | Dict[Any, pd.DataFrame],
    clean_dataframe: pd.DataFrame,
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
    thresholds: float | List[float] | NDArray[np.float_] = DEFAULT_THRESHOLDS,
    data_offsets: float | List[float] | NDArray[np.float_] = DEFAULT_DATA_OFFSET,
    value_name: str = "values",
    facet_kws: Optional[dict] = None,
    group_no: Optional[int] = None,
    **kwargs,
):
    if group_no is None:
        group_no = len(model_names)
    plot_df = calc_stats_classification(
        dataframes=dataframes,
        clean_dataframe=clean_dataframe,
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
        recursive_df = calc_stats_classification(
            dataframes=dataframes,
            clean_dataframe=clean_dataframe,
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
    dataframes: pd.DataFrame | Dict[Any, pd.DataFrame],
    clean_dataframe: pd.DataFrame,
    model_names: str | List[str] = MODEL_NAMES,
    x_var: STATS_CLASS_VARS = "threshold_%",
    y_var: STATS_CLASS_VARS | List[STATS_CLASS_VARS] = "group_len",
    hue: STATS_CLASS_VARS = "var_add",
    style: Optional[STATS_CLASS_VARS] = None,
    col: STATS_CLASS_VARS = "modified_%",
    row: STATS_CLASS_VARS = "var_add",
    group_stat: Literal["abs", "rel", "both"] = "abs",
    thresholds: float | List[float] | NDArray[np.float_] = DEFAULT_THRESHOLDS,
    data_offsets: float | List[float] | NDArray[np.float_] = DEFAULT_DATA_OFFSET,
    value_name: str = "values",
    facet_kws: Optional[dict] = None,
    **kwargs,
):
    stats = calc_stats_classification(
        dataframes=dataframes,
        clean_dataframe=clean_dataframe,
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
