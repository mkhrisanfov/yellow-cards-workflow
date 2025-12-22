import os
from typing import List, Literal, Optional, Union, Iterable

import mordred as md
import numpy as np
import pandas as pd
from mordred import Autocorrelation
from numpy.typing import NDArray
from rdkit import Chem
from tqdm.auto import tqdm

from yellow_cards_workflow import (
    BASE_DIR,
    DEFAULT_VALUE_NAME,
    DEFAULT_DATA_OFFSET,
    DEFAULT_THRESHOLD,
    DEFAULT_THRESHOLDS,
    MODEL_NAMES,
    SEED,
    TOL,
)
from itertools import combinations


def load_descriptors(
    descriptors_file: os.PathLike | str = BASE_DIR
    / "data/processed/autocorr-descriptors.npy",
    scaler_mean_file: os.PathLike | str = BASE_DIR
    / "data/processed/autocorr-synthetic-mean.npy",
    scaler_scale_file: os.PathLike | str = BASE_DIR
    / "data/processed/autocorr-synthetic-scale.npy",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load descriptors and their normalization parameters.

    Parameters
    ----------
    descriptors_file : str or os.PathLike
        Path to the descriptors numpy array file
    scaler_mean_file : str or os.PathLike
        Path to the mean values for feature scaling
    scaler_scale_file : str or os.PathLike
        Path to the scale values for feature scaling
    verbose : bool, default=False
        If True, print the loaded descriptors names

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing (descriptors_names, scaler_mean, scaler_scale)
    """
    descriptors_names = np.load(descriptors_file, allow_pickle=False)
    scaler_mean = np.load(scaler_mean_file, allow_pickle=False)
    scaler_scale = np.load(scaler_scale_file, allow_pickle=False)
    if verbose:
        print(descriptors_names)
    return descriptors_names, scaler_mean, scaler_scale


def get_modified_entries(
    data: pd.DataFrame | NDArray[np.float64] | List[float],
    ref_data: pd.DataFrame | NDArray[np.float64] | List[float],
    value_name=DEFAULT_VALUE_NAME,
    **kwargs,
) -> pd.Series:
    """
    Compare data against reference values to identify modified entries.

    Parameters
    ----------
    data : pd.DataFrame or NDArray[np.float64]
        Input data to compare against reference values.
    ref_data : pd.DataFrame or NDArray[np.float64]
        Reference data for comparison.
    value_name : str, default=DEFAULT_VALUE_NAME
        Column name to extract values from DataFrames.
    **kwargs
        Additional keyword arguments passed to get_ref_data.

    Returns
    -------
    pd.DataFrame or pd.Series
        Boolean Series indicating which values differ beyond tolerance.
    """
    if isinstance(data, (np.ndarray, list)):
        normalized_data = pd.Series(data, name="#")
    elif isinstance(data, pd.DataFrame):
        normalized_data = data.loc[:, value_name]
    comp_values = get_ref_data(ref_data, value_name)
    modified_mask = np.abs(normalized_data - comp_values) > TOL
    return pd.Series(
        modified_mask, index=normalized_data.index, dtype=bool, name="modified"
    )


def get_synthetic_values(
    molecules: Chem.Mol | List[Chem.Mol],
) -> Union[np.float64, np.ndarray]:
    """
    Calculate synthetic values for molecules based on molecular descriptors.

    This function computes synthetic values for molecules using autocorrelation
    descriptors and random coefficients. It supports both single molecules
    and batches of molecules.

    Parameters
    ----------
    molecules : Chem.Mol | List[Chem.Mol]
        A single molecule (Chem.Mol) or a list/array of molecules to process.

    Returns
    -------
    Union[np.float64, np.ndarray]
        Single synthetic value for one molecule or array of values for multiple
        molecules.

    Raises
    ------
    TypeError
        If the input is not a Chem.Mol, list, or numpy array of molecules.
    """
    descriptors_names, scaler_mean, scaler_scale = load_descriptors()
    rng = np.random.default_rng(seed=SEED)
    random_coeffs = rng.random(len(descriptors_names))

    md_autocorr_calc = md.Calculator(Autocorrelation)
    if isinstance(molecules, Chem.Mol):
        res = md_autocorr_calc(molecules).asdict()
        vals = np.array([res[x] for x in descriptors_names])
        scaled_val = (vals - scaler_mean) / scaler_scale
        val = np.sum(scaled_val * random_coeffs) / np.sum(random_coeffs)
    elif isinstance(molecules, (list, np.ndarray)):
        vals = md_autocorr_calc.pandas(molecules)
        # select descriptors from a predefined list
        vals = vals.loc[:, descriptors_names]
        scaled_val = (vals - scaler_mean) / scaler_scale
        val = np.sum(scaled_val * random_coeffs, axis=1) / np.sum(random_coeffs)
    else:
        raise TypeError(
            """Unsupported input argument type.
            Either supply a single Chem.Mol molecule or a sequence of molecules."""
        )

    return val


def get_ref_data(
    ref_data: List[float] | pd.Series | NDArray[np.float64] | pd.DataFrame,
    value_name: Optional[str] = DEFAULT_VALUE_NAME,
) -> pd.Series:
    """
    Extract reference data into a pandas Series format.

    This function normalizes various input data types (list, numpy array,
    pandas Series, or DataFrame column) into a consistent pandas Series format.

    Parameters
    ----------
    ref_data : List[float] | pd.Series | NDArray[np.float64] | pd.DataFrame
        Input data to be converted. Can be a list of floats, numpy array,
        pandas Series, or pandas DataFrame.
    value_name : str, optional
        Column name to extract from DataFrame. Defaults to DEFAULT_VALUE_NAME.

    Returns
    -------
    pd.Series
        A pandas Series containing the data with appropriate index and name.

    Raises
    ------
    TypeError
        If ref_data is not one of the supported types.
    """
    if isinstance(ref_data, (list, np.ndarray)):
        return pd.Series(ref_data, index=np.arange(len(ref_data)), name="#")
    elif isinstance(ref_data, pd.DataFrame):
        return ref_data.loc[:, value_name]
    elif isinstance(ref_data, pd.Series):
        return ref_data
    else:
        raise TypeError("Unsupported ref_data type.")


def calc_values_diffs(
    dataframe: pd.DataFrame,
    ref_data: List[float] | pd.Series | NDArray[np.float64] | pd.DataFrame,
    value_name: str = DEFAULT_VALUE_NAME,
) -> pd.Series:
    """
    Calculate differences between dataframe values and reference data.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing values to calculate deltas from
    ref_data : List[float] | pd.Series | NDArray[np.float64] | pd.DataFrame
        Reference data to subtract from dataframe values
    value_name : str, default=DEFAULT_VALUE_NAME
        Column name containing values to calculate deltas from

    Returns
    -------
    pd.Series
        Series containing the calculated deltas
    """
    ref_data = get_ref_data(ref_data=ref_data, value_name=value_name)
    results = dataframe.loc[:, value_name] - ref_data
    return results


def calc_basic_stats(
    dataframe: pd.DataFrame,
    model_names: str | List[str] = MODEL_NAMES,
    ref_data: Optional[
        List[float] | pd.Series | NDArray[np.float64] | pd.DataFrame
    ] = None,
    value_name: str = DEFAULT_VALUE_NAME,
    data_offset: float = DEFAULT_DATA_OFFSET,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate basic prediction statistics across the models.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame containing model predictions and reference data.
    model_names : str or List[str], optional
        Name(s) of model columns to use for calculations. Defaults to MODEL_NAMES.
    ref_data : array-like, optional
        Reference data for comparison. If None, uses dataframe as reference.
    value_name : str, default DEFAULT_VALUE_NAME
        Column name containing reference values when ref_data is provided.
    data_offset : float, default DEFAULT_DATA_OFFSET
        Offset value to prevent division by zero in relative difference calculations.
    **kwargs
        Additional keyword arguments passed to underlying functions.

    Returns
    -------
    pd.DataFrame
        DataFrame containing prediction statistics:
        - p_median: median of model predictions
        - p_mean: mean of model predictions
        - p_std: standard deviation of model predictions
        - d_mean_a: absolute difference between mean prediction and reference
        - d_mean_r: relative difference between mean prediction and reference
        - d_median_a: absolute difference between median prediction and reference
        - d_median_r: relative difference between median prediction and reference
    """
    # get normalized reference data
    if ref_data is not None:
        ref_data = get_ref_data(ref_data=ref_data, value_name=value_name)
    else:
        ref_data = get_ref_data(ref_data=dataframe, value_name=value_name)

    # calculate basic predictions statistics
    results = pd.DataFrame(index=dataframe.index)
    model_data = dataframe.loc[:, model_names]
    results["p_median"] = model_data.median(axis=1)
    results["p_mean"] = model_data.mean(axis=1)
    results["p_std"] = model_data.std(axis=1)

    # calculate statistics for mean and median predictions
    results["d_mean_a"] = (results["p_mean"] - ref_data).abs()
    results["d_mean_r"] = results["d_mean_a"] / (results["p_mean"] + data_offset)
    results["d_median_a"] = (results["p_median"] - ref_data).abs()
    results["d_median_r"] = results["d_median_a"] / (results["p_median"] + data_offset)
    return results


# TODO: vectorize the function
def calc_models_diffs(
    dataframe: pd.DataFrame,
    ref_data: Optional[
        List[float] | pd.Series | NDArray[np.float64] | pd.DataFrame
    ] = None,
    model_names: str | List[str] = MODEL_NAMES,
    data_offset: float = DEFAULT_DATA_OFFSET,
    value_name: str = DEFAULT_VALUE_NAME,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate differences between model predictions and reference data.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame containing model predictions and reference values.
    ref_data : array-like, optional
        Reference data for comparison. If None, uses values from `value_name` column.
    model_names : str or list of str, default MODEL_NAMES
        Name(s) of model column(s) to calculate deltas for.
    data_offset : float, default DEFAULT_DATA_OFFSET
        Small offset added to median for relative delta calculation to avoid division by zero.
    value_name : str, default DEFAULT_VALUE_NAME
        Column name containing reference values when ref_data is None.
    **kwargs
        Additional keyword arguments passed to get_ref_data function.

    Returns
    -------
    pd.DataFrame
        DataFrame containing delta, delta_abs, and delta_rel columns for each model,
        with model_name column indicating which model each row corresponds to.
    """
    # calculate median of predictions
    p_median = dataframe.loc[:, model_names].median(
        axis=1  # pyright: ignore[reportArgumentType]
    )

    # get normalized reference data
    if ref_data is not None:
        ref_data = get_ref_data(ref_data, value_name)
    else:
        ref_data = dataframe.loc[:, value_name]

    deltas = []
    for i, model in enumerate(model_names):
        model_deltas = []
        delta = dataframe.loc[:, model] - ref_data
        delta_abs = np.abs(delta)
        delta_rel = np.abs(delta_abs / (p_median + data_offset))

        model_deltas.append(pd.Series(delta, name="delta"))
        model_deltas.append(pd.Series(delta_abs, name="delta_abs"))
        model_deltas.append(pd.Series(delta_rel, name="delta_rel"))
        model_delta = pd.concat(model_deltas, axis=1)
        model_delta["model_name"] = model
        deltas.append(model_delta)
    results = pd.concat(deltas)
    # results.set_index("model_name", append=True, inplace=True)
    return results


# TODO: add groups by normalized error
def calc_groups(
    dataframe: pd.DataFrame,
    model_names: str | List[str] | Iterable[str] = MODEL_NAMES,
    thresholds: (
        float | List[float] | NDArray[np.float64] | pd.Series
    ) = DEFAULT_THRESHOLD,
    data_offsets: (
        float | List[float] | NDArray[np.float64] | pd.Series
    ) = DEFAULT_DATA_OFFSET,
    value_name: str = DEFAULT_VALUE_NAME,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate group number for each entry depending on threshold and data offset.

    This function computes absolute and relative deviations for each
    model in the dataset, grouped by specified variables. It calculates quantiles for
    each group and determines how many models exceed given thresholds for absolute
    and relative deviations.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame containing model data and reference values.
    model_names : str or list of str, optional
        Name(s) of the model columns to process. Defaults to MODEL_NAMES.
    thresholds : float, list of float, np.ndarray, or pd.Series, optional
        Threshold(s) for quantile calculation. Defaults to DEFAULT_THRESHOLD.
    data_offsets : float, list of float, np.ndarray, or pd.Series, optional
        Offset value(s) for relative deviation calculation. Defaults to DEFAULT_DATA_OFFSET.
    value_name : str, optional
        Name of the column containing true values. Defaults to DEFAULT_VALUE_NAME.
    **kwargs
        Additional keyword arguments passed to underlying functions.

    Returns
    -------
    pd.DataFrame
        DataFrame with statistical results including:
        - s_{n_models}_a: Count of models exceeding absolute threshold
        - s_{n_models}_r: Count of models exceeding relative threshold
        - s_{n_models}_b: Count of models exceeding both thresholds
        - s_{n_models}_n: Placeholder for consistency
        - threshold_%: Applied threshold value
        - data_offset: Applied data offset value
    """
    p_median = dataframe.loc[:, model_names].median(axis=1)

    # normalized thresholds and data_offsets
    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds]
    if isinstance(data_offsets, (int, float)):
        data_offsets = [data_offsets]

    # get index levels
    idx_vars = ["var_add", "modified_%"]
    idx_vars = [x for x in idx_vars if x in dataframe.index.names]
    if len(idx_vars) == 0:
        idx_vars = ["dummy"]
        dataframe = pd.concat({"dummy": dataframe}, names=["dummy"])

    p_median = dataframe.loc[:, model_names].median(axis=1).to_numpy()
    y_true = dataframe.loc[:, value_name].to_numpy()

    # convert model data to a 2D numpy array for fast operations
    model_data = dataframe.loc[:, model_names].to_numpy()
    n_models = model_data.shape[1]  # pyright: ignore[reportGeneralTypeIssues]

    # get indices for each separate (var_add, modified_%) dataset
    groups = dataframe.groupby(level=idx_vars).indices
    results = []
    delta_abs = np.abs(model_data - y_true[:, None])
    for t in thresholds:
        for d in data_offsets:
            delta_rel = delta_abs / (p_median[:, None] + d)

            s_a = np.zeros(len(y_true), dtype=np.int8)
            s_r = np.zeros(len(y_true), dtype=np.int8)
            s_b = np.zeros(len(y_true), dtype=np.int8)

            # Compute quantiles per group efficiently
            for g_idx in groups.values():
                abs_g = delta_abs[g_idx, :]
                rel_g = delta_rel[g_idx, :]
                abs_q = np.quantile(abs_g, (100 - t) / 100.0, axis=0)
                rel_q = np.quantile(rel_g, (100 - t) / 100.0, axis=0)

                mask_abs = abs_g >= abs_q
                mask_rel = rel_g >= rel_q
                s_a[g_idx] = mask_abs.sum(axis=1)
                s_r[g_idx] = mask_rel.sum(axis=1)
                s_b[g_idx] = (mask_abs & mask_rel).sum(axis=1)

            result = pd.DataFrame(
                {
                    f"s_{n_models}_a": s_a,
                    f"s_{n_models}_r": s_r,
                    f"s_{n_models}_b": s_b,
                    f"s_{n_models}_n": 0,  # placeholder for consistency
                    "threshold_%": t,
                    "data_offset": d,
                },
                index=dataframe.index,
            )
            results.append(result)
    final_df = pd.concat(results)
    final_df.set_index(["threshold_%", "data_offset"], append=True, inplace=True)
    return final_df


def calc_diffs_corrs(
    dataframe: pd.DataFrame,
    ref_data: Optional[
        List[float] | pd.Series | NDArray[np.float64] | pd.DataFrame
    ] = None,
    model_names: str | List[str] = MODEL_NAMES,
    data_offset: float = DEFAULT_DATA_OFFSET,
    value_name: str = DEFAULT_VALUE_NAME,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate correlation matrix of prediction errors between the models.

    This function computes correlations between delta values  of the models.
    It uses the calc_models_diffs function to generate the delta values and then
    constructs a correlation matrix.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe containing the data to analyze.
    ref_data : array-like, optional
        Reference data for comparison. If None, uses default reference.
    model_names : str or list of str, default MODEL_NAMES
        Name(s) of the statistical models to use.
    data_offset : float, default DEFAULT_DATA_OFFSET
        Offset value to apply to the data.
    value_name : str, default DEFAULT_VALUE_NAME
        Name of the column containing values to analyze.
    **kwargs
        Additional keyword arguments passed to calc_models_diffs.

    Returns
    -------
    pd.DataFrame
        Correlation matrix of delta values from the statistical models.
        If multiple grouping variables are present, returns grouped correlation
        matrices.
    """
    stats_df = calc_models_diffs(
        dataframe=dataframe,
        ref_data=ref_data,
        model_names=model_names,
        data_offset=data_offset,
        value_name=value_name,
        **kwargs,
    ).reset_index()

    idx_vars = ["#", "modified_%", "var_add"]
    idx_vars = [x for x in idx_vars if x in stats_df.columns]
    stats_df = stats_df.pivot(index=idx_vars, columns="model_name", values="delta")

    # correlation across the datasets, therefore exclude entry index
    if len(idx_vars) > 1:
        grouped = stats_df.groupby(level=idx_vars[1:], group_keys=True)
        return grouped.corr()
    else:
        return stats_df.corr()


def calc_classification_metrics(
    dataframe: pd.DataFrame,
    ref_dataframe: pd.DataFrame,
    model_names: str | List[str] = MODEL_NAMES,
    thresholds: float | List[float] | NDArray[np.float64] = DEFAULT_THRESHOLD,
    data_offsets: float | List[float] | NDArray[np.float64] = DEFAULT_DATA_OFFSET,
    value_name: str = DEFAULT_VALUE_NAME,
    group_stat: Literal["abs", "rel", "both", "norm"] = "abs",
    last_group_only: bool = False,
    recursive: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate classification statistics for grouped data.

    This function computes various classification metrics (precision, recall, F1-score)
    for grouped data based on thresholds and data offsets. It compares model predictions
    against ground truth data to evaluate performance.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe containing model predictions.
    ref_dataframe : pd.DataFrame
        Reference dataframe with ground truth values.
    model_names : str or List[str], optional
        Name(s) of the models to analyze. Default is MODEL_NAMES.
    thresholds : float, List[float], or NDArray[np.float64], optional
        Threshold(s) for classification. Default is DEFAULT_THRESHOLD.
    data_offsets : float, List[float], or NDArray[np.float64], optional
        Data offset(s) for classification. Default is DEFAULT_DATA_OFFSET.
    value_name : str, optional
        Name of the value column to analyze. Default is DEFAULT_VALUE_NAME.
    group_stat : {'abs', 'rel', 'both', 'norm'}, optional
        Group statistic variant to use. Default is 'abs'.
    last_group_only : bool, optional
        If True, only the last group is considered. Default is False.
    recursive : bool, optional
        If True, enables recursive logic (not yet implemented). Default is False.
    **kwargs
        Additional keyword arguments passed to underlying functions.

    Returns
    -------
    pd.DataFrame
        DataFrame containing classification metrics including:
        - group_len: Number of items in the group
        - group_pos: Number of positive items in the group
        - group_ratio: Ratio of group size to total dataset size
        - group_pos_ratio: Ratio of positive items to total dataset size
        - precision: Precision score for the group
        - recall: Recall score for the group
        - f1: F1-score for the group
    """
    gstat_variants = {
        "abs": f"s_{len(model_names)}_a",
        "rel": f"s_{len(model_names)}_r",
        "both": f"s_{len(model_names)}_b",
        "norm": f"s_{len(model_names)}_n",
    }
    gstat = gstat_variants.get(group_stat, f"s_{len(model_names)}_a")
    modified = get_modified_entries(
        data=dataframe,
        ref_data=ref_dataframe,
        value_name=value_name,
    ).to_frame("modified")
    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds]

    # Precompute group stats with the optimized function
    group_stats = calc_groups(
        dataframe=dataframe,
        model_names=model_names,
        thresholds=thresholds,
        data_offsets=data_offsets,
        value_name=value_name,
    )
    group_stats.set_index(gstat, append=True, inplace=True)

    if recursive:
        raise NotImplementedError("Recursive logic not yet implemented")

    base_df = (
        group_stats.xs(thresholds[0], level="threshold_%").join(modified).reset_index()
    )

    idx_vars = ["var_add", "modified_%", "data_offset"]
    ds_idx_vars = [x for x in idx_vars if x in base_df.columns]

    ds_counts = (
        base_df.groupby(ds_idx_vars)["modified"]
        .value_counts(dropna=False)
        .unstack(fill_value=0)
        .rename(columns={True: "ds_pos", False: "neg_ds"})
    )

    ds_counts["ds_len"] = ds_counts["ds_pos"] + ds_counts["neg_ds"]

    results = {}
    for t in thresholds:
        gs = group_stats.xs(t, level="threshold_%").join(modified)
        df_tmp = gs.reset_index()
        group_keys = [x for x in idx_vars + [gstat] if x in df_tmp.columns]

        # Compute positive/negative counts per group
        grp = df_tmp.groupby(group_keys)["modified"]
        group_counts = grp.value_counts(dropna=False).unstack(fill_value=0)
        group_counts = group_counts.rename(columns={True: "group_pos", False: "neg"})
        group_counts["group_len"] = group_counts["group_pos"] + group_counts["neg"]

        # Merge with dataset-level counts (by group hierarchy)
        merge_keys = [k for k in ds_counts.index.names if k in group_counts.index.names]
        merged = group_counts.join(ds_counts, on=merge_keys, how="left").fillna(0)

        # Compute classification metrics
        merged["precision"] = merged["group_pos"] / (merged["group_len"] + TOL)
        merged["recall"] = merged["group_pos"] / (merged["ds_pos"] + TOL)
        merged["f1"] = (2 * merged["precision"] * merged["recall"]) / (
            merged["precision"] + merged["recall"] + TOL
        )
        merged["group_ratio"] = merged["group_len"] / (merged["ds_len"] + TOL)
        merged["group_pos_ratio"] = merged["group_pos"] / (merged["ds_len"] + TOL)

        metric_cols = [
            "group_len",
            "group_pos",
            "group_ratio",
            "group_pos_ratio",
            "precision",
            "recall",
            "f1",
        ]

        results[t] = merged[metric_cols]
    full_idx = ["threshold_%"] + idx_vars + [gstat]
    full_idx = [x for x in full_idx if x in group_stats.index.names]
    full_index = pd.MultiIndex.from_product(
        [group_stats.index.get_level_values(name).unique() for name in full_idx]
    )
    results = (
        pd.concat(results, names=["threshold_%"])
        .reorder_levels(full_idx)
        .reindex(index=full_index, fill_value=0)
        .rename_axis(index={gstat: "group_#"})
    )
    return results


def calc_multi_model_stats(
    dataframe: pd.DataFrame,
    ref_dataframe: pd.DataFrame,
    model_names: List[str],
    data_offset: float = DEFAULT_DATA_OFFSET,
) -> pd.DataFrame:
    """
    Calculate classification statistics for all combinations of models with default threshold.

    This function computes classification statistics for all possible combinations
    of the provided model names, from single models up to all models combined.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The main dataframe containing the data to analyze
    ref_dataframe : pd.DataFrame
        The reference dataframe for comparison
    model_names : List[str]
        List of model names to consider for combinations
    data_offset : float, optional
        Offset value for data processing (default: DEFAULT_DATA_OFFSET)

    Returns
    -------
    pd.DataFrame
        Concatenated statistics for all model combinations, indexed by number of models
        and model combinations
    """
    models_stats = {}
    for i in tqdm(range(1, len(model_names) + 1)):
        combos = {}
        for combo in combinations(model_names, i):
            stats = calc_classification_metrics(
                dataframe=dataframe,
                ref_dataframe=ref_dataframe,
                model_names=combo,
                data_offset=data_offset,
            )
            combos[";".join([str(x) for x in combo])] = stats.xs(
                len(combo), level="group_#"
            )
        combos = pd.concat(combos, names=["model_combo"])
        models_stats[i] = combos
    models_stats = pd.concat(models_stats, names=["n_models"])
    # models_stats = models_stats.reset_index()
    return models_stats


def calc_classification_metrics_multi_model(
    dataframe: pd.DataFrame,
    ref_dataframe: pd.DataFrame,
    model_names: List[str],
    data_offset: float = DEFAULT_DATA_OFFSET,
    thresholds: float | List[float] | NDArray[np.float64] = DEFAULT_THRESHOLDS,
    criterium: str = "max",
) -> pd.DataFrame:
    """
    Calculate classification statistics for all combinations of models from 1 to all.

    This function computes classification statistics for various combinations of
    models and selects the best or most average performing model combination.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The main dataframe containing the data to analyze
    ref_dataframe : pd.DataFrame
        The reference dataframe for comparison
    model_names : List[str]
        List of model names to consider for combination
    data_offset : float, optional
        Data offset value for calculations (default: DEFAULT_DATA_OFFSET)
    thresholds : float | List[float] | NDArray[np.float64], optional
        Thresholds for classification (default: DEFAULT_THRESHOLDS)
    criterium : str, optional
        Selection criterion for best model ('max' or 'median') (default: "max")

    Returns
    -------
    pd.DataFrame
        Concatenated plot data with classification statistics for the combinations of the models
    """
    models_stats = calc_multi_model_stats(
        dataframe=dataframe,
        ref_dataframe=ref_dataframe,
        model_names=model_names,
        data_offset=data_offset,
    )
    models_best = {}
    for i in range(1, len(model_names) + 1):
        models_slice = models_stats.xs(i, level="n_models")
        if criterium == "max":
            models_best[i] = models_slice.loc[[models_slice["f1"].idxmax()]]
        elif criterium == "median":
            models_best[i] = models_slice.iloc[
                np.argmin(np.abs(models_slice["f1"] - models_slice["f1"].median()))
            ]

    models_best = pd.concat(models_best, names=["n_models"])
    plot_data = {}
    for i, row in models_best.reset_index().iterrows():
        model_names = row["model_combo"].split(";")
        res = calc_classification_metrics(
            dataframe=dataframe,
            ref_dataframe=ref_dataframe,
            model_names=model_names,
            thresholds=thresholds,
            data_offsets=data_offset,
            group_stat="abs",
        )
        res = res.xs(len(model_names), level="group_#")
        plot_data[row["n_models"]] = res
    plot_data = pd.concat(plot_data, names=["n_models"])
    return plot_data


def compare_threshold_methods(
    dataframe,
    ref_data,
    model_names=MODEL_NAMES,
    data_offsets=DEFAULT_DATA_OFFSET,
    value_name=DEFAULT_VALUE_NAME,
    num_points=100,
):
    """
    Calculate comparison between yellow cards and quantile thresholding.


    Parameters
    ----------
    dataframe : pandas.DataFrame
        The main dataframe containing model predictions.
    ref_data : pandas.DataFrame
        The reference dataframe containing ground truth values.
    model_names : list of str, optional
        List of model names to analyze. Defaults to MODEL_NAMES.
    data_offsets : list of int, optional
        Data offsets to consider. Defaults to DEFAULT_DATA_OFFSET.
    value_name : str, optional
        Name of the value column to analyze. Defaults to DEFAULT_VALUE_NAME.
    num_points : int, optional
        Number of points for threshold calculation. Defaults to 100.

    Returns
    -------
    pandas.DataFrame
        Combined statistical results with precision, recall, and F1-score
        for both quantile and yellow cards approaches.
    """
    # calculate basic prediction stats
    basic_stats = calc_basic_stats(
        dataframe=dataframe,
        model_names=model_names,
        value_name=value_name,
        data_offset=data_offsets,
    )

    # get modified status
    modified = get_modified_entries(
        data=dataframe, ref_data=ref_data, value_name=value_name
    )
    stats = basic_stats.join(modified)

    # set index variables
    idx_vars = ["var_add", "modified_%"]
    idx_vars = [x for x in idx_vars if x in stats.index.names]

    quant_results = {}

    # select only two necessary columns for quantile thresholding
    quant_df = stats[["d_median_a", "modified"]].copy()

    # add modification status to index
    quant_df = quant_df.set_index("modified", append=True)

    # make all possible combinations of MultiIndex so that
    # output directly matches the inpu DataFrame
    all_combos = [
        quant_df.index.get_level_values(name).unique()
        for name in (idx_vars + ["modified"])
    ]
    all_combos.append(pd.Index([False, True], name="error_quant"))
    all_combos = pd.MultiIndex.from_product(all_combos)

    # generate distribution for thresholds
    thresholds = np.concatenate(
        [
            np.geomspace(0.01, 90, num_points // 2),
            100 - np.geomspace(0.0001, 10, num_points // 2)[::-1],
        ]
    )

    # calculate dataset statistics
    ds_quant = (
        quant_df.groupby(idx_vars + ["modified"])
        .size()
        .unstack(level="modified")
        .fillna(0)
        .rename(columns={True: "ds_pos", False: "ds_neg"})
    )
    ds_quant["ds_total"] = ds_quant["ds_pos"] + ds_quant["ds_neg"]

    quant_gb = quant_df.groupby(idx_vars)["d_median_a"]
    for t in thresholds:
        quant_df["error_quant"] = quant_df["d_median_a"].gt(quant_gb.quantile(t / 100))

        # per group statistics
        group_quant = (
            quant_df.groupby(by=idx_vars + ["modified", "error_quant"])
            .size()
            .unstack(level="modified")
            .fillna(0)
            .rename(columns={True: "pos", False: "neg"})
        )
        group_quant["total"] = group_quant["pos"] + group_quant["neg"]

        merged_quant = group_quant.join(ds_quant).xs(True, level="error_quant")
        merged_quant["group_len"] = merged_quant["total"]
        merged_quant["precision"] = merged_quant["pos"] / (merged_quant["total"] + TOL)
        merged_quant["recall"] = merged_quant["pos"] / (merged_quant["ds_pos"] + TOL)
        merged_quant["f1"] = (
            2
            * merged_quant["precision"]
            * merged_quant["recall"]
            / (merged_quant["precision"] + merged_quant["recall"] + TOL)
        )
        quant_results[t] = merged_quant[["precision", "recall", "f1"]]
    quant_results = pd.concat(quant_results, names=["threshold"])
    quant_results["type"] = "Quantile"
    quant_results = quant_results.set_index("type", append=True)

    group_results = calc_classification_metrics(
        dataframe=dataframe,
        ref_dataframe=ref_data,
        model_names=model_names,
        thresholds=np.geomspace(0.0001, 100, num_points),
    )[["precision", "recall", "f1"]]
    group_results["type"] = "Yellow Cards"
    group_results = group_results.set_index("type", append=True)
    group_results = group_results.xs(5, level="group_#")
    group_results = group_results.rename_axis(
        index={"threshold_%": "threshold"}
    ).droplevel(level="data_offset", axis=0)
    group_results = group_results.reorder_levels(quant_results.index.names)
    plot_data = pd.concat([quant_results, group_results])
    return plot_data.fillna(0).sort_index()
