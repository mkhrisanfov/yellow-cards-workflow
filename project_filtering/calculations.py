import os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

import mordred as md
import numpy as np
import pandas as pd
from mordred import Autocorrelation
from numpy.typing import NDArray
from rdkit import Chem
from tqdm.auto import tqdm
from sklearn.metrics import auc, average_precision_score

from project_filtering import (
    BASE_DIR,
    DEFAULT_DATA_OFFSET,
    DEFAULT_DATA_OFFSETS,
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
    descriptors_names = np.load(descriptors_file, allow_pickle=False)
    scaler_mean = np.load(scaler_mean_file, allow_pickle=False)
    scaler_scale = np.load(scaler_scale_file, allow_pickle=False)
    if verbose:
        print(descriptors_names)
    return descriptors_names, scaler_mean, scaler_scale


def get_ground_truth(
    dataframes: pd.DataFrame | NDArray[np.float_] | Dict[Any, pd.DataFrame],
    clean_dataframe: pd.DataFrame | NDArray[np.float_],
    value_name="values",
    **kwargs,
) -> pd.DataFrame | pd.Series:
    """
    Get the ground truth values from the dataframe.

    Args:
        dataframe (pd.DataFrame): The dataframe containing the predictions and true values.
        clean_dataframe (pd.DataFrame): The clean dataframe containing the true values.
        value_name (str): The name of the column containing the true values.

    Returns:
        np.ndarray: An array containing the ground truth values.
    """
    if isinstance(dataframes, np.ndarray):
        df_values = dataframes
    elif isinstance(dataframes, pd.DataFrame):
        df_values = dataframes[value_name]
    elif isinstance(dataframes, dict):
        results = {}
        new_name = "dict_name"
        for key, dataframe in dataframes.items():
            results[key] = get_ground_truth(
                dataframes=dataframe,
                clean_dataframe=clean_dataframe,
                value_name=value_name,
            )
            if isinstance(dataframe, dict):
                new_name = "var_add"
            elif isinstance(dataframe, pd.DataFrame):
                new_name = "modified_%"
        results = pd.concat(results)
        results.index.names = [new_name] + results.index.names[1:]
        return results
    else:
        raise TypeError("Unsupported dataframe type")
    if isinstance(clean_dataframe, np.ndarray):
        comp_values = clean_dataframe
    elif isinstance(clean_dataframe, pd.DataFrame):
        comp_values = clean_dataframe[value_name]
    else:
        raise TypeError("Unsupported clean_dataframe type")
    modified_mask = np.abs(df_values - comp_values) > TOL
    return pd.Series(modified_mask, name="modified")


def calc_synth_values(
    molecules: Chem.Mol | List[Chem.Mol],
) -> Union[np.float_, np.ndarray]:
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


def wrapper_dataframes(
    function: Callable,
    dataframes: Dict[Any, pd.DataFrame] | Dict[Any, Dict[Any, pd.DataFrame]],
    **kwargs,
):
    if isinstance(dataframes, dict):
        results = {}
        new_name = "dict_name"
        for key, dataframe in dataframes.items():
            if isinstance(dataframe, dict):
                new_name = "var_add"
            elif isinstance(dataframe, pd.DataFrame):
                new_name = "modified_%"
            else:
                raise TypeError("Unsupported nested type.")
            results[key] = function(
                dataframes=dataframe,
                **kwargs,
            )
        results = pd.concat(results)
        results.index.names = [new_name] + results.index.names[1:]
        return results
    else:
        raise TypeError(
            "Unsupported dataframes type. "
            "Supply a Dict[Any, pd.DataFrame] or a Dict[Any, Dict[Any, pd.DataFrame]]."
        )


def wrapper_thresholds(
    function: Callable,
    thresholds: NDArray[np.float_] | List[float] | pd.Series,
    **kwargs,
):
    if isinstance(thresholds, (list, np.ndarray, pd.Series)):
        results = {}
        for threshold in thresholds:
            results[threshold] = function(
                thresholds=threshold,
                **kwargs,
            )
        results = pd.concat(results)
        results.index.names = ["threshold_%"] + results.index.names[1:]
        return results
    else:
        raise TypeError("Unsupported thresholds type. Supply an Iterable of floats.")


def wrapper_data_offsets(
    function: Callable,
    data_offsets: NDArray[np.float_] | List[float] | pd.Series,
    **kwargs,
):
    if isinstance(data_offsets, (list, np.ndarray, pd.Series)):
        results = {}
        for data_offset in data_offsets:
            results[data_offset] = function(
                data_offsets=data_offset,
                **kwargs,
            )
        results = pd.concat(results)
        results.index.names = ["data_offset"] + results.index.names[1:]
        return results
    else:
        raise TypeError("Unsupported data_offsets type. Supply an Iterable of floats.")


def get_values(
    dataframes: (
        pd.DataFrame | Dict[Any, pd.DataFrame] | Dict[Any, Dict[Any, pd.DataFrame]]
    ),
    value_name: str = "values",
):
    if isinstance(dataframes, pd.DataFrame):
        results = dataframes.loc[:, value_name]
        results.index.names = ["compound_#"] + results.index.names[1:]
        return results
    elif isinstance(dataframes, dict):
        return wrapper_dataframes(
            get_values, dataframes=dataframes, value_name=value_name
        )
    else:
        raise TypeError(
            "Unsupported dataframes type. "
            "Supply a pd.DataFrame, Dict[Any, pd.DataFrame], or  Dict[Any, Dict[Any, pd.DataFrame]]."
        )


def get_clean_data(
    clean_data: List[float] | pd.Series | NDArray[np.float_] | pd.DataFrame,
    value_name: Optional[str] = "values",
):
    if isinstance(clean_data, list):
        return np.array(clean_data)
    elif isinstance(clean_data, np.ndarray):
        return clean_data
    elif isinstance(clean_data, pd.DataFrame):
        return clean_data.loc[:, value_name].to_numpy()
    else:
        raise TypeError("Unsupported clean_data type.")


def get_deltas(
    dataframes: (
        pd.DataFrame | Dict[Any, pd.DataFrame] | Dict[Any, Dict[Any, pd.DataFrame]]
    ),
    clean_data: List[float] | pd.Series | NDArray[np.float_] | pd.DataFrame,
    value_name: str = "values",
):
    if isinstance(dataframes, pd.DataFrame):
        clean_data = get_clean_data(clean_data=clean_data, value_name=value_name)
        results = dataframes.loc[:, value_name] - clean_data
        results.index.names = ["compound_#"] + results.index.names[1:]
        return results
    elif isinstance(dataframes, dict):
        return wrapper_dataframes(
            get_deltas,
            dataframes=dataframes,
            clean_data=clean_data,
            value_name=value_name,
        )
    else:
        raise TypeError(
            "Unsupported dataframes type. "
            "Supply a pd.DataFrame, Dict[Any, pd.DataFrame], or  Dict[Any, Dict[Any, pd.DataFrame]]."
        )


def calc_stats_basic_prediction(
    dataframes: pd.DataFrame | Dict[Any, pd.DataFrame],
    model_names: str | List[str] = MODEL_NAMES,
    clean_data: Optional[
        List[float] | pd.Series | NDArray[np.float_] | pd.DataFrame
    ] = None,
    value_name: str = "values",
    data_offset: float = DEFAULT_DATA_OFFSET,
    **kwargs,
) -> pd.DataFrame:
    if isinstance(dataframes, pd.DataFrame):
        if isinstance(clean_data, pd.DataFrame):
            ref_data = clean_data.loc[:, value_name].to_numpy()
        elif isinstance(clean_data, (np.ndarray, list, pd.Series)):
            ref_data = np.array(clean_data)
        else:
            ref_data = dataframes.loc[:, value_name].to_numpy()

        preds_median = dataframes.loc[:, model_names].median(axis=1)
        preds_mean = dataframes.loc[:, model_names].mean(axis=1)
        preds_std = dataframes.loc[:, model_names].std(axis=1)
        delta_mean_abs = np.abs(ref_data - preds_mean)
        delta_mean_rel = np.abs(delta_mean_abs / (preds_mean + data_offset))
        delta_median_abs = np.abs(ref_data - preds_median)
        delta_median_rel = np.abs(delta_median_abs / (preds_median + data_offset))

        idx = dataframes.index
        idx.names = ["compound_#"] + idx.names[1:]
        results = pd.DataFrame(
            {
                "p_mean": preds_mean,
                "p_median": preds_median,
                "p_std": preds_std,
                "d_mean_a": delta_mean_abs,
                "d_mean_r": delta_mean_rel,
                "d_median_a": delta_median_abs,
                "d_median_r": delta_median_rel,
            },
            index=idx,
        )
        return results
    elif isinstance(dataframes, dict):
        return wrapper_dataframes(
            calc_stats_basic_prediction,
            dataframes=dataframes,
            model_names=model_names,
            clean_data=clean_data,
            value_name=value_name,
            data_offset=data_offset,
            **kwargs,
        )
    else:
        raise TypeError(
            "Unsupported dataframes type. Supply a pd.DataFrame or a dict of pd.Dataframes."
        )


def calc_stats_models(
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
) -> pd.DataFrame:
    if isinstance(dataframes, pd.DataFrame):
        p_median = dataframes.loc[:, model_names].median(axis=1)
        if isinstance(clean_data, pd.DataFrame):
            clean_data = np.array(clean_data.loc[:, value_name])
        elif isinstance(clean_data, (list, np.ndarray, pd.Series)):
            clean_data = np.array(clean_data)
        else:
            clean_data = dataframes.loc[:, value_name]

        deltas = []
        for i, model in enumerate(model_names):
            model_deltas = []
            delta = dataframes.loc[:, model] - clean_data
            delta_abs = np.abs(delta)
            delta_rel = np.abs(delta_abs / (p_median + data_offset))

            model_deltas.append(pd.Series(delta, name="delta"))
            model_deltas.append(pd.Series(delta_abs, name="delta_abs"))
            model_deltas.append(pd.Series(delta_rel, name="delta_rel"))
            model_delta = pd.concat(model_deltas, axis=1)
            model_delta["model_name"] = model
            deltas.append(model_delta)
        results = pd.concat(deltas)
        results.index.names = ["compound_#"] + results.index.names[1:]
        return results
    elif isinstance(dataframes, dict):
        return wrapper_dataframes(
            calc_stats_models,
            dataframes=dataframes,
            clean_data=clean_data,
            model_names=model_names,
            data_offset=data_offset,
            value_name=value_name,
            **kwargs,
        )
    else:
        raise TypeError(
            "Unsupported dataframes type. Supply a pd.DataFrame or a dict of pd.Dataframes."
        )


def calc_stats_groups(
    dataframes: pd.DataFrame | Dict[Any, pd.DataFrame],
    model_names: str | List[str] = MODEL_NAMES,
    thresholds: (
        float | List[float] | NDArray[np.float_] | pd.Series
    ) = DEFAULT_THRESHOLD,
    data_offsets: (
        float | List[float] | NDArray[np.float_] | pd.Series
    ) = DEFAULT_DATA_OFFSET,
    value_name: str = "values",
    **kwargs,
) -> pd.DataFrame:
    if isinstance(dataframes, pd.DataFrame):
        if isinstance(thresholds, (float, int)):
            if isinstance(data_offsets, (float, int)):
                p_median = dataframes.loc[:, model_names].median(axis=1)

                bins = np.linspace(
                    np.quantile(dataframes.loc[:, value_name], 0.01),
                    np.quantile(dataframes.loc[:, value_name], 0.99),
                    25,
                )
                bin_num = np.digitize(dataframes.loc[:, value_name], bins)
                d_median_a = np.abs(p_median - dataframes.loc[:, value_name])
                bin_med = pd.Series(d_median_a).groupby(bin_num).median().to_numpy()
                bin_med = bin_med[bin_num]

                sna, snr, snb, snn = (
                    np.zeros(len(dataframes)),
                    np.zeros(len(dataframes)),
                    np.zeros(len(dataframes)),
                    np.zeros(len(dataframes)),
                )

                for i, model in enumerate(model_names):
                    delta_abs = np.abs(
                        dataframes.loc[:, model] - dataframes.loc[:, value_name]
                    )
                    delta_rel = np.abs(delta_abs / (p_median + data_offsets))
                    delta_norm = np.abs(delta_abs / bin_med)

                    mask_abs = delta_abs > np.percentile(delta_abs, 100 - thresholds)
                    sna += mask_abs

                    mask_rel = delta_rel > np.percentile(delta_rel, 100 - thresholds)
                    snr += mask_rel

                    mask_both = mask_abs & mask_rel
                    snb += mask_both

                    mask_norm = delta_norm > np.percentile(delta_norm, 100 - thresholds)
                    snn += mask_norm
                results = pd.DataFrame(
                    {
                        f"s_{len(model_names)}_a": sna,
                        f"s_{len(model_names)}_r": snr,
                        f"s_{len(model_names)}_b": snb,
                        f"s_{len(model_names)}_n": snn,
                    },
                    index=dataframes.index,
                )
                results.index.names = ["compound_#"] + results.index.names[1:]
                return results
            elif isinstance(data_offsets, (list, np.ndarray, pd.Series)):
                return wrapper_data_offsets(
                    calc_stats_groups,
                    data_offsets=data_offsets,
                    dataframes=dataframes,
                    model_names=model_names,
                    thresholds=thresholds,
                    value_name=value_name,
                    **kwargs,
                )
            else:
                raise ValueError(
                    "Unsupported data_offsets type. Use an Iterable or an ArrayLike or a float."
                )
        elif isinstance(thresholds, (list, np.ndarray, pd.Series)):
            return wrapper_thresholds(
                calc_stats_groups,
                thresholds=thresholds,
                dataframes=dataframes,
                model_names=model_names,
                data_offsets=data_offsets,
                value_name=value_name,
                **kwargs,
            )
        else:
            raise ValueError(
                "Unsupported thresholds type. Use an Iterable or an ArrayLike or a float."
            )
    elif isinstance(dataframes, dict):
        return wrapper_dataframes(
            calc_stats_groups,
            dataframes=dataframes,
            model_names=model_names,
            thresholds=thresholds,
            data_offsets=data_offsets,
            **kwargs,
        )
    else:
        raise TypeError(
            "Unsupported dataframes type. Supply a pd.DataFrame or a dict of pd.Dataframes."
        )


# TODO: MERGE TWO FUNCTIONS BELOW


def calc_stats_models_corr(
    dataframes: pd.DataFrame | Dict[Any, pd.DataFrame],
    model_names: List[str],
    stat_name: Literal["delta", "abs_delta", "rel_delta"] = "delta",
    data_offset: float = DEFAULT_DATA_OFFSET,
    **kwargs,
) -> pd.DataFrame:
    stats_suffixes = {"delta": "", "abs_delta": "_a", "rel_delta": "_r"}
    if isinstance(dataframes, pd.DataFrame):
        stats_df = calc_stats_models(
            dataframe=dataframes,
            model_names=model_names,
            data_offset=data_offset,
            **kwargs,
        )
        stats_df = stats_df[
            sorted([f"{x}{stats_suffixes.get(stat_name, '')}" for x in model_names])
        ]
        return stats_df.corr()
    elif isinstance(dataframes, dict):
        results = {}
        for key, dataframe in dataframes.items():
            results[key] = calc_stats_models_corr(
                dataframes=dataframe,
                model_names=model_names,
                stat_name=stat_name,
                data_offset=data_offset,
                **kwargs,
            )
        return pd.concat(results)
    else:
        raise TypeError(
            "Unsupported dataframes type. Supply a pd.DataFrame or a dict of pd.Dataframes."
        )


def calc_delta_corr(
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
) -> pd.DataFrame:
    stats_df = calc_stats_models(
        dataframes=dataframes,
        clean_data=clean_data,
        model_names=model_names,
        data_offset=data_offset,
        value_name=value_name,
        **kwargs,
    )
    stats_df = stats_df.reset_index()

    idx_vars = ["compound_#", "modified_%", "var_add"]
    idx_vars = [x for x in idx_vars if x in stats_df.columns]
    stats_df = stats_df.pivot(index=idx_vars, columns="model_name", values="delta")
    if len(idx_vars) > 1:
        grouped = stats_df.groupby(level=idx_vars[1:], group_keys=True)
        return grouped.corr()
    else:
        return stats_df.corr()


def calc_stats_classification(
    dataframes: (
        pd.DataFrame | Dict[Any, pd.DataFrame] | Dict[Any, Dict[Any, pd.DataFrame]]
    ),
    clean_dataframe: pd.DataFrame,
    model_names: str | List[str] = MODEL_NAMES,
    thresholds: float | List[float] | NDArray[np.float_] = DEFAULT_THRESHOLD,
    data_offsets: float | List[float] | NDArray[np.float_] = DEFAULT_DATA_OFFSET,
    value_name: str = "values",
    group_stat: Literal["abs", "rel", "both", "norm"] = "abs",
    recursive: bool = False,
    **kwargs,
) -> pd.DataFrame:
    gstat_variants = {
        "abs": f"s_{len(model_names)}_a",
        "rel": f"s_{len(model_names)}_r",
        "both": f"s_{len(model_names)}_b",
        "norm": f"s_{len(model_names)}_n",
    }
    if isinstance(dataframes, pd.DataFrame):
        if isinstance(thresholds, (float, int)):
            if isinstance(data_offsets, (float, int)):
                group_stats = calc_stats_groups(
                    dataframes=dataframes,
                    model_names=model_names,
                    thresholds=thresholds,
                    value_name=value_name,
                    data_offsets=data_offsets,
                )
                modified = get_ground_truth(
                    dataframes=dataframes,
                    clean_dataframe=clean_dataframe,
                    value_name=value_name,
                )
                gstat = gstat_variants.get(group_stat, f"s_{len(model_names)}_a")
                if not recursive:
                    results = {}
                    for i in range(len(model_names) + 1):
                        group_len = np.count_nonzero(group_stats[gstat] == i)
                        all_positive = np.count_nonzero(modified)
                        errros_in_group = np.count_nonzero(
                            (group_stats[gstat] == i) & (modified)
                        )
                        recall = errros_in_group / (all_positive + TOL)
                        precision = errros_in_group / (group_len + TOL)
                        f1 = 2 * precision * recall / (precision + recall + TOL)
                        results[i] = pd.Series(
                            {
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                                "group_len": group_len,
                                "group_ratio": group_len / len(modified),
                                "group_true_ratio": errros_in_group / len(modified),
                            }
                        )
                else:
                    results = {}
                    all_positive = np.count_nonzero(modified)
                    temp_df = dataframes.copy()
                    temp_clean_df = clean_dataframe.copy()
                    group_len_rec = 0
                    errors_in_group_rec = 0
                    steps = 1
                    for i in range(50):
                        modified = get_ground_truth(
                            dataframes=temp_df,
                            clean_dataframe=temp_clean_df,
                            value_name=value_name,
                        )
                        group_stats = calc_stats_groups(
                            dataframes=temp_df,
                            model_names=model_names,
                            thresholds=thresholds,
                            data_offsets=data_offsets,
                            value_name=value_name,
                        )
                        group_len = np.count_nonzero(
                            group_stats[gstat] == len(model_names)
                        )
                        errros_in_group = np.count_nonzero(
                            (group_stats[gstat] == len(model_names)) & (modified)
                        )
                        group_len_rec += group_len
                        errors_in_group_rec += errros_in_group
                        steps += 1

                        if group_len <= 1:
                            steps = i
                            break
                        # elif precision_stop is not None:
                        #     if errros_in_group / (group_len + TOL) < precision_stop:
                        #         break

                        temp_df = temp_df.loc[
                            group_stats[gstat] < len(model_names), :
                        ].copy()
                        temp_clean_df = temp_clean_df.loc[
                            group_stats[gstat] < len(model_names), :
                        ].copy()

                    recall = errors_in_group_rec / (all_positive + TOL)
                    precision = errors_in_group_rec / (group_len_rec + TOL)
                    f1 = 2 * precision * recall / (precision + recall + TOL)
                    results[len(model_names)] = pd.Series(
                        {
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "group_len": group_len_rec,
                            "group_ratio": group_len_rec / len(modified),
                            "group_true_ratio": errors_in_group_rec / len(modified),
                        }
                    )
                results = pd.concat(results, axis=1).T
                results.index.names = ["group_#"] + results.index.names[1:]
                return results

            elif isinstance(data_offsets, (list, np.ndarray, pd.Series)):
                if group_stat == "abs" or group_stat == "norm":
                    results = calc_stats_classification(
                        dataframes=dataframes,
                        clean_dataframe=clean_dataframe,
                        model_names=model_names,
                        thresholds=thresholds,
                        data_offsets=DEFAULT_DATA_OFFSET,
                        value_name=value_name,
                        group_stat=group_stat,
                        recursive=recursive,
                        **kwargs,
                    )
                    return results
                else:
                    results = {}
                    for data_offset in data_offsets:
                        results[data_offset] = calc_stats_classification(
                            dataframes=dataframes,
                            clean_dataframe=clean_dataframe,
                            model_names=model_names,
                            thresholds=thresholds,
                            data_offsets=data_offset,
                            value_name=value_name,
                            group_stat=group_stat,
                            recursive=recursive,
                            **kwargs,
                        )
                    results = pd.concat(results)
                    results.index.names = ["data_offset"] + results.index.names[1:]
                    return results
            else:
                raise TypeError(
                    "Unsupported data_offsets type. Supply a list, tuple or ArrayLike of floats."
                )
        elif isinstance(thresholds, (list, np.ndarray, pd.Series)):
            results = {}
            for threshold in thresholds:
                results[threshold] = calc_stats_classification(
                    dataframes=dataframes,
                    clean_dataframe=clean_dataframe,
                    model_names=model_names,
                    thresholds=threshold,
                    data_offsets=data_offsets,
                    value_name=value_name,
                    group_stat=group_stat,
                    recursive=recursive,
                    **kwargs,
                )
            results = pd.concat(results)
            results.index.names = ["threshold_%"] + results.index.names[1:]
            return results
        else:
            raise TypeError(
                "Unsupported thresholds type. Supply a list, tuple or ArrayLike of floats."
            )

    elif isinstance(dataframes, dict):
        results = {}
        new_name = "dict_name"
        for key, dataframe in dataframes.items():
            results[key] = calc_stats_classification(
                dataframes=dataframe,
                clean_dataframe=clean_dataframe,
                model_names=model_names,
                thresholds=thresholds,
                data_offsets=data_offsets,
                group_stat=group_stat,
                value_name=value_name,
                recursive=recursive,
                **kwargs,
            )
            if isinstance(dataframe, dict):
                new_name = "var_add"
            elif isinstance(dataframe, pd.DataFrame):
                new_name = "modified_%"
        results = pd.concat(results)
        results.index.names = [new_name] + results.index.names[1:]
        return results
    else:
        raise TypeError(
            "Unsupported dataframes type. Supply a pd.DataFrame or a dict of pd.Dataframes."
        )


def calc_stats_classification_multi_model_base(
    dataframes: pd.DataFrame,
    clean_dataframe: pd.DataFrame,
    model_names: List[str],
    data_offset: float = DEFAULT_DATA_OFFSET,
):
    gnn_stats = {}
    for i in tqdm(range(1, len(model_names) + 1)):
        combos = {}
        for combo in combinations(model_names, i):
            stats = calc_stats_classification(
                dataframes=dataframes,
                clean_dataframe=clean_dataframe,
                model_names=combo,
                data_offset=data_offset,
            )
            combos[";".join([str(x) for x in combo])] = stats.loc[len(combo), :]
        combos = pd.concat(combos, axis=1).T
        combos.index.names = ["model_combo"] + combos.index.names[1:]
        gnn_stats[i] = combos
    gnn_stats = pd.concat(gnn_stats)
    gnn_stats.index.names = ["n_models"] + gnn_stats.index.names[1:]
    gnn_stats = gnn_stats.reset_index()
    return gnn_stats


def calc_stats_classification_multi_model(
    dataframes: pd.DataFrame,
    clean_dataframe: pd.DataFrame,
    model_names: List[str],
    data_offset: float = DEFAULT_DATA_OFFSET,
    thresholds: float | List[float] | NDArray[np.float_] = DEFAULT_THRESHOLDS,
    criterium: str = "max",
):
    gnn_stats = calc_stats_classification_multi_model_base(
        dataframes=dataframes,
        clean_dataframe=clean_dataframe,
        model_names=model_names,
        data_offset=data_offset,
    )

    gnn_best = {}
    for i in range(1, len(model_names) + 1):
        gnn_slice = gnn_stats.loc[gnn_stats["n_models"] == i, :]
        if criterium == "max":
            gnn_best[i] = gnn_slice.iloc[np.argmax(gnn_slice["f1"])]
        elif criterium == "median":
            gnn_best[i] = gnn_slice.iloc[
                np.argmin(np.abs(gnn_slice["f1"] - gnn_slice["f1"].median()))
            ]

    gnn_best = pd.concat(gnn_best, axis=1).T

    plot_data = {}
    for i, row in gnn_best.iterrows():
        model_names = row["model_combo"].split(";")
        res = calc_stats_classification(
            dataframes=dataframes,
            clean_dataframe=clean_dataframe,
            model_names=model_names,
            thresholds=thresholds,
            data_offsets=data_offset,
            group_stat="abs",
        )
        res = res.reset_index()
        res = res.loc[res["group_#"] == len(model_names)]
        res["model_names"] = row["model_combo"]
        plot_data[row["n_models"]] = res
    plot_data = pd.concat(plot_data)
    return plot_data


def calc_stats_classification_threshold(
    dataframes,
    clean_data,
    thresholds,
    model_names=MODEL_NAMES,
    data_offsets=DEFAULT_DATA_OFFSET,
    value_name="values",
):
    group_stats = calc_stats_groups(
        dataframes=dataframes,
        model_names=model_names,
        thresholds=DEFAULT_THRESHOLD,
        data_offsets=data_offsets,
    )
    basic_stats = calc_stats_basic_prediction(
        dataframes=dataframes,
        model_names=model_names,
        value_name=value_name,
        data_offset=data_offsets,
    )
    modified = get_ground_truth(
        dataframes=dataframes, clean_dataframe=clean_data, value_name=value_name
    )
    stats = pd.concat([group_stats, basic_stats, modified], axis=1).reset_index()

    results = {}
    for t in thresholds:
        temp = (
            stats[stats["d_median_a"] > t * stats["d_median_a"].mean()]
            .groupby("modified_%")["modified"]
            .value_counts()
            .to_frame()
        )
        temp["type"] = "plain"
        temp = temp.set_index("type", append=True)
        temp2 = (
            stats[
                (stats["d_median_a"] > t * stats["d_median_a"].mean())
                & (stats["s_5_a"] == 5)
            ]
            .groupby("modified_%")["modified"]
            .value_counts()
            .to_frame()
        )
        temp2["type"] = "plain + grouped"
        temp2 = temp2.set_index("type", append=True)
        results[t] = pd.concat([temp, temp2])
    results = pd.concat(results)
    results.index.names = ["threshold"] + results.index.names[1:]
    # return results

    plot_data = pd.DataFrame()
    plot_data["precision"] = (
        results.xs(True, level="modified")["count"]
        / results.groupby(["threshold", "modified_%", "type"]).sum()["count"]
    )
    plot_data["recall"] = (
        results.xs(True, level="modified")["count"]
        / modified.groupby("modified_%").sum().to_frame()["modified"]
    )
    plot_data["f1"] = (
        2
        * plot_data["precision"]
        * plot_data["recall"]
        / (plot_data["precision"] + plot_data["recall"] + 1e-8)
    )
    plot_data = plot_data.reset_index()  # .drop(labels="modified")

    temp1 = calc_stats_classification(
        dataframes=dataframes,
        clean_dataframe=clean_data,
        model_names=model_names,
        thresholds=DEFAULT_THRESHOLDS,
    )
    temp1["type"] = "grouped"
    temp1 = temp1.set_index("type", append=True)
    temp1 = temp1.xs(5, level="group_#").reset_index()
    plot_data = pd.concat([plot_data, temp1])
    return plot_data


def calc_stats_comparison(
    dataframes,
    clean_data,
    model_names=MODEL_NAMES,
    data_offsets=DEFAULT_DATA_OFFSET,
    value_name="values",
):
    group_stats = calc_stats_groups(
        dataframes=dataframes,
        model_names=model_names,
        thresholds=DEFAULT_THRESHOLD,
        data_offsets=data_offsets,
    )
    basic_stats = calc_stats_basic_prediction(
        dataframes=dataframes,
        model_names=model_names,
        value_name=value_name,
        data_offset=data_offsets,
    )
    modified = get_ground_truth(
        dataframes=dataframes, clean_dataframe=clean_data, value_name=value_name
    )
    stats = pd.concat([group_stats, basic_stats, modified], axis=1).reset_index()
    idx_vars = ["var_add", "modified_%"]
    idx_vars = [x for x in idx_vars if x in stats.columns]
    # results = {}
    # for t in np.geomspace(0.001, 100, 100):
    #     temp = (
    #         stats[stats["d_median_a"] > t * stats["d_median_a"].mean()]
    #         .groupby(idx_vars)["modified"]
    #         .value_counts()
    #         .to_frame()
    #     )
    #     temp["type"] = "abs_error"
    #     temp = temp.set_index("type", append=True)
    #     results[t] = temp
    # results = pd.concat(results)
    # results.index.names = ["threshold"] + results.index.names[1:]

    results1 = {}
    for t in np.concatenate(
        [np.geomspace(0.01, 90, 50), 100 - np.geomspace(0.0001, 10, 50)]
    ):
        temp = (
            stats[stats["d_median_a"] > stats["d_median_a"].quantile(t / 100)]
            .groupby(idx_vars)["modified"]
            .value_counts()
            .to_frame()
        )
        temp["type"] = "Quantile"
        temp = temp.set_index("type", append=True)
        results1[t] = temp
    results1 = pd.concat(results1)
    results1.index.names = ["threshold"] + results1.index.names[1:]
    # return results
    # results = pd.concat([results,results1])
    results = results1

    plot_data = pd.DataFrame()
    plot_data["precision"] = (
        results.xs(True, level="modified")["count"]
        / results.groupby(idx_vars + ["threshold", "type"]).sum()["count"]
    )
    plot_data["recall"] = (
        results.xs(True, level="modified")["count"]
        / modified.groupby(idx_vars).sum().to_frame()["modified"]
    )
    plot_data["f1"] = (
        2
        * plot_data["precision"]
        * plot_data["recall"]
        / (plot_data["precision"] + plot_data["recall"] + 1e-8)
    )
    plot_data = plot_data.reset_index()  # .drop(labels="modified")

    temp1 = calc_stats_classification(
        dataframes=dataframes,
        clean_dataframe=clean_data,
        model_names=model_names,
        thresholds=np.geomspace(0.0001, 100, 100),
    )
    temp1["type"] = "Yellow Cards"
    temp1 = temp1.set_index("type", append=True)
    temp1 = temp1.xs(5, level="group_#").reset_index()
    temp1["threshold"] = temp1["threshold_%"]
    plot_data = pd.concat([plot_data, temp1])
    return plot_data
