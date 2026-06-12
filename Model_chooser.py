import numpy as np
import pandas as pd
import copy
import json

from tqdm.auto import tqdm

from functions.PCA_processing import process_data
from functions.SARIMAX_routines import (
    mape,
    fit_and_compute_sarimax,
    SarimaxSpec,
    apply_out_of_domain,
)

import warnings
import statsmodels.tools.sm_exceptions as sm_warnings

import matplotlib.pyplot as plt


def sarimax_pipeline(spec, endog, exog=None, val_size=12):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=sm_warnings.ValueWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        res = fit_and_compute_sarimax(spec=spec, endog=endog[:-val_size], exog=exog)
        return {
            "model": res["model"],
            "result": res["results"],
            "aic": res["results"].aic,
            "predictions": res["results"].predict(),
            "summary": res["results"].summary(),
            "predicts": apply_out_of_domain(res["results"], 0, ENDOG=endog, EXOG=exog),
        }


def __main__():
    spec = SarimaxSpec(order=(2, 1, 1), seasonal_order=(0, 1, 1, 12))
    LAGS = 12
    VAL_SIZE = 12
    NORM = 10_000

    PLT_SIZE = 12
    VAL_START = 60
    out_prec = 3

    data_start = "2019-01-01"

    processed_regions = {"225": {"label": "Россия"}}

    birthes = pd.DataFrame({"Российская Федерация": [1, 2, 3, 4, 5]})
    wordstat_to_birthes = {"Россия": "Российская Федерация"}

    requests_metagroups = ["a"]

    wordstat_requests_to_process = {"a": ["b"]}
    wordstat_structured_data = {"225": {"b": pd.DataFrame([1, 2, 3])}}

    google_requests_to_process = {"a": ["b"]}
    google_structured_data = {"225": {"b": pd.DataFrame([1, 2, 3])}}

    output = "argmins.json"

    unified_index = {}

    unified_index["global"] = {}
    unified_index["global"]["google"] = []
    unified_index["global"]["wordstat"] = []
    for metagroup in requests_metagroups:
        if len(google_requests_to_process.get(metagroup, default=[])) and len(
            wordstat_requests_to_process.get(metagroup, default=[])
        ):
            unified_index[metagroup] = {}

            unified_index[metagroup]["google"] = copy(
                google_requests_to_process[metagroup]
            )
            unified_index[metagroup]["wordstat"] = copy(
                wordstat_requests_to_process[metagroup]
            )

            unified_index["global"]["google"].extend(
                google_requests_to_process[metagroup]
            )
            unified_index["global"]["wordstat"].extend(
                wordstat_requests_to_process[metagroup]
            )

    wordstat_global = {}
    for metagroup in requests_metagroups:
        if metagroup in unified_index:
            result = process_data(
                wordstat_structured_data,
                processed_regions=processed_regions,
                processed_requests=unified_index[metagroup]["wordstat"],
                column="share",
                max_components=1,
            )
            wordstat_global[metagroup] = copy(result)
    wordstat_global["global"] = process_data(
        wordstat_structured_data,
        processed_regions=processed_regions,
        processed_requests=unified_index["global"]["wordstat"],
        column="share",
        max_components=1,
    )

    google_global = {}
    for metagroup in requests_metagroups:
        if metagroup in unified_index:
            result = process_data(
                google_structured_data,
                processed_regions=processed_regions,
                processed_requests=unified_index[metagroup]["google"],
                column="share",
                max_components=1,
            )
            google_global[metagroup] = copy(result)
    google_global["global"] = process_data(
        google_structured_data,
        processed_regions=processed_regions,
        processed_requests=unified_index["global"]["google"],
        column="share",
        max_components=1,
    )

    trends_mixed = {}
    for region in processed_regions:
        label = processed_regions[region]
        trends_mixed[label] = {}
        for key in wordstat_global:
            trends_mixed[label][key] = pd.merge(
                wordstat_global[key].components_and_info[label],
                google_global[key].components_and_info[label],
                how="inner",
                left_index=True,
                right_index=True,
                suffixes=("_wordstat", "_trends"),
            ).drop(columns=["Месяц"])

    compare_data_sources = {}
    for key in processed_regions:
        label = processed_regions[key]["label"]
        if label not in wordstat_to_birthes:
            continue
        birthes_key = wordstat_to_birthes[label]
        if birthes_key in compare_data_sources:
            continue  # Кеширование работы
        compare_data_sources[label] = {}
        for key, elem_df in tqdm(trends_mixed.items()):
            tmp_pair = {}

            elem = elem_df["0_wordstat"]
            tmp = []
            for lags in range(0, LAGS):
                tmp.append(
                    sarimax_pipeline(
                        spec=spec,
                        endog=(birthes[birthes_key] / NORM[label]),
                        exog=elem.shift(lags).iloc[LAGS:],
                        val_size=VAL_SIZE,
                    )
                )
            tmp_pair["wordstat"] = copy(tmp)

            elem = elem_df["0_trends"]
            tmp = []
            for lags in range(0, LAGS):
                tmp.append(
                    sarimax_pipeline(
                        spec=spec,
                        endog=(birthes[birthes_key] / NORM[label]),
                        exog=elem.shift(lags).iloc[LAGS:],
                        val_size=VAL_SIZE,
                    )
                )
            tmp_pair["trends"] = copy(tmp)

            compare_data_sources[label][key] = copy(tmp_pair)
        compare_data_sources[label]["NO_QUERIES"] = copy(
            sarimax_pipeline(
                spec=spec,
                endog=(birthes[birthes_key] / NORM[label]).loc[data_start:],
                exog=None,
                val_size=VAL_SIZE,
            )
        )

    argmins = {}
    reports = []

    for label in compare_data_sources:
        normalized_endog_data = birthes[wordstat_to_birthes[label]] / NORM[label]
        normalized_endog_data.name = label
        const_df = pd.merge(
            normalized_endog_data,
            compare_data_sources[label]["NO_QUERIES"]["predicts"][1][
                VAL_START : VAL_START + PLT_SIZE
            ],
            left_index=True,
            right_index=True,
            how="inner",
        )
        base_mape = mape(const_df[label], const_df["predicted_mean"])
        summary_dict = {
            "baseline": compare_data_sources[label]["NO_QUERIES"]["summary"]
        }
        metastring = "\n".join(
            [
                f"Регион: {label}",
                f"Модель без запросов имеет минимум {base_mape:.{out_prec}g}",
            ]
        )
        reports.append(metastring)

        sheet_dicts = {}
        for key, models in compare_data_sources[label].items():
            argmins[label] = argmins.get(label, {})
            if key == "NO_QUERIES":
                argmins[label][key] = {
                    "wordstat": {"argmin": 0, "min": base_mape},
                    "trends": {"argmin": 0, "min": base_mape},
                }
                continue
            wordstat_mapes = []
            for item in models["wordstat"]:
                birthes_taken = pd.merge(
                    normalized_endog_data,
                    item["predicts"][1][VAL_START : VAL_START + PLT_SIZE],
                    left_index=True,
                    right_index=True,
                    how="inner",
                )
                wordstat_mapes.append(
                    mape(birthes_taken[label], birthes_taken["predicted_mean"])
                )

            trends_mapes = []
            for item in models["trends"]:
                birthes_taken = pd.merge(
                    normalized_endog_data,
                    item["predicts"][1][VAL_START : VAL_START + PLT_SIZE],
                    left_index=True,
                    right_index=True,
                    how="inner",
                )
                trends_mapes.append(
                    mape(birthes_taken[label], birthes_taken["predicted_mean"])
                )

            sheet_dicts[key] = pd.DataFrame(
                {"wordstat": wordstat_mapes, "trends": trends_mapes}
            ).rename_axis("Номер лага", axis="index")

            argmins[label][key] = {
                "wordstat": {
                    "argmin": np.argmin(wordstat_mapes),
                    "min": min(wordstat_mapes),
                },
                "trends": {"argmin": np.argmin(trends_mapes), "min": min(trends_mapes)},
            }

            for data_source, argmin in argmins[label][key].items():
                summary_dict[key + "_" + data_source + "_" + str(argmin)] = models[
                    data_source
                ][argmin]["summary"]

            metastring = "\n".join(
                [
                    f"Регион: {label}",
                    f"Группа: {key}",
                    f"Модель на запросах wordstat достигает минимума на {np.argmin(wordstat_mapes)} лаге со значением {min(wordstat_mapes):.{out_prec}g}",
                    f"Модель на запросах GTrends достигает минимума на {np.argmin(trends_mapes)} лаге со значением {min(trends_mapes):.{out_prec}g}",
                ]
            )
            reports.append(metastring)

    with open(output, "w", encoding="utf-8") as file:
        json.dump(
            {"argmins": argmins, "reports": reports}, file, ensure_ascii=False, indent=4
        )
