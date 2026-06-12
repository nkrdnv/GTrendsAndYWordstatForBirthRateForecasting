from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults


@dataclass(frozen=True)
class SarimaxSpec:
    """
    Спецификация SARIMAX модели (параметры конструктора statsmodels.SARIMAX).
    """

    order: Tuple[int, int, int] = (1, 0, 0)  # (p, d, q)
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (P, D, Q, s)
    trend: Optional[str] = None  # 'n','c','t','ct' или None
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True
    mle_regression: bool = True
    # Можно расширять при необходимости другими параметрами SARIMAX


def fit_and_compute_sarimax(
    spec: Union[SarimaxSpec, Dict[str, Any]],
    endog: Union[pd.Series, pd.DataFrame],
    exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
    *,
    predict_start: Optional[Union[str, pd.Timestamp]] = None,
    predict_end: Optional[Union[str, pd.Timestamp]] = None,
    forecast_steps: int = 0,
    exog_future: Optional[Union[pd.Series, pd.DataFrame]] = None,
    p_value: float = 0.05,
    dropna: bool = True,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Обучает SARIMAX на временных рядах (endog, exog) и вычисляет:
      - in-sample прогноз (predict / get_prediction)
      - out-of-sample прогноз (get_forecast), если forecast_steps > 0

    Вход:
      spec         : SarimaxSpec или dict с аргументами для SARIMAX(...)
      endog        : pd.Series (или 1-колоночный DataFrame) с datetime-like индексом
      exog         : pd.DataFrame/Series с тем же индексом, что и endog (по времени)
      predict_start/predict_end : границы in-sample предсказания (по индексу)
      forecast_steps : горизонт прогноза вперед (кол-во шагов)
      exog_future  : экзогенные признаки на период прогноза (обязательны, если exog задан)
      p_value        : уровень значимости для доверительных интервалов
      dropna       : если True — удаляет строки, где есть NaN в endog или exog
      fit_kwargs   : параметры для results = model.fit(**fit_kwargs)

    Выход: dict со следующими ключами:
      - "model"      : объект SARIMAX
      - "results"    : SARIMAXResults
      - "insample"   : DataFrame с колонками ["mean", "lower", "upper"] (может быть None)
      - "forecast"   : DataFrame с колонками ["mean", "lower", "upper"] (может быть None)
    """
    if isinstance(spec, dict):
        spec_kwargs = dict(spec)
    else:
        spec_kwargs = spec.__dict__.copy()

    if isinstance(endog, pd.DataFrame):
        if endog.shape[1] != 1:
            raise ValueError("endog должен быть pd.Series или DataFrame с 1 колонкой.")
        endog = endog.iloc[:, 0]
    elif not isinstance(endog, pd.Series):
        raise TypeError("endog должен быть pd.Series или DataFrame с 1 колонкой.")

    # --- Проверка индекса ---
    if not isinstance(endog.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise TypeError(
            "endog должен иметь datetime-like индекс (DatetimeIndex или PeriodIndex)."
        )

    # --- Приведение exog к DataFrame ---
    if exog is not None:
        if isinstance(exog, pd.Series):
            exog = exog.to_frame()
        elif not isinstance(exog, pd.DataFrame):
            raise TypeError("exog должен быть pd.Series или pd.DataFrame.")

        if not exog.index.equals(endog.index):
            # безопасное выравнивание по пересечению индексов
            common_idx = endog.index.intersection(exog.index)
            endog = endog.loc[common_idx]
            exog = exog.loc[common_idx]

    if dropna:
        if exog is None:
            endog = endog.dropna()
        else:
            df = pd.concat([endog.rename("endog"), exog], axis=1).dropna()
            endog = df["endog"]
            exog = df.drop(columns=["endog"])

    # --- Построение и обучение модели ---
    model = SARIMAX(endog=endog, exog=exog, **spec_kwargs)

    fit_kwargs = fit_kwargs or {}
    fit_kwargs.setdefault("disp", False)

    results: SARIMAXResults = model.fit(**fit_kwargs)

    # --- In-sample prediction ---
    insample_df = None
    if predict_start is not None or predict_end is not None:
        pred = results.get_prediction(
            start=predict_start,
            end=predict_end,
            exog=exog,  # in-sample, уже выровнено
        )
        mean = pred.predicted_mean.rename("mean")
        ci = pred.conf_int(alpha=p_value)
        # В statsmodels названия колонок CI зависят от имени ряда, что не удобно
        ci = ci.rename(columns={ci.columns[0]: "lower", ci.columns[1]: "upper"})
        insample_df = pd.concat([mean, ci], axis=1)
    forecast_df = None
    if forecast_steps and forecast_steps > 0:
        if exog is not None and exog_future is None:
            raise ValueError(
                "Для прогноза необходимо передать exog_future, т.к. модель обучалась с exog."
            )

        if exog_future is not None:
            if isinstance(exog_future, pd.Series):
                exog_future = exog_future.to_frame()
            if not isinstance(exog_future, pd.DataFrame):
                raise TypeError("exog_future должен быть pd.Series или pd.DataFrame.")
            if len(exog_future) != forecast_steps:
                raise ValueError("Длина exog_future должна совпадать с forecast_steps.")

        fc = results.get_forecast(steps=forecast_steps, exog=exog_future)
        mean = fc.predicted_mean.rename("mean")
        ci = fc.conf_int(alpha=p_value)
        ci = ci.rename(columns={ci.columns[0]: "lower", ci.columns[1]: "upper"})
        forecast_df = pd.concat([mean, ci], axis=1)

    return {
        "model": model,
        "results": results,
        "insample": insample_df,
        "forecast": forecast_df,
    }


def apply_out_of_domain(
    fitted_model: SARIMAXResults,
    forecast_steps: int,
    ENDOG: Union[pd.DataFrame, pd.Series],
    EXOG: Optional[Union[pd.DataFrame, pd.Series]] = None,
    return_train_preds: bool = False,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Применяет обученную модель на новых данных. Может возвращать предсказания и прогнозы.

    Вход:
      fitted_model       : SARIMAXResults -- обученная SARIMAX модель.
      ENDOG              : pd.Series (или 1-колоночный DataFrame) с datetime-like индексом
      EXOG               : pd.DataFrame/Series с тем же индексом, что и endog (по времени)
      forecast_steps     : горизонт прогноза вперед (кол-во шагов)
      return_train_preds : возвращать ли предсказания для новых входных данных

    Выход: пара опциональных pd.DataFrame, соответствующих insample и outsample предсказаниям
    """
    if EXOG is not None:
        if isinstance(EXOG, pd.Series):
            EXOG = EXOG.to_frame()
        elif not isinstance(EXOG, pd.DataFrame):
            raise TypeError("EXOG должен быть pd.Series или pd.DataFrame.")
        if not EXOG.index.equals(ENDOG.index):
            # безопасное выравнивание по началу индексов для сохранения возможности предсказания
            common_idx_start = ENDOG.index.intersection(EXOG.index)[0]
            ENDOG = ENDOG.loc[common_idx_start:]
            EXOG = EXOG.loc[common_idx_start:]

    model = fitted_model.model.clone(
        endog=ENDOG,
        exog=EXOG,
    )
    param_dict = dict(zip(fitted_model.param_names, fitted_model.params))
    with model.fix_params(param_dict):
        refitted_model = model.fit()
    if forecast_steps == 0:
        return None, refitted_model.predict()
    if return_train_preds:
        return refitted_model.forecast(forecast_steps), refitted_model.predict()
    return refitted_model.forecast(forecast_steps), None


def mape(y_true, y_pred, zero_division=1e-7):
    return np.mean(
        np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), zero_division)))
    )


# ------------------ Пример использования ------------------
# if __name__ == "__main__":
#     # y: pd.Series, X: pd.DataFrame с DatetimeIndex
#     spec = SarimaxSpec(order=(1,1,1), seasonal_order=(1,0,1,12), trend="c")
#     out = fit_and_compute_sarimax(spec, endog=y, exog=X, predict_start=y.index[0], predict_end=y.index[-1])
#     print(out["results"].summary())S
