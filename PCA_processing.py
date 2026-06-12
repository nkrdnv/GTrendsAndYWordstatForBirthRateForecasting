from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from collections import namedtuple

from typing import (
    Optional,
    Union,
    Tuple,
    Dict,
    Any,
    Callable,
    NamedTuple,
    List,
    Protocol,
)


class Normalizator(Protocol):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Производит нормализацию входных данных
        ...

    def inverse(self, x: np.ndarray) -> np.ndarray:
        # Восстанавливает исходные данные из нормализованных
        # this.inverse(this(x)) == x
        ...


class mean_normalizator:
    def __init__(self, x: np.ndarray, normalization: float = 15):
        self.mean = np.mean(x) + normalization

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.mean

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.mean + self.mean


def find(arr, elem, lam: Callable[[Any], Any] = lambda x: x):
    for i in arr:
        if lam(i) == elem:
            return i
    return None


def arg_find(arr, elem, lam: Callable[[Any], Any] = lambda x: x):
    for i in range(len(arr)):
        if lam(arr[i]) == elem:
            return i
    return None


def weekly_to_monthly_summator(weekly_df: pd.DataFrame) -> pd.DataFrame:
    monthly_df = weekly_df.resample("ME").sum()
    monthly_df -= (
        (
            (
                weekly_df
                * (((weekly_df.index.shift(0, "ME") - weekly_df.index).days + 1) / 7)
                .to_numpy()
                .reshape(-1, 1)
            ).iloc[(weekly_df.index.shift(0, "ME") - weekly_df.index).days < 7]
        )
        .shift(-1, "D")
        .shift(1, "ME")
    )
    monthly_df.add(
        (
            (
                weekly_df
                * (
                    1
                    - ((weekly_df.index.shift(0, "ME") - weekly_df.index).days + 1) / 7
                )
                .to_numpy()
                .reshape(-1, 1)
            ).iloc[(weekly_df.index.shift(0, "ME") - weekly_df.index).days < 7]
        )
        .shift(-1, "D")
        .shift(2, "ME"),
        fill_value=0,
    )
    return monthly_df


class Processed_PCAs(NamedTuple):
    models: Dict[
        str, PCA
    ]  # Модели PCA. Ожидаемое применение models[region].transform(transforms[region][phrase](phrase_series))
    components_and_info: Dict[
        str, pd.DataFrame
    ]  # Возвращает max_components главных компонент + центроид для каждого из регионов. Также возвращает в __info объяснённые дисперсии
    pcas_projections: Dict[
        str, pd.DataFrame
    ]  # Спроецированные на плоскость главных компонент наблюдения. Как представляются ряды после "расшумления" методом главных компонент.
    coordinats_in_PCA_basis: Dict[
        str, pd.DataFrame
    ]  # Как получить ряд зная главные компоненты.
    PCA_from_points: Dict[
        str, np.ndarray
    ]  # Как получить главные компоненты как линейную комбинацию точек
    transforms: Dict[
        str, Dict[str, Normalizator]
    ]  # Преобразования входных данных для приведения их к 0 среднему и относительным отклонениям: (99, 100, 101) -> (-1%, 0%, 1%)
    requests: List[str]  # Список запросов
    time_index: pd.Index  # Индекс временного ряда


def process_data(
    structured_time_series: Dict[str, Dict[str, pd.DataFrame]],
    processed_regions: Dict[str, dict[str, Any]],
    processed_requests: List[str],
    column: Optional[str] = None,
    column_normalization: float = 0.000001,
    var_limit: float = 0.85,
    max_components: int = 6,
) -> Processed_PCAs:
    """
    Применяет метод главных компонент (PCA) к нормализованным временным рядам
    заданных запросов по выбранным регионам. Вычисляет модели PCA, проекции,
    координаты в базисах и вспомогательные преобразования.

    Параметры
    ----------
    structured_time_series : dict
        Исходные данные. Структура: {регион: {запрос: DataFrame}}.
        Каждый DataFrame содержит временной ряд, индекс используется как общая
        временная шкала.
    processed_regions : dict
        Словарь интересующих регионов. Ключи - идентификаторы регионов,
        значения - словари, обязательно содержащие ключ 'label' (например,
        {"225": {"label": "Россия"}}).
        'label' используется только для сохранения понимания принадлежности рядов к регионам.
    processed_requests : list of str
        Список названий запросов (ключей), по которым будут взяты ряды.
    column : str или None, optional
        Имя столбца Dataframe внутри запроса в structured_time_series, который следует анализировать.
        Если None, используется весь DataFrame (предполагается один столбец).
        По умолчанию None.
    column_normalization : float, optional
        Параметр регуляризации для нормализатора `mean_normalizator`
        (предотвращает деление на ноль). По умолчанию 1e-6.
    var_limit : float, optional
        Порог накопленной объяснённой дисперсии (от 0 до 1). Число
        компонент выбирается так, чтобы накопленная дисперсия превышала
        этот порог. По умолчанию 0.85.
    max_components : int, optional
        Максимально допустимое число главных компонент. Фактическое число
        не может превышать количество запросов. По умолчанию 6.

    Возвращает
    ----------
    Processed_PCAs
        Именованный кортеж со следующими полями:
        - models : dict {регион: PCA}
            Обученные модели PCA для каждого региона.
        - components_and_info : dict {str: DataFrame}
            Ключ - метка региона; значение - DataFrame, где каждая строка
            (временная метка) содержит коэффициенты главных компонент и
            среднее `mean`. Специальный ключ '__info' хранит накопленные
            объяснённые дисперсии по регионам.
        - pcas_projections : dict {запрос: DataFrame}
            Для каждого запроса - DataFrame с исходным рядом и его
            объяснённой (восстановленной) частью после PCA-сжатия.
        - coordinats_in_PCA_basis : dict {запрос: DataFrame}
            Для каждого запроса - координаты наблюдений в пространстве
            главных компонент (с дополнением NaN до максимального числа
            компонент).
        - PCA_from_points : dict {регион: np.ndarray}
            Матрица перехода от исходных точек к главным компонентам для
            каждого региона.
        - transforms : dict {регион: {запрос: Normalizator}}
            Использованные экземпляры нормализаторов для каждой пары
            (регион, запрос).
        - requests : list of str
            Список обработанных запросов (декодер).
        - time_index : pd.Index
            Общий временной индекс, взятый из первого запроса первого региона.

    Примечания
    ----------
    - Если `column` не указан, предполагается, что DataFrame запроса содержит
      ровно ОДИН столбец.
    - Внутренне `max_components` усекается до `len(interesting_request)`.
    - Для согласованного представления координат используется максимальное
      число компонент среди всех регионов (но не более `max_components`).
    """

    transforms = {}
    index = structured_time_series[list(processed_regions.keys())[0]][
        processed_requests[0]
    ].index
    for region in processed_regions:
        transforms[region] = {}
        for elem in processed_requests:
            if column is None:
                transforms[region][elem] = mean_normalizator(
                    structured_time_series[region][elem].to_numpy(),
                    column_normalization,
                )
            else:
                transforms[region][elem] = mean_normalizator(
                    structured_time_series[region][elem][column].to_numpy(),
                    column_normalization,
                )
    pcas = {}
    data = {}
    info = {}
    pcas_compositions = {}
    mx = -1
    max_components = min(max_components, len(processed_requests))

    for region in processed_regions:
        info[region] = {}
        tmp = PCA()
        kk = []
        decoder = []
        for key in processed_requests:
            elem = structured_time_series[region][key]
            if key == "metainfo":
                continue
            decoder.append(key)
            if column is None:
                kk.append(transforms[region][key](elem.to_numpy()))
            else:
                kk.append(transforms[region][key](elem[column].to_numpy()))
        tmp.fit(kk)
        data[region] = {"decoder": decoder, "data": kk}
        info[region]["len"] = (
            np.cumsum(tmp.explained_variance_ratio_) > var_limit
        ).argmax() + 1
        info[region]["detailed"] = np.cumsum(tmp.explained_variance_ratio_)[
            : info[region]["len"]
        ]
        tmp = PCA(max(info[region]["len"], max_components))
        tmp.fit(kk)
        mx = max(mx, tmp.n_components_)
        pcas[region] = tmp
        pcas_compositions[region] = (
            tmp.transform(kk) / ((tmp.singular_values_) ** 2)
        ).T

    pcas_vectors = {}
    info_df = {}
    for key, elem in pcas.items():
        num_components = info[key]["len"]
        info_df[processed_regions[key]["label"]] = info[key]["detailed"]
        tmp_df = pd.DataFrame(index).set_index(index)
        for i in range(elem.components_.shape[0]):
            tmp_df[str(i)] = elem.components_[i]
        tmp_df["mean"] = elem.mean_
        pcas_vectors[processed_regions[key]["label"]] = tmp_df

    for key, elem in info_df.items():
        info_df[key] = np.pad(
            elem, (0, mx - elem.shape[0]), "constant", constant_values=np.nan
        )

    pcas_vectors["__info"] = pd.DataFrame(info_df)

    pcas_explained = {}
    for elem in decoder:
        tmp_df = pd.DataFrame(index).set_index(index)
        for key, component in pcas.items():
            x = data[key]["data"][arg_find(data[key]["decoder"], elem)]
            tmp_df[processed_regions[key]["label"]] = x
            tmp_df["Объяснённая часть для " + processed_regions[key]["label"]] = (
                component.inverse_transform(component.transform(x.reshape(1, -1)))[0]
            )

        pcas_explained[elem] = tmp_df

    pcas_decompositions = {}
    for elem in decoder:
        tmp_df = {}
        for key, component in pcas.items():
            x = data[key]["data"][arg_find(data[key]["decoder"], elem)]
            tmp_df[processed_regions[key]["label"]] = np.pad(
                component.transform(x.reshape(1, -1))[0],
                (0, mx - component.n_components_),
                "constant",
                constant_values=np.nan,
            )
        tmp_df = pd.DataFrame(tmp_df)
        pcas_decompositions[elem] = tmp_df

    return Processed_PCAs(
        pcas,
        pcas_vectors,
        pcas_explained,
        pcas_decompositions,
        pcas_compositions,
        transforms,
        decoder,
        index,
    )
