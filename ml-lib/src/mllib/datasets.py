from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Generic, Self, TypeVar
import numpy as np
import numpy.typing as npt
import pandas as pd

FeatureType = TypeVar("FeatureType", bound=np.number)
TargetType = TypeVar("TargetType", bound=np.number)


@dataclass
class BaseDataset(Generic[FeatureType, TargetType], ABC):
    data: npt.NDArray[FeatureType]
    target: npt.NDArray[TargetType]
    feature_names: ClassVar[list[str]]
    target_names: ClassVar[list[str]]
    filepath: ClassVar[Path]

    @classmethod
    def load_data(cls) -> Self:
        """Load the dataset."""
        df = pd.read_csv(cls.filepath)

        X = df[cls.feature_names].to_numpy()
        y = cls.process_target(df)

        return cls(data=X, target=y)

    @classmethod
    @abstractmethod
    def process_target(cls, df: pd.DataFrame) -> npt.NDArray[TargetType]:
        pass


class BreastCancerDataset(BaseDataset[np.float64, np.int8]):
    feature_names = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
    ]
    target_names = ["malignant", "benign"]
    filepath = Path(__file__).parent / "data" / "breast_cancer_data.csv"

    @classmethod
    def process_target(cls, df: pd.DataFrame) -> npt.NDArray[np.int8]:
        return np.where(df["diagnosis"] == "M", 1, 0)


class IrisDataset(BaseDataset[np.float16, np.int8]):
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    target_names = ["setosa", "versicolor", "virginica"]
    filepath = Path(__file__).parent / "data" / "iris_data.csv"

    @classmethod
    def process_target(cls, df: pd.DataFrame) -> npt.NDArray[np.int8]:
        label_map = {name: i for i, name in enumerate(cls.target_names)}
        return df["species"].map(lambda s: label_map[s]).to_numpy()


def load_breast_cancer() -> BreastCancerDataset:
    """
    Load the breast cancer dataset.
    """
    return BreastCancerDataset.load_data()


def load_iris() -> IrisDataset:
    """
    Load the iris dataset.
    """
    return IrisDataset.load_data()
