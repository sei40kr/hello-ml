from dataclasses import dataclass
from pathlib import Path
import numpy as np
import numpy.typing as npt
from pandas import read_csv


@dataclass
class Dataset:
    data: npt.NDArray[np.generic]
    target: npt.NDArray[np.generic]
    feature_names: list[str]
    target_names: list[str]


def load_breast_cancer() -> Dataset:
    """
    Load the breast cancer dataset.
    """
    current_dir = Path(__file__).parent
    data_path = current_dir / "data" / "breast_cancer_data.csv"

    df = read_csv(data_path)

    target_column = "diagnosis"
    feature_names = [col for col in df.columns if col != target_column]
    target_names = ["malignant", "benign"]

    data = df[feature_names].to_numpy(dtype=np.float64)
    target = np.where(df[target_column] == "M", 1, 0)

    return Dataset(
        data=data,
        target=target,
        feature_names=feature_names,
        target_names=target_names,
    )
