from __future__ import annotations

from dataclasses import dataclass

import openml
import pandas as pd
from openml import OpenMLDataset
from sklearn.preprocessing import LabelEncoder


@dataclass
class Dataset:
    name: str
    id: int
    features: pd.DataFrame
    labels: pd.DataFrame
    openml: OpenMLDataset
    encoders: dict[str, LabelEncoder]

    @staticmethod
    def from_openml(id: int) -> Dataset:
        """Processes an binary classification OpenMLDataset into its features and targets

        Parameters
        ----------
        id: int
            The id of the dataset

        Returns
        -------
        Dataset
        """
        dataset = openml.datasets.get_dataset(id)
        target = dataset.default_target_attribute
        data, _, _, _ = dataset.get_data()

        assert isinstance(data, pd.DataFrame)

        # Process the features and turn all categorical columns into ints
        features = data.drop(columns=target)
        encoders: dict[str, LabelEncoder] = {}

        for name, col in features.items():
            if col.dtype in ["object", "category", "string"]:
                encoder = LabelEncoder()
                features[name] = encoder.fit_transform(col)
                encoders[name] = encoder

        labels = data[target]

        # Since we assume binary classification, we convert the labels
        # labels = labels.astype(bool)

        return Dataset(
            name=dataset.name,
            id=id,
            features=features,
            labels=labels,
            openml=dataset,
            encoders=encoders,
        )
