import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import anomed_challenge
import anomed_utils
import numpy as np
import pandas as pd
import requests
from art.estimators import BaseEstimator
from art.estimators.classification import ClassifierMixin

__all__ = [
    "ARTWrapper",
    "pickle_deanonymizer",
    "SupervisedLearningMIA",
    "TabularDataReidentificationAttack",
    "unpickle_deanonymizer",
    "WebClassifier",
]


class SupervisedLearningMIA(ABC):
    """A base class for membership inference attacks (MIA) on anonymizers
    (privacy preserving machine learning models) which rely on the supervised
    learning paradigm.

    Subclasses need to define a way to ...

    * fit/train the attack they represent using only member features, member
      targets, non-member features and  array and a target
      array (i.e. without explicitly given hyperparameters)
    * use the (fitted) attack for membership inference
    * save the (trained) attack to disk
    * validate attack input feature arrays
    """

    @abstractmethod
    def fit(
        self,
        X_member: np.ndarray,
        y_member: np.ndarray,
        X_nonmember: np.ndarray,
        y_nonmember: np.ndarray,
    ) -> None:
        """Fit the attack against a target (which is assumed to be known in
        advance).

        Parameters
        ----------
        X_member : np.ndarray
            A feature array which was part of the attack target's training set.
        y_member : np.ndarray
            A target array which was part of the attack target's training set.
        X_nonmember : np.ndarray
            A feature array which was not part of the attack target's training
            set.
        y_nonmember : np.ndarray
            A target array which was not part of the attack target's training
            set.
        """
        pass

    @abstractmethod
    def infer_memberships(
        self, X: np.ndarray, y: np.ndarray, batch_size: int | None = None
    ) -> np.ndarray:
        """Infer the memberships of target values of a feature array.

        Parameters
        ----------
        (X, y) : (np.ndarray, np.ndarray)
            The features and targets (samples) to infer the memberships of.
        batch_size : int | None, optional
            The batch size to use while inferring (to limit compute resource
            consumption). By default `None`, which results in processing the
            whole arrays at once.

        Returns
        -------
        np.ndarray
            The memberships (boolean array). If `memberships[i]` is `True`,
            `(X[i], y[i])` is a member, i.e. part of the training dataset. If
            `memberships[i]` is `False`, it is not a member, i.e. part of the
            validation dataset.

        """
        pass

    @abstractmethod
    def save(self, filepath: str | Path) -> None:
        """Save the instance to disk, maintaining the current training progress.

        Parameters
        ----------
        filepath : str | Path
            Where to save the instance.
        """
        pass

    @abstractmethod
    def validate_input(self, feature_array: np.ndarray) -> None:
        """Check whether the input array is a valid feature array argument for
        `fit` and for `predict` (parameter `X`).

        If so, do nothing. Otherwise, raise a `ValueError`.

        Parameters
        ----------
        feature_array : np.ndarray
            The feature array to validate.

        Raises
        ------
        ValueError
            If `feature_array` is incompatible with this anonymizer.
        """
        pass


class ARTWrapper(SupervisedLearningMIA):
    """A wrapper to make membership inference attacks from the ART library
    compatible with the `SupervisedLearningMIA` interface."""

    def __init__(self, art_mia, input_validator: Callable[[np.ndarray], None]) -> None:
        """
        Parameters
        ----------
        art_mia : _type_
            The attack to wrap.
        input_validator : Callable[[np.ndarray], None]
            A function to validate feature arrays (dictated by
            `SupervisedLearningMIA`).
        """
        self._art_mia = art_mia
        self._validate_input = input_validator

    def fit(
        self,
        X_member: np.ndarray,
        y_member: np.ndarray,
        X_nonmember: np.ndarray,
        y_nonmember: np.ndarray,
    ) -> None:
        if not hasattr(self._art_mia, "fit"):
            raise NotImplementedError("Wrapped attack does not offer fit method.")
        self._art_mia.fit(
            x=X_member, y=y_member, test_x=X_nonmember, test_y=y_nonmember
        )

    def infer_memberships(
        self, X: np.ndarray, y: np.ndarray, batch_size: int | None = None
    ) -> np.ndarray:
        if not hasattr(self._art_mia, "infer"):
            raise NotImplementedError("Wrapped attack does not offer infer method.")
        return np.squeeze(self._art_mia.infer(x=X, y=y)).astype(np.bool_)

    def save(self, filepath: str | Path) -> None:
        pickle_deanonymizer(self, filepath)

    def validate_input(self, feature_array: np.ndarray) -> None:
        self._validate_input(feature_array)


class WebClassifier(ClassifierMixin, BaseEstimator):
    """A wrapper class to make supervised learning classification anonymizers
    available (via web) to ART's membership inference attacks.
    """

    def __init__(
        self,
        url: str,
        input_shape: tuple[int, ...],
        nb_classes: int,
        clip_values: tuple[float, float] | None = None,
        x_name: str = "X",
        pred_name: str = "prediction",
    ):
        """Instantiate a WebClassifier wrapper.

        Parameters
        ----------
        url : str
            The url to a POST API accepting a numpy array with the name
            `x_name`.
        input_shape : Tuple[int, ...]
            The shape of the array (except for the batch dimension) the API at
            `url` expects (alternatively: the shape of a single sample).
        nb_classes : int
            The number of classes the classifier at `url` is able to
            distinguish.
        x_name : str
            The name of the numpy array the API at `url` expects as input. By
            default "X".
        clip_values: tuple[float, float] | None
            Optional tuple of the form `(min, max)` of floats, representing the
            minimum and maximum values allowed for features. These will be used
            as the range of all features. For some attacks, this is not
            necessary to provide.
        pred_name : str
            The name of the numpy array the API at `url` returns. By default
            "prediction".
        """
        self.url = url
        self._input_shape = input_shape
        self._nb_classes = nb_classes
        self.x_name = x_name
        self.pred_name = pred_name
        self._clip_values = clip_values

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Feed an array via web to an anonymizer and retrieve the prediction
        back from the response.

        Parameters
        ----------
        x: np.ndarray
            The features.

        **kwargs
            Unused. Only present for compatibility with `BaseEstimator`.

        Returns
        -------
        prediction: np.ndarray
            The prediction.
        """
        arrays = {self.x_name: x}
        response = requests.post(
            url=self.url, data=anomed_utils.named_ndarrays_to_bytes(arrays)
        )
        response.raise_for_status()
        prediction = anomed_utils.bytes_to_named_ndarrays(response.content)
        return prediction[self.pred_name]

    def fit(self, x, y, **kwargs):
        """This has to be overridden because of the base class'
        `@abstractmethod` decorator - however it actually is not needed for the
        membership inference attack. Just ignore this function.

        Raises
        ------
        NotImplementedError
            Immediately.
        """
        raise NotImplementedError

    @property
    def input_shape(self) -> tuple[int, ...]:
        """The input shape of a single sample (i.e. ignoring the batch
        dimension).

        Returns
        -------
        tuple[int, ...]
            A single sample's input shape.
        """
        return self._input_shape

    @property
    def nb_classes(self) -> int:  # type: ignore
        return self._nb_classes

    @property
    def clip_values(self) -> tuple[float, float] | None:
        return self._clip_values


def pickle_deanonymizer(deanonymizer: Any, filepath: str | Path) -> None:
    """A default serializer that you may use, if your deanonymizer objects are
    pickle-able.

    Parameters
    ----------
    deanonymizer : Any
        The deanonymizer to pickle.
    filepath : str | Path
        Where to store the pickled deanonymizer.
    """
    with open(filepath, "wb") as file:
        pickle.dump(deanonymizer, file)


def unpickle_deanonymizer(filepath: str | Path) -> Any:
    """A default de-serializer that you may use, if your deanonymizer object was
    pickled.

    Parameters
    ----------
    filepath : str | Path
        Where to load the deanonymizer from.

    Returns
    -------
    deanonymizer: Any
        The unpickled deanonymizer.
    """
    with open(filepath, "rb") as file:
        return pickle.load(file)


class TabularDataReidentificationAttack(ABC):
    """A base class for re-identification attacks that process anonymized data
    (created by subclasses of `TabularDataAnonymizer`) and background knowledge.

    The goal of data re-identification attacks is to re-construct the data as
    close as possible to the original/leaky data, that has been processed by the
    `TabularDataAnonymizer`. Background knowledge may be used too.

    Subclasses need to define a way to re-construct/re-identify the original
    data from anonymized data and background knowledge.
    """

    @abstractmethod
    def reidentify(
        self,
        anonymized_data: pd.DataFrame,
        anonymization_scheme: anomed_challenge.AnonymizationScheme,
        background_knowledge: pd.DataFrame,
    ) -> pd.DataFrame:
        """Re-identify original/leaky tabular data from anonymized data and
        background knowledge.

        Parameters
        ----------
        anonymized_data : pd.DataFrame
            The anonymized data, obeying the `anonymization_scheme`.
        anonymization_scheme : anomed_challenge.AnonymizationScheme
            The anonymization scheme `anonymized_data` follows.
        background_knowledge : pd.DataFrame
            Additional knowledge which may be used by this attack to re-identify
            the original tabular data.
        Returns
        -------
        pd.DataFrame
            The re-identified data.
        """
        pass
