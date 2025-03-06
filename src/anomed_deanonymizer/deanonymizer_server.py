import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

import anomed_utils as utils
import falcon
import numpy as np
import requests

from . import deanonymizer

__all__ = [
    "EvaluationResource",
    "supervised_learning_MIA_server_factory",
    "validate_deanonymizer_input_or_raise",
]

logger = logging.getLogger(__name__)


class EvaluationResource:
    """This resource is intended for evaluating membership inference attacks
    targeting estimators which follow the supervised learning paradigm.
    """

    def __init__(
        self,
        anonymizer_identifier: str,
        deanonymizer_identifier: str,
        model_filepath: str | Path,
        model_loader: Callable[[str | Path], deanonymizer.SupervisedLearningMIA],
        default_batch_size: int,
        evaluation_data_url: str,
        utility_evaluation_url: str,
    ) -> None:
        """
        Parameters
        ----------
        anonymizer_identifier : str
            The attack target's identifier.
        deanonymizer_identifier : str
            The attacker's identifier.
        model_filepath : str | Path
            Where to save the attack model to disk after fitting.
        model_loader : Callable[[str  |  Path], deanonymizer.SupervisedLearningMIA]
            How to load a fitted attack model from disk.
        default_batch_size : int
            The batch size to use for membership inference if not provided
            otherwise. Choose sensibly to avoid too much resource consumption.
        evaluation_data_url : str
            Where to obtain attack success evaluation data from.
        utility_evaluation_url : str
            Where to submit inferred memberships to, such that they will be
            evaluated.
        """
        self._anon_id = anonymizer_identifier
        self._deanon_id = deanonymizer_identifier
        self._model_filepath = Path(model_filepath)
        self._load_model = model_loader
        self._default_batch_size = default_batch_size
        self._evaluation_data_url = evaluation_data_url
        self._utility_evaluation_url = utility_evaluation_url
        self._timeout = 10.0
        self._loaded_model: deanonymizer.SupervisedLearningMIA = None  # type: ignore
        self._loaded_model_modification_time: datetime = None  # type: ignore
        self._expected_array_labels = ["X", "y"]

    def on_post(self, req: falcon.Request, resp: falcon.Response) -> None:
        self._load_most_recent_model()

        try:
            data_split = req.get_param("data_split", required=True)
        except falcon.HTTPBadRequest:
            raise falcon.HTTPBadRequest(
                description="Query parameter 'data_split' is missing!"
            )

        if data_split not in ["tuning", "validation"]:
            raise falcon.HTTPBadRequest(
                description="Query parameter data_split needs to be either 'tuning' or "
                "'validation'."
            )

        array = utils.get_named_arrays_or_raise(
            self._evaluation_data_url,
            expected_array_labels=self._expected_array_labels,
            params=dict(
                data_split=data_split,
                anonymizer=self._anon_id,
                deanonymizer=self._deanon_id,
            ),
            timeout=self._timeout,
        )

        (X, y) = (
            array[self._expected_array_labels[0]],
            array[self._expected_array_labels[1]],
        )
        logger.debug(
            "Inferring memberships (context: evaluation) for features with shape "
            f"{X.shape} and dtype {X.dtype}, and targets with shape {y.shape} and dtype "
            f"{y.dtype}."
        )
        memberships = self._loaded_model.infer_memberships(
            X=X, y=y, batch_size=self._default_batch_size
        )

        logger.debug(
            f"Submitting memberships of shape {memberships.shape} and dtype "
            f"{memberships.dtype} to {self._utility_evaluation_url} for evaluation."
        )
        evaluation_response = requests.post(
            url=self._utility_evaluation_url,
            data=utils.named_ndarrays_to_bytes(dict(prediction=memberships)),
            params=dict(
                anonymizer=self._anon_id,
                deanonymizer=self._deanon_id,
                data_split=data_split,
            ),
        )
        if evaluation_response.status_code == 201:
            resp.text = json.dumps(
                dict(
                    message=(
                        f"The deanonymizer has been evaluated based on {data_split} data."
                    ),
                    evaluation=evaluation_response.json(),
                )
            )
            resp.status = falcon.HTTP_CREATED
        else:
            raise falcon.HTTPInternalServerError(
                description="Utility evaluation failed."
            )

    def _load_most_recent_model(self) -> None:
        if not self._model_filepath.exists():
            raise falcon.HTTPBadRequest(
                description="This deanonymizer is not fitted yet.",
            )
        mod_time_from_disk = datetime.fromtimestamp(
            self._model_filepath.stat().st_mtime
        )
        if _is_older(self._loaded_model_modification_time, mod_time_from_disk):
            logger.debug(
                "Loading model from disk as it is more recent than the already loaded "
                "model."
            )
            self._loaded_model = self._load_model(self._model_filepath)
            self._loaded_model_modification_time = mod_time_from_disk
        else:
            logger.debug(
                "Keeping the currently loaded model, is it is already the most recent "
                "one."
            )
            pass


def _is_older(dt1: datetime | None, dt2: datetime) -> bool:
    """Tell whether `dt1` is older (i.e. more in the past) than `dt2`. If `dt1`
    is the same as `dt2`, or even if `dt1` is `None`, output `True`."""
    if dt1 is None:
        return True
    else:
        return dt1 <= dt2


def supervised_learning_MIA_server_factory(
    anonymizer_identifier: str,
    deanonymizer_identifier: str,
    deanonymizer_obj: deanonymizer.SupervisedLearningMIA,
    model_filepath: str | Path,
    default_batch_size: int,
    member_url: str,
    nonmember_url: str,
    evaluation_data_url: str,
    utility_evaluation_url: str,
    model_loader: Callable[[str | Path], deanonymizer.SupervisedLearningMIA],
) -> falcon.App:
    """A factory to create a web application object which hosts an
    `deanonymizer.SupervisedLearningMIA`, a basic membership inference attack
    (MIA) on anonymizers.

    By using this factory, you don't have to worry any web-programming issues,
    as they are hidden from you. The generated web app will feature the
    following routes (more details may be found in this project's openapi
    specification):

    * [GET] `/`
    * [POST] `/fit`
    * [POST] `/evaluate`

    Parameters
    ----------
    anonymizer_identifier : str
        The identifier of the anonymizer under attack.
    deanonymizer_identifier : str
        The identifier of `deanonymizer_obj`.
    deanonymizer_obj : deanonymizer.SupervisedLearningMIA
        A membership inference attack against an anonymizer, which is based on
        the supervised learning paradigm.
    model_filepath : str | Path
        Where to write fitted attacks to disk.
    default_batch_size : int
        Which batch size to use when inferring memberships, if not specified
        otherwise.
    member_url : str
        Where to download the feature array and target array, which are members
        of the training dataset.
    nonmember_url : str
        Where to download the feature array and target array, which are not
        members of the training dataset.
    evaluation_data_url : str
        Where to download the
    utility_evaluation_url : str
        Where to submit inferred memberships to, for evaluation.
    model_loader : Callable[[str  |  Path], deanonymizer.SupervisedLearningMIA]
        A function to load a deanonymizer, that has been saved to disk by using
        `deanonymizer_obj.save(...)`.

    Returns
    -------
    falcon.App
        A web application object based on the falcon web framework.
    """
    app = falcon.App()

    app.add_route(
        "/", utils.StaticJSONResource(dict(message="Deanonymizer server is alive!"))
    )
    app.add_route(
        "/fit",
        utils.FitResource(
            data_getter=_get_deanonymizer_fit_data(
                deanonymizer=deanonymizer_obj,
                member_url=member_url,
                nonmember_url=nonmember_url,
                timeout=10.0,
            ),
            model=deanonymizer_obj,
            model_filepath=model_filepath,
        ),
    )
    app.add_route(
        "/evaluate",
        EvaluationResource(
            anonymizer_identifier=anonymizer_identifier,
            deanonymizer_identifier=deanonymizer_identifier,
            model_filepath=model_filepath,
            model_loader=model_loader,
            default_batch_size=default_batch_size,
            evaluation_data_url=evaluation_data_url,
            utility_evaluation_url=utility_evaluation_url,
        ),
    )
    return app


def _get_deanonymizer_fit_data(
    deanonymizer: deanonymizer.SupervisedLearningMIA,
    member_url: str,
    nonmember_url: str,
    timeout: float,
    expected_array_labels: list[str] | None = None,
) -> Callable[[], dict[str, np.ndarray]]:
    if expected_array_labels is None:
        expected_array_labels = ["X", "y"]

    def getter():
        fit_data: dict[str, np.ndarray] = {}
        for url, tag in [
            (member_url, "member"),
            (nonmember_url, "nonmember"),
        ]:
            arrays = utils.get_named_arrays_or_raise(
                url,
                expected_array_labels,
                timeout=timeout,
            )

            for var, idx in [("X", 0), ("y", 1)]:
                fit_data[f"{var}_{tag}"] = arrays[expected_array_labels[idx]]
            validate_deanonymizer_input_or_raise(
                fit_data[f"X_{tag}"],
                deanonymizer,
                falcon.HTTP_INTERNAL_SERVER_ERROR,
                f"The deanonymizer is not compatible with the {tag} feature array.",
            )
        return fit_data

    return getter


def validate_deanonymizer_input_or_raise(
    feature_array: np.ndarray,
    deanonymizer: deanonymizer.SupervisedLearningMIA,
    error_status: str | int | None = falcon.HTTP_INTERNAL_SERVER_ERROR,
    error_msg: str | None = None,
) -> None:
    """Validate the input for a deanonymizer. If validation fails, raise a
    `falcon.HTTPError` instead.

    Parameters
    ----------
    feature_array : np.ndarray
        A NumPy array containing the features for this deanonymizer.
    deanonymizer : deanonymizer.SupervisedLearningMIA
        The deanonymizer to validate input for. This function will use the
        deanonymizer's `validate_input` method.
    error_status : str | int | None, optional
        The error status to use if validation fails. By default,
        `falcon.HTTP_INTERNAL_SERVER_ERROR`.
    error_msg : str | None, optional
        The error message to output. By default `None`, which will result in a
        generic message derived from the `error_status`.

    Raises
    ------
    falcon.HTTPError
        If validation fails.
    """
    try:
        deanonymizer.validate_input(feature_array)
    except ValueError:
        if error_status is None:
            error_status = falcon.HTTP_INTERNAL_SERVER_ERROR
        raise falcon.HTTPError(status=error_status, description=error_msg)
