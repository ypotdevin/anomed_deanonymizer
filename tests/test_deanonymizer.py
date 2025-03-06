from unittest.mock import MagicMock

import anomed_utils as utils
import numpy as np
import pytest
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

import anomed_deanonymizer as deanonymizer


@pytest.fixture()
def example_features() -> np.ndarray:
    return np.arange(10)


@pytest.fixture()
def example_targets() -> np.ndarray:
    return np.zeros(shape=(10,))


def test_web_classifier(mocker, example_features, example_targets):
    web_clf = deanonymizer.WebClassifier(
        url="http://example.com/predict",
        input_shape=(4,),
        nb_classes=3,
        clip_values=(0.0, 1.0),
    )
    assert web_clf.input_shape == (4,)
    assert web_clf.nb_classes == 3
    assert web_clf.clip_values == (0.0, 1.0)

    mock = _mock_post_numpy_arrays(mocker, dict(prediction=example_targets))
    assert np.array_equal(web_clf.predict(example_features), example_targets)
    mock.assert_called_once()


def _mock_post_numpy_arrays(
    _mocker, named_arrays: dict[str, np.ndarray], status_code: int = 201
) -> MagicMock:
    mock_response = _mocker.MagicMock()
    mock_response.status_code = status_code
    mock_response.content = utils.named_ndarrays_to_bytes(named_arrays)
    return _mocker.patch("requests.post", return_value=mock_response)


def test_art_wrapper(mocker, example_targets, tmp_path):
    attack_target = deanonymizer.WebClassifier(
        url="http://example.com/predict",
        input_shape=(10,),
        nb_classes=2,
    )
    attack_art = MembershipInferenceBlackBox(
        estimator=attack_target,  # type: ignore
        attack_model_type="gb",
    )
    attack = deanonymizer.ARTWrapper(
        art_mia=attack_art, input_validator=_validate_input_array
    )
    example_features = np.arange(100, dtype=np.float_).reshape(10, 10)

    mock = _mock_post_numpy_arrays(mocker, dict(prediction=example_targets))
    attack.validate_input(example_features)
    attack.fit(
        X_member=example_features,
        y_member=example_targets,
        X_nonmember=example_features,
        y_nonmember=example_targets,
    )
    mock.assert_called()

    assert np.array_equal(
        attack.infer_memberships(example_features, example_targets),
        example_targets.astype(np.bool_),
    )

    p = tmp_path / "attack.pkl"
    attack.save(p)
    attack_ = deanonymizer.unpickle_deanonymizer(p)
    assert attack_ is not None


def test_failing_art_wrapper(example_features, example_targets):
    attack = deanonymizer.ARTWrapper(42, lambda _: None)
    with pytest.raises(NotImplementedError):
        attack.fit(example_features, example_targets, example_features, example_targets)
    with pytest.raises(NotImplementedError):
        attack.infer_memberships(example_features, example_targets)


def _validate_input_array(feature_array: np.ndarray) -> None:
    if feature_array.shape[1] != 10 or len(feature_array.shape) != 2:
        raise ValueError("Feature array needs to have shape (n_samples, 10).")
    if feature_array.dtype != np.float_:
        raise ValueError("Feature array must be an array of floats.")
