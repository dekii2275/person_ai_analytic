"""M04-01 tests for body attribute contracts and taxonomy."""

from __future__ import annotations

import ast
import inspect
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import src.body_attributes.base as base_module
import src.body_attributes.schemas as schemas_module
from src.body_attributes import (
    BODY_ATTRIBUTE_NAMESPACE,
    BODY_ATTRIBUTE_SCHEMA_VERSION,
    BaseBodyAttributeRecognizer,
    BodyAttributeKey,
    BodyAttributePrediction,
)


class TestBodyAttributeTaxonomy:
    def test_namespace_is_stable(self):
        assert BODY_ATTRIBUTE_NAMESPACE == "body"

    def test_baseline_keys_are_fixed(self):
        assert [key.value for key in BodyAttributeKey] == [
            "backpack",
            "bag",
            "hat",
            "long_sleeve",
        ]

    def test_keys_are_json_serializable_strings(self):
        assert json.dumps({"key": BodyAttributeKey.BACKPACK}) == (
            '{"key": "backpack"}'
        )


class TestBodyAttributePrediction:
    def test_to_dict_has_stable_json_shape(self):
        prediction = BodyAttributePrediction(
            key=BodyAttributeKey.BACKPACK,
            value=True,
            score=0.87,
        )

        assert prediction.schema_version == BODY_ATTRIBUTE_SCHEMA_VERSION
        assert prediction.to_dict() == {
            "schema_version": "1.0",
            "key": "backpack",
            "value": True,
            "score": 0.87,
        }
        assert json.loads(json.dumps(prediction.to_dict())) == (
            prediction.to_dict()
        )

    def test_integer_score_is_normalized_to_float(self):
        prediction = BodyAttributePrediction(
            BodyAttributeKey.HAT,
            False,
            1,
        )
        assert prediction.score == 1.0
        assert isinstance(prediction.score, float)

    def test_is_frozen(self):
        prediction = BodyAttributePrediction(
            BodyAttributeKey.BAG,
            False,
            0.2,
        )
        with pytest.raises(FrozenInstanceError):
            prediction.score = 0.9

    @pytest.mark.parametrize(
        "key",
        ["backpack", "glasses", None, 1],
    )
    def test_rejects_non_canonical_key(self, key):
        with pytest.raises(TypeError):
            BodyAttributePrediction(key, True, 0.8)

    @pytest.mark.parametrize("value", [0, 1, "true", None, []])
    def test_rejects_non_boolean_value(self, value):
        with pytest.raises(TypeError):
            BodyAttributePrediction(BodyAttributeKey.HAT, value, 0.8)

    @pytest.mark.parametrize(
        "score",
        [-0.01, 1.01, float("nan"), float("inf"), float("-inf")],
    )
    def test_rejects_invalid_numeric_score(self, score):
        with pytest.raises(ValueError):
            BodyAttributePrediction(BodyAttributeKey.LONG_SLEEVE, True, score)

    @pytest.mark.parametrize("score", [True, "0.8", None])
    def test_rejects_non_numeric_score(self, score):
        with pytest.raises(TypeError):
            BodyAttributePrediction(BodyAttributeKey.LONG_SLEEVE, True, score)


class _FakeRecognizer(BaseBodyAttributeRecognizer):
    def __init__(self) -> None:
        self.released = False

    def predict(
        self,
        person_crop: np.ndarray,
    ) -> list[BodyAttributePrediction]:
        return [
            BodyAttributePrediction(
                BodyAttributeKey.LONG_SLEEVE,
                True,
                0.75,
            )
        ]

    def release(self) -> None:
        self.released = True


class TestBaseBodyAttributeRecognizer:
    def test_base_class_is_abstract(self):
        with pytest.raises(TypeError):
            BaseBodyAttributeRecognizer()

    def test_backend_can_return_only_contract_objects(self):
        crop = np.zeros((128, 64, 3), dtype=np.uint8)
        recognizer = _FakeRecognizer()

        predictions = recognizer.predict(crop)

        assert predictions == [
            BodyAttributePrediction(
                BodyAttributeKey.LONG_SLEEVE,
                True,
                0.75,
            )
        ]
        assert all(
            isinstance(item, BodyAttributePrediction)
            for item in predictions
        )

    def test_context_manager_releases_resources(self):
        recognizer = _FakeRecognizer()
        with recognizer as active:
            assert active is recognizer
        assert recognizer.released is True

    @pytest.mark.parametrize("module", [schemas_module, base_module])
    def test_contract_modules_do_not_import_inference_backends(self, module):
        tree = ast.parse(inspect.getsource(module))
        forbidden = {"torch", "ultralytics", "onnxruntime", "tensorrt"}
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])

        assert imported_roots.isdisjoint(forbidden)
