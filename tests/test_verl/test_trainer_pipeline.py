import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

pytestmark = pytest.mark.skipif(torch is None, reason="torch is not installed")
if torch is None:  # pragma: no cover
    pytest.skip("torch is not installed", allow_module_level=True)

from grpo_composer.integrations.verl import trainer as composer_trainer


class DummyData:
    def __init__(self, batch: dict, non_tensor_batch: dict | None = None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}


def test_unlikeliness_transform_preserves_sequence_sums():
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    token_level_rewards = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    old_log_probs = torch.tensor(
        [
            [-1.0, -1.2, 0.0],
            [-0.8, -0.9, -1.0],
            [-1.5, 0.0, 0.0],
            [-2.0, -2.2, 0.0],
        ]
    )

    data = DummyData(
        batch={
            "token_level_rewards": token_level_rewards.clone(),
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
        },
        non_tensor_batch={"uid": ["a", "a", "b", "b"]},
    )

    composer_trainer._apply_unlikeliness_reward_transform(data, {"unlikeliness_beta": 0.5})

    transformed_token_rewards = data.batch["token_level_rewards"]
    transformed_sequence_rewards = data.batch["composer_sequence_rewards"]

    recovered_sequence_rewards = (transformed_token_rewards * response_mask).sum(dim=-1)
    assert torch.allclose(recovered_sequence_rewards, transformed_sequence_rewards, atol=1e-6)


def test_multi_reward_transform_rejects_bad_token_shape():
    response_mask = torch.ones(4, 3)
    data = DummyData(
        batch={
            "response_mask": response_mask,
            "token_level_rewards": torch.ones(4, 3),
            "composer_multi_rewards": torch.ones(4, 4, 2),
        },
        non_tensor_batch={"uid": [0, 0, 1, 1]},
    )

    with pytest.raises(ValueError, match="3D multi_rewards must align"):
        composer_trainer._apply_multi_reward_transform(data, {"multi_reward_weights": [1.0, 1.0]})


def test_standard_context_injection_sets_sequence_rewards():
    response_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    token_level_rewards = torch.tensor([[0.5, 0.5], [1.0, 0.0]])

    data = DummyData(
        batch={
            "response_mask": response_mask,
            "token_level_rewards": token_level_rewards,
        }
    )

    composer_trainer._inject_standard_composer_context(data)
    assert "composer_sequence_rewards" in data.batch
    expected = torch.tensor([1.0, 1.0])
    assert torch.allclose(data.batch["composer_sequence_rewards"], expected)
