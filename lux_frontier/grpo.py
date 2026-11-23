"""Pure GRPO preparation, shared by tests and the real Tinker adapter."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Completion:
    raw_response: str
    response_text: str
    tokens: list[int]
    logprobs: list[float] | None
    stop_reason: str
    clean: bool


def group_advantages(rewards: list[float]) -> list[float]:
    """Mean-centered GRPO, as in Tinker's first-RL example (no std scaling).

    Explicitly zero constant groups before summing: floating point rounding of
    repeated nonbinary rewards must not turn a constant group into fake signal.
    """
    if not rewards or not all(isfinite(r) for r in rewards):
        raise ValueError("group rewards must be nonempty and finite")
    if max(rewards) == min(rewards):
        return [0.0] * len(rewards)
    baseline = sum(rewards) / len(rewards)
    return [r - baseline for r in rewards]


def rl_arrays(prompt_tokens: list[int], completion: Completion, advantage: float) -> dict:
    """Next-token alignment; prompt positions have zero advantage (loss mask)."""
    if not prompt_tokens or not completion.tokens:
        raise ValueError("RL datum requires nonempty prompt and completion")
    if completion.logprobs is None or len(completion.logprobs) != len(completion.tokens):
        raise ValueError("RL requires one sampling logprob per completion token")
    if not isfinite(advantage) or not all(isfinite(p) for p in completion.logprobs):
        raise ValueError("RL advantages and logprobs must be finite")
    full = prompt_tokens + completion.tokens
    prefix = len(prompt_tokens) - 1
    return dict(input_tokens=full[:-1], target_tokens=full[1:],
                logprobs=[0.0] * prefix + completion.logprobs,
                advantages=[0.0] * prefix + [advantage] * len(completion.tokens))
