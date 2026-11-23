"""Installed SDK boundary checks; no account, network, or model weights needed."""

import asyncio
from types import SimpleNamespace

import pytest

from lux_frontier.grpo import Completion
from lux_frontier.tinker_backend import TinkerBackend
from lux_frontier.training import Settings

tinker = pytest.importorskip("tinker")
pytest.importorskip("tinker_cookbook")


def test_real_sdk_datums_have_completion_only_advantages_and_update_is_awaited():
    backend = TinkerBackend(Settings())
    backend.tinker = tinker
    events = []

    class Training:
        async def forward_backward(self, data, loss_fn):
            assert loss_fn == "importance_sampling"
            assert len(data) == 1 and isinstance(data[0], tinker.Datum)
            datum = data[0]
            assert datum.model_input.to_ints() == [11, 12, 31]
            assert set(datum.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}
            assert datum.loss_fn_inputs["target_tokens"].to_numpy().tolist() == [12, 31, 32]
            assert datum.loss_fn_inputs["advantages"].to_numpy().tolist() == [0, 0.5, 0.5]
            events.append("backward-completed")
            return SimpleNamespace(metrics={"loss:sum": 1.5})

        async def optim_step(self, params):
            assert events == ["backward-completed"]
            assert params.learning_rate == 1e-5
            events.append("optimizer-completed")
            return SimpleNamespace(metrics={"step": 1})

    backend.training = Training()
    completion = Completion("raw", "text", [31, 32], [-0.3, -0.4], "stop", True)
    result = asyncio.run(backend.update([([11, 12], completion, 0.5)]))
    assert events == ["backward-completed", "optimizer-completed"]
    assert result["datums"] == 1 and result["optimizer"] == {"step": 1}


def test_stop_reason_and_clean_chat_termination_are_both_required():
    backend = TinkerBackend(Settings())
    backend.tinker = tinker
    backend.tokenizer = SimpleNamespace(decode=lambda tokens, **_: "untouched raw <|im_end|>")
    def parse_response(tokens):
        if tokens[0] == 3:
            raise ValueError("malformed chat framing")
        return ({"role": "assistant", "content": '{"starter_ids":[3,7]}'},
                SimpleNamespace(is_clean=tokens[0] != 4))

    backend.renderer = SimpleNamespace(
        get_stop_sequences=lambda: [99],
        parse_response=parse_response,
    )

    class Sampler:
        async def sample(self, *, prompt, num_samples, sampling_params):
            assert isinstance(prompt, tinker.ModelInput)
            assert num_samples == 4
            assert sampling_params.max_tokens == 128
            assert sampling_params.stop == [99] and sampling_params.seed == 17
            return SimpleNamespace(sequences=[
                tinker.SampledSequence(stop_reason=reason, sequence_id=str(index),
                                      _tokens_list=[index, 99], _logprobs_list=[-0.1, -0.2])
                for index, reason in [(1, "stop"), (2, "length"), (3, "stop"), (4, "stop")]
            ])

    backend.sampler = Sampler()
    completions = asyncio.run(backend.sample([11, 12], 4, 17))
    assert [c.clean for c in completions] == [True, False, False, False]
    assert completions[0].raw_response == "untouched raw <|im_end|>"
    assert completions[0].response_text == '{"starter_ids":[3,7]}'
    assert completions[2].response_text == completions[2].raw_response


def test_sampler_refresh_checkpoint_paths_ttl_and_reload_use_actual_sdk_signatures():
    backend = TinkerBackend(Settings())
    events = []

    class Training:
        async def save_weights_and_get_sampling_client(self):
            events.append("refresh")
            return "current-sampler"

        async def save_state(self, *, name, ttl_seconds):
            assert ttl_seconds == 86400
            return SimpleNamespace(path="tinker://test/state/final")

        async def save_weights_for_sampler(self, *, name, ttl_seconds):
            assert ttl_seconds == 86400
            return SimpleNamespace(path="tinker://test/sampler/final")

    class Service:
        async def create_sampling_client(self, *, model_path):
            assert model_path == "tinker://test/sampler/final"
            events.append("reload")
            return "reloaded-sampler"

        async def close(self, status):
            events.append(status)

    backend.training, backend.service = Training(), Service()

    async def exercise():
        await backend.refresh_sampler()
        assert backend.sampler == "current-sampler"
        checkpoint = await backend.save_checkpoint()
        await backend.reload_checkpoint(checkpoint)
        assert backend.sampler == "reloaded-sampler"
        await backend.close("success")

    asyncio.run(exercise())
    assert events == ["refresh", "reload", "success"]
