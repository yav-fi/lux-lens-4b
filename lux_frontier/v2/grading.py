"""Dependency-free cached grading, used by Lux and copied verbatim into Harbor."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

VERSION = "lux-selection-v2.0"
PHASES = ("construction", "training", "development", "final")
MODES = ("nominal", "yield")
MAX_RESPONSE_BYTES = 4096


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


@dataclass(frozen=True)
class Grade:
    starter_id: int | None
    reward: float
    nominal_pass: bool
    yield_fraction: float
    final_pass: bool
    error: str | None = None


def grade(task, response, *, phase="final", mode="yield", clean=True):
    if task["version"] != VERSION or phase not in PHASES or mode not in MODES:
        raise ValueError("unsupported grading contract, phase or reward mode")
    error = None
    if not clean:
        error = "unclean_completion"
    else:
        try:
            if not isinstance(response, str) or len(response.encode("utf-8")) > MAX_RESPONSE_BYTES:
                raise ValueError("submission exceeds the JSON text contract")
            payload = json.loads(response, object_pairs_hook=unique_object)
            if not isinstance(payload, dict) or set(payload) != {"starter_id"} or type(payload["starter_id"]) is not int:
                raise ValueError("expected one integer starter_id")
            local_id = payload["starter_id"]
        except (ValueError, RecursionError, TypeError):
            error = "invalid_submission"
        else:
            if str(local_id) not in task["outcomes"]:
                error = "unknown_starter"
    if error:
        return Grade(None, 0.0, False, 0.0, False, error)
    outcome = task["outcomes"][str(local_id)]
    nominal = outcome["nominal_pass"]
    manufactured = outcome["simulations"].get(phase) if mode == "yield" else None
    if nominal and mode == "yield" and manufactured is None:
        raise ValueError("requested draw phase is unavailable")
    yield_fraction = manufactured["passed"] / manufactured["tested"] if manufactured else 0.0
    if mode == "nominal":
        reward = 1.0 if nominal else outcome["near_miss"]
    else:
        # The same monotone affine transform applies to every nominal success.
        # Invalid nominal designs sit below even a design with measured zero yield.
        reward = 0.05 + 0.95 * yield_fraction if nominal else outcome["near_miss"] * 0.04
    return Grade(local_id, reward, nominal, yield_fraction,
                 nominal and manufactured is not None and yield_fraction >= task["yield_target"])


def submit(local_id):
    return json.dumps({"starter_id": int(local_id)})


def verify_harbor(cache, response, reward):
    envelope = json.loads(Path(cache).read_text())
    task = envelope["task"]
    if digest(task) != envelope["sha256"]:
        raise ValueError("cached Harbor grading data failed integrity check")
    try:
        with Path(response).open("rb") as stream:
            raw = stream.read(MAX_RESPONSE_BYTES+1)
        answer = raw.decode() if len(raw) <= MAX_RESPONSE_BYTES else ""
    except (FileNotFoundError, UnicodeError, IsADirectoryError):
        answer = ""
    result = grade(task, answer, phase=envelope["phase"])
    Path(reward).write_text(f"{result.reward:.12g}\n")


if __name__ == "__main__":
    import sys

    verify_harbor(*sys.argv[1:])
