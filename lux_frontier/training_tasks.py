"""Shared task loading, rendering and grading for offline checks and live RL."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lux_frontier.prompting import render_prompt
from lux_frontier.synthetic_training import Grade, Grader, LensEnvironment, grade_response, make_environment, synthetic_grader


@dataclass(frozen=True)
class TrainingTask:
    environment: LensEnvironment
    grader: Grader
    provenance: Literal["synthetic", "development_optical"] = "synthetic"
    source_record: str | None = None

    def render(self) -> str:
        return render_prompt(self.environment.case, self.environment.tolerance_profile, provenance=self.provenance)

    def grade(self, response: str, *, truncated: bool = False) -> Grade:
        return grade_response(self.environment, response, truncated=truncated, grader=self.grader)


def synthetic_task(seed: int, split: str, index: int, grader: Grader = synthetic_grader) -> TrainingTask:
    return TrainingTask(make_environment(seed, split, index), grader)


def load_optical_task(source_path: Path) -> TrainingTask:
    # Optional, local-only import: synthetic preview still needs no optics or SDK.
    from lux_frontier.optical_grading import OpticalCache, OpticalGrader, load_recorded_case
    from lux_frontier.optical_prescriptions import ArtifactStore

    source, case, profile, settings = load_recorded_case(source_path)
    grader = OpticalGrader(ArtifactStore(Path(source["artifact_directory"])),
                          OpticalCache(Path(source["cache_directory"])), settings, profile)
    return TrainingTask(LensEnvironment(case, profile), grader, "development_optical", str(source_path))


@dataclass(frozen=True)
class FixedTaskSource:
    """Single-brief development integration source; does not claim held-out data."""

    task: TrainingTask

    def __call__(self, seed: int, split: str, index: int) -> TrainingTask:
        return self.task
