"""Small JSON contract; no optimization, tolerancing, or reward policy.

Lengths are mm, field angles degrees, wavelengths micrometers, frequencies
image-space line pairs/mm. MTF and yield are fractions in [0, 1], never percent.
"""

from math import isclose
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

Text = Annotated[str, Field(min_length=1)]
Positive = Annotated[float, Field(gt=0)]
Nonnegative = Annotated[float, Field(ge=0)]
Fraction = Annotated[float, Field(ge=0, le=1)]
LocalID = Annotated[int, Field(strict=True, ge=0)]
Count = Annotated[int, Field(strict=True, ge=0)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ArtifactRef(Contract):
    """Full artifact lives outside the prompt. Revision must be immutable.

    The artifact owner must resolve (uri, revision) to the same bytes forever.
    A prescription includes all surfaces, materials, aperture, fields, wavelengths
    and sensor position. A tolerance profile defines error bounds/distributions,
    units, correlations, and allowed compensators (including final refocus).
    This milestone references those artifacts; it does not implement them.
    """

    uri: Text
    revision: Text


class MTFPoint(Contract):
    field_angle_deg: tuple[float, float]  # object-space (x, y), object at infinity
    spatial_frequency_lp_per_mm: Positive
    wavelength_um: Positive  # monochromatic measurement; no implicit spectral mix
    orientation: Literal["sagittal", "tangential"]


class MTFThreshold(MTFPoint):
    minimum_mtf_fraction: Fraction


class MTFMeasurement(MTFPoint):
    mtf_fraction: Fraction


class Brief(Contract):
    brief_id: Text
    target_focal_length_mm: Positive
    focal_length_tolerance_mm: Nonnegative  # symmetric absolute +/- tolerance
    max_first_surface_to_sensor_mm: Positive  # axial distance, not back focal length
    target_f_number: Positive
    f_number_tolerance: Nonnegative  # symmetric absolute +/- tolerance, dimensionless
    mtf_thresholds: Annotated[tuple[MTFThreshold, ...], Field(min_length=1)]
    minimum_simulated_yield_fraction: Fraction
    manufacturing_tolerance_profile_ref: ArtifactRef


class NominalMetrics(Contract):
    effective_focal_length_mm: Positive
    first_surface_to_sensor_mm: Positive
    f_number: Positive
    mtf: Annotated[tuple[MTFMeasurement, ...], Field(min_length=1)]


class GeometrySummary(Contract):
    """Decision features, not a replacement for the full prescription artifact."""

    element_count: Annotated[int, Field(strict=True, gt=0)]
    minimum_edge_thickness_mm: Positive
    minimum_air_gap_mm: Nonnegative
    maximum_clear_aperture_mm: Positive
    maximum_absolute_curvature_per_mm: Nonnegative


class Candidate(Contract):
    candidate_id: LocalID  # local to the enclosing BriefCase
    prescription_ref: ArtifactRef
    layout: Text  # prompt-visible description, ordered from object to sensor
    glass_types: Annotated[tuple[Text, ...], Field(min_length=1)]  # same element order
    nominal_metrics: NominalMetrics
    geometry: GeometrySummary | None = None  # optional for existing contract callers


class Submission(Contract):
    starter_ids: tuple[LocalID, LocalID]

    @model_validator(mode="after")
    def distinct_ids(self) -> Self:
        if self.starter_ids[0] == self.starter_ids[1]:
            raise ValueError("starter_ids must contain two distinct IDs")
        return self


class SimulationResult(Contract):
    status: Literal["not_run", "failed", "completed"] = "not_run"
    builds_passed: Count | None = None
    builds_tested: Count | None = None
    yield_fraction: Fraction | None = None
    error: Text | None = None

    @model_validator(mode="after")
    def consistent_measurement(self) -> Self:
        if self.status == "completed":
            if (self.builds_tested is None or self.builds_tested == 0
                    or self.builds_passed is None or self.yield_fraction is None):
                raise ValueError("completed simulation requires counts and yield, tested > 0")
            if self.builds_passed > self.builds_tested or not isclose(
                self.yield_fraction, self.builds_passed / self.builds_tested,
                rel_tol=0, abs_tol=1e-12,
            ):
                raise ValueError("yield must equal builds_passed / builds_tested")
        elif any(v is not None for v in (
            self.builds_passed, self.builds_tested, self.yield_fraction,
        )):
            raise ValueError("unmeasured simulations must have null counts and yield")
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("only failed simulations require an error")
        return self


class GraderResult(Submission):
    brief_id: Text
    shipped_prescription_ref: ArtifactRef | None = None  # may be a refined derivative
    nominal_metrics: NominalMetrics | None = None  # metrics of the shipped lens
    constraint_failures: tuple[Text, ...] = ()
    simulation: SimulationResult = Field(default_factory=SimulationResult)
    final_pass: StrictBool | None = None  # null means no final decision
    reward: float | None = None  # null until a later reward policy assigns a scalar

    @model_validator(mode="after")
    def completed_pass(self) -> Self:
        if self.final_pass is True and (
            self.simulation.status != "completed" or self.nominal_metrics is None
            or self.shipped_prescription_ref is None or self.constraint_failures
        ):
            raise ValueError("a pass requires a shipped lens, metrics, and completed simulation without failures")
        return self


class BriefCase(Contract):
    """The current brief and its candidate namespace; use its parsers at ingress."""

    brief: Brief
    candidates: Annotated[tuple[Candidate, ...], Field(min_length=2)]

    @model_validator(mode="after")
    def unique_candidate_ids(self) -> Self:
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate IDs must be unique within a brief")
        return self

    def _check_ids(self, submission: Submission) -> None:
        unknown = set(submission.starter_ids) - {c.candidate_id for c in self.candidates}
        if unknown:
            raise ValueError(f"unknown starter IDs for brief {self.brief.brief_id}: {sorted(unknown)}")

    def parse_submission(self, payload: str) -> Submission:
        submission = Submission.model_validate_json(payload)
        self._check_ids(submission)
        return submission

    def parse_result(self, payload: str) -> GraderResult:
        result = GraderResult.model_validate_json(payload)
        if result.brief_id != self.brief.brief_id:
            raise ValueError("result belongs to a different brief")
        self._check_ids(result)
        return result
