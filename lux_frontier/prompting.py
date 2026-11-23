"""Plain-text model input; resolved assumptions are explicit, never inferred."""

from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, model_validator

from lux_frontier.contracts import ArtifactRef, BriefCase, Contract, Nonnegative, Positive, Text


class ToleranceError(Contract):
    parameter: Literal["radius_error", "thickness_error", "air_gap_error", "decenter", "tilt", "index_error"]
    applies_to: Text
    distribution: Literal["uniform", "truncated_normal"]
    units: Literal["mm", "deg", "dimensionless"]
    lower_bound: float
    upper_bound: float
    mean: float = 0
    standard_deviation: Positive | None = None

    @model_validator(mode="after")
    def consistent_distribution(self) -> Self:
        if not self.lower_bound < self.upper_bound:
            raise ValueError("tolerance lower bound must be below upper bound")
        if not self.lower_bound <= self.mean <= self.upper_bound:
            raise ValueError("tolerance mean must lie within bounds")
        if (self.distribution == "truncated_normal") != (self.standard_deviation is not None):
            raise ValueError("only truncated_normal requires a standard deviation")
        if self.distribution == "uniform" and abs(self.mean - (self.lower_bound + self.upper_bound) / 2) > 1e-12:
            raise ValueError("uniform mean must be the midpoint of its bounds")
        expected = {"radius_error": "dimensionless", "thickness_error": "mm", "air_gap_error": "mm",
                    "decenter": "mm", "tilt": "deg", "index_error": "dimensionless"}
        if self.units != expected[self.parameter]:
            raise ValueError(f"{self.parameter} requires units {expected[self.parameter]}")
        return self


class RefocusAssumptions(Contract):
    allowed: StrictBool
    maximum_sensor_shift_mm: Nonnegative
    objective: Text

    @model_validator(mode="after")
    def consistent_shift(self) -> Self:
        if not self.allowed and self.maximum_sensor_shift_mm != 0:
            raise ValueError("disallowed refocus must have zero sensor shift")
        return self


class ResolvedToleranceProfile(Contract):
    reference: ArtifactRef
    errors: Annotated[tuple[ToleranceError, ...], Field(min_length=1)]
    correlations: Text  # explicit independence or a description of shared errors
    refocus: RefocusAssumptions
    provisional: StrictBool = False


def resolve_profile(
    ref: ArtifactRef,
    profiles: Mapping[tuple[str, str], ResolvedToleranceProfile],
) -> ResolvedToleranceProfile:
    profile = profiles.get((ref.uri, ref.revision))
    if profile is None or profile.reference != ref:
        raise ValueError(f"required tolerance profile cannot be resolved: {ref.uri}@{ref.revision}")
    return profile


def render_prompt(case: BriefCase, profile: ResolvedToleranceProfile | None, *,
                  provenance: Literal["synthetic", "development_optical"] = "synthetic") -> str:
    """Whitelist model-visible fields. Artifact URIs and grading stay in the environment."""
    brief = case.brief
    if profile is None or profile.reference != brief.manufacturing_tolerance_profile_ref:
        raise ValueError("required tolerance profile is unresolved or does not match the brief")
    lines = [
        "Select two distinct starter lens designs for the brief below.",
        ("These candidates and measurements are SYNTHETIC training data." if provenance == "synthetic"
         else "DEVELOPMENT optical case: candidate measurements are computed from actual prescriptions using Optiland 0.6.3."),
        "Consider nominal requirements and robustness to the stated manufacturing errors.",
        "Brief requirements:",
        f"Effective focal length: {brief.target_focal_length_mm:g} +/- {brief.focal_length_tolerance_mm:g} mm.",
        f"Maximum first-surface-to-sensor axial distance: {brief.max_first_surface_to_sensor_mm:g} mm.",
        f"F-number: {brief.target_f_number:g} +/- {brief.f_number_tolerance:g} (dimensionless).",
        f"Minimum simulated manufacturing yield: {brief.minimum_simulated_yield_fraction:g} (fraction).",
        "Object at infinity; field angles are object-space (x, y).",
    ]
    for point in brief.mtf_thresholds:
        lines.append(
            f"Required MTF: field {point.field_angle_deg} deg; {point.spatial_frequency_lp_per_mm:g} lp/mm; "
            f"wavelength {point.wavelength_um:g} um; {point.orientation}; "
            f"minimum {point.minimum_mtf_fraction:g} (fraction)."
        )
    lines.append("Resolved manufacturing tolerance assumptions:")
    if profile.provisional:
        lines.append("PROVISIONAL tolerance values: unreviewed development assumptions, requiring manufacturing review.")
    for error in profile.errors:
        spread = (f"; standard deviation {error.standard_deviation:g} {error.units}"
                  if error.standard_deviation is not None else "")
        lines.append(
            f"{error.parameter} ({error.applies_to}): {error.distribution}; mean {error.mean:g} {error.units}"
            f"{spread}; hard bounds [{error.lower_bound:g}, {error.upper_bound:g}] {error.units}."
        )
    lines += [
        f"Error correlations: {profile.correlations}",
        f"Final refocus allowed: {'yes' if profile.refocus.allowed else 'no'}; "
        f"sensor shift bound +/- {profile.refocus.maximum_sensor_shift_mm:g} mm; "
        f"objective: {profile.refocus.objective}.",
    ]
    for candidate in case.candidates:
        metrics = candidate.nominal_metrics
        lines += ["", f"Candidate ID {candidate.candidate_id}:", f"Layout: {candidate.layout}",
                  f"Glass types, object to sensor: {', '.join(candidate.glass_types)}."]
        geometry = candidate.geometry
        if geometry is None:
            lines.append("Geometry summary: unavailable.")
        else:
            lines += [
                f"Elements: {geometry.element_count}.",
                f"Minimum edge thickness: {geometry.minimum_edge_thickness_mm:g} mm.",
                f"Minimum air gap: {geometry.minimum_air_gap_mm:g} mm.",
                f"Maximum clear aperture: {geometry.maximum_clear_aperture_mm:g} mm.",
                f"Maximum absolute surface curvature: {geometry.maximum_absolute_curvature_per_mm:g} 1/mm.",
            ]
        lines += [f"Nominal effective focal length: {metrics.effective_focal_length_mm:g} mm.",
                  f"Nominal first-surface-to-sensor distance: {metrics.first_surface_to_sensor_mm:g} mm.",
                  f"Nominal f-number: {metrics.f_number:g} (dimensionless)."]
        for point in metrics.mtf:
            lines.append(
                f"Nominal MTF: field {point.field_angle_deg} deg; {point.spatial_frequency_lp_per_mm:g} lp/mm; "
                f"wavelength {point.wavelength_um:g} um; {point.orientation}; "
                f"{point.mtf_fraction:g} (fraction)."
            )
    lines += ["", 'Return only JSON with two different candidate IDs, for example: {"starter_ids":[3,7]}',
              "The example illustrates the output format only. Choose from this brief's candidate IDs."]
    return "\n".join(lines)
