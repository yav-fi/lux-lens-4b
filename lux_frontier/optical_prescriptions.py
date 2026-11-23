"""Content-addressed Cooke development prescriptions and PROVISIONAL profiles."""

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from lux_frontier.contracts import ArtifactRef, Contract, Positive
from lux_frontier.prompting import ResolvedToleranceProfile

PROTOCOL = "cooke-optical-v1"
CORRELATIONS = "Independent across error parameters, surfaces, gaps and elements; x/y draws independent; each element's two surfaces share one rigid transform."
REFOCUS_OBJECTIVE = "maximize the minimum required MTF-to-threshold ratio at one common physical sensor position"
SCOPES = {
    "radius_error": "fractional radius change, independently on each of six spherical surfaces",
    "thickness_error": "center thickness, independently on each of three elements",
    "air_gap_error": "axial gap, independently between elements 1/2 and 2/3; sensor initially fixed",
    "decenter": "rigid whole element, independent x/y; both surfaces move together",
    "tilt": "rigid whole element, independent rotations about x/y at its front vertex",
}


def digest(value) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Prescription(Contract):
    protocol: Literal["cooke-optical-v1"] = PROTOCOL
    source: Literal["optiland-0.6.3:CookeTriplet"] = "optiland-0.6.3:CookeTriplet"
    radii_mm: tuple[float, float, float, float, float, float]
    # Surface 1..5: glass, air, glass, air, glass. Sensor is separately fixed.
    spacings_mm: tuple[Positive, Positive, Positive, Positive, Positive]
    sensor_z_mm: Positive
    stop_diameter_mm: Positive
    clear_semidiameters_mm: tuple[Positive, Positive, Positive]
    glasses: tuple[tuple[str, str], tuple[str, str], tuple[str, str]] = (
        ("SK16", "hikari"), ("F2", "schott"), ("SK16", "hikari"))
    fields_deg: tuple[tuple[float, float], ...] = ((0, 0), (0, 14), (0, 20))
    wavelengths_um: tuple[Positive, ...] = (0.48, 0.55, 0.65)
    primary_wavelength_um: Positive = 0.55
    material_sha256: tuple[str, str, str]

    @model_validator(mode="after")
    def supported_prescription(self) -> Self:
        if any(r == 0 for r in self.radii_mm):
            raise ValueError("zero spherical radius is invalid")
        if self.primary_wavelength_um not in self.wavelengths_um:
            raise ValueError("primary wavelength must be present")
        # The pinned FFT slices are fixed pupil Y/X, not arbitrarily rotated
        # field-meridian slices. Restrict this development protocol to +Y fields.
        if not self.fields_deg or any(x != 0 or y < 0 or y >= 85 for x, y in self.fields_deg):
            raise ValueError("development MTF supports only +Y meridional fields below 85 deg")
        if (0, 0) not in self.fields_deg:
            raise ValueError("on-axis field is required")
        if any(len(key) != 64 or any(c not in "0123456789abcdef" for c in key) for key in self.material_sha256):
            raise ValueError("each glass requires its immutable catalog data SHA-256")
        return self

    @property
    def reference(self) -> ArtifactRef:
        return ArtifactRef(uri="local://lux/prescriptions/" + digest(self), revision=PROTOCOL)


class DevelopmentProfile(ResolvedToleranceProfile):
    protocol: Literal["cooke-optical-v1"] = PROTOCOL
    label: Literal["development", "stress-test", "zero-test"] = "development"
    correlation_mode: Literal["independent_with_rigid_elements"] = "independent_with_rigid_elements"
    tilt_pivot: Literal["front_vertex"] = "front_vertex"
    sensor_mount: Literal["fixed_to_first_vertex"] = "fixed_to_first_vertex"

    @model_validator(mode="after")
    def supported_settings(self) -> Self:
        parameters = [e.parameter for e in self.errors]
        if len(parameters) != len(SCOPES) or set(parameters) != set(SCOPES):
            raise ValueError("unsupported optical profile: require exactly radius, center thickness, air gap, decenter and tilt errors")
        if self.correlations != CORRELATIONS:
            raise ValueError("unsupported optical error correlations")
        if self.refocus.objective != REFOCUS_OBJECTIVE:
            raise ValueError("unsupported refocus objective")
        for error in self.errors:
            if error.applies_to != SCOPES[error.parameter]:
                raise ValueError(f"unsupported error scope for {error.parameter}")
            if error.parameter == "tilt" and max(abs(error.lower_bound), abs(error.upper_bound)) > 0.25:
                raise ValueError("unsupported profile: tilt bounds exceed the 0.25 deg development domain")
            if error.parameter == "decenter" and max(abs(error.lower_bound), abs(error.upper_bound)) > 0.2:
                raise ValueError("unsupported profile: decenter bounds exceed the 0.2 mm development domain")
        return self


def profile_payload(profile: DevelopmentProfile) -> dict:
    return profile.model_dump(mode="json", exclude={"reference"})


def profile_reference(payload: dict) -> ArtifactRef:
    return ArtifactRef(uri="local://lux/tolerance-profiles/" + digest(payload), revision=PROTOCOL)


def verify_profile(profile: DevelopmentProfile) -> None:
    if profile.reference != profile_reference(profile_payload(profile)):
        raise ValueError("tolerance profile contents do not match immutable reference")


def development_profile(*, stress: bool = False) -> DevelopmentProfile:
    scale = 5 if stress else 1
    payload = dict(protocol=PROTOCOL, label="stress-test" if stress else "development", provisional=True,
                   correlations=CORRELATIONS, correlation_mode="independent_with_rigid_elements",
                   tilt_pivot="front_vertex", sensor_mount="fixed_to_first_vertex",
                   refocus=dict(allowed=True, maximum_sensor_shift_mm=0.2, objective=REFOCUS_OBJECTIVE), errors=[])
    specs = [("radius_error", "dimensionless", "truncated_normal", 0.003, 0.001),
             ("thickness_error", "mm", "uniform", 0.02, None),
             ("air_gap_error", "mm", "uniform", 0.03, None),
             ("decenter", "mm", "truncated_normal", 0.03, 0.01),
             ("tilt", "deg", "uniform", 0.03, None)]
    for name, units, distribution, bound, sigma in specs:
        payload["errors"].append(dict(parameter=name, units=units, applies_to=SCOPES[name],
                                      distribution=distribution, lower_bound=-bound * scale,
                                      upper_bound=bound * scale, mean=0,
                                      standard_deviation=None if sigma is None else sigma * scale))
    # Normalize defaults through the typed model before addressing its contents.
    provisional = DevelopmentProfile.model_validate(dict(payload, reference=dict(uri="pending", revision=PROTOCOL)))
    return DevelopmentProfile.model_validate(dict(profile_payload(provisional), reference=profile_reference(profile_payload(provisional))))


class ArtifactStore:
    """Local immutable JSON artifacts; URI resolution verifies the entire content."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def put(self, prescription: Prescription) -> ArtifactRef:
        ref = prescription.reference
        path = self.directory / (digest(prescription) + ".json")
        if path.exists():
            if Prescription.model_validate_json(path.read_text()) != prescription:
                raise ValueError("immutable prescription artifact collision")
        else:
            path.write_text(prescription.model_dump_json(indent=2) + "\n")
        return ref

    def resolve(self, ref: ArtifactRef) -> Prescription:
        if ref.revision != PROTOCOL or not ref.uri.startswith("local://lux/prescriptions/"):
            raise ValueError(f"unsupported prescription reference: {ref}")
        key = ref.uri.rsplit("/", 1)[-1]
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("invalid prescription content address")
        path = self.directory / (key + ".json")
        if not path.is_file():
            raise ValueError(f"required prescription cannot be resolved: {ref.uri}")
        prescription = Prescription.model_validate_json(path.read_text())
        if prescription.reference != ref:
            raise ValueError("prescription contents do not match immutable reference")
        return prescription
