"""Pinned Optiland physical evaluation; no model SDK or remote services."""

import importlib.metadata
import hashlib
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Avoid a repeated, unwritable home-directory font-cache initialization on CLI use.
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache/matplotlib"))

import optiland.backend as be
from optiland.mtf import FFTMTF
from optiland.optic import Optic
from optiland.paraxial_path import ParaxialDomainWarning
from optiland.physical_apertures import RadialAperture
from optiland.samples.objectives import CookeTriplet
from optiland.utils import get_working_FNO

from lux_frontier.contracts import Brief, GeometrySummary, MTFMeasurement, NominalMetrics
from lux_frontier.optical_prescriptions import Prescription


class InvalidBuild(ValueError):
    """Identified physical/trace invalidity; counts in the yield denominator."""


@dataclass(frozen=True)
class BuildErrors:
    radius_fraction: tuple[float, ...] = (0.0,) * 6
    center_thickness_mm: tuple[float, ...] = (0.0,) * 3
    air_gap_mm: tuple[float, ...] = (0.0,) * 2
    decenter_mm: tuple[tuple[float, float], ...] = ((0.0, 0.0),) * 3
    tilt_deg: tuple[tuple[float, float], ...] = ((0.0, 0.0),) * 3


def pinned_backend() -> None:
    if importlib.metadata.version("optiland") != "0.6.3":
        raise RuntimeError("optical protocol requires exactly optiland==0.6.3")
    be.set_backend("numpy")


def cooke_prescription() -> Prescription:
    pinned_backend()
    lens = CookeTriplet()
    y, _ = lens.paraxial.trace_generic(1.0, 0.0, -1, lens.primary_wavelength)
    stop_diameter = float(np.asarray(lens.paraxial.EPD() * y[4]).item())
    return Prescription(
        radii_mm=tuple(float(lens.surfaces[i].geometry.radius) for i in range(1, 7)),
        spacings_mm=tuple(float(lens.surfaces[i].thickness) for i in range(1, 6)),
        sensor_z_mm=float(lens.image_surface.geometry.cs.z), stop_diameter_mm=stop_diameter,
        clear_semidiameters_mm=(9.0, 4.5, 8.0),
        material_sha256=tuple(hashlib.sha256(Path(lens.surfaces[i].material_post.filename).read_bytes()).hexdigest()
                              for i in (1, 3, 5)),
    )


def sag(radius: float, height: float) -> float:
    if abs(radius) <= height:
        raise InvalidBuild("geometry:spherical_clear_aperture_exceeds_radius")
    return radius - math.copysign(math.sqrt(radius * radius - height * height), radius)


def geometry_summary(prescription: Prescription) -> GeometrySummary:
    edges = []
    for e, aperture in enumerate(prescription.clear_semidiameters_mm):
        edges.append(prescription.spacings_mm[2 * e] + sag(prescription.radii_mm[2 * e + 1], aperture)
                     - sag(prescription.radii_mm[2 * e], aperture))
    if min(edges) <= 0:
        raise InvalidBuild("geometry:nonpositive_edge_thickness")
    return GeometrySummary(element_count=3, minimum_edge_thickness_mm=min(edges),
                           minimum_air_gap_mm=min(prescription.spacings_mm[1::2]),
                           maximum_clear_aperture_mm=2 * max(prescription.clear_semidiameters_mm),
                           maximum_absolute_curvature_per_mm=max(abs(1 / r) for r in prescription.radii_mm))


def rotation_matrix(tilt_deg: tuple[float, float]) -> np.ndarray:
    """Optiland globalize order: rotate X, then Y; radians at its API boundary."""
    ax, ay = np.deg2rad(tilt_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    return ry @ rx


def make_optic(prescription: Prescription, errors: BuildErrors | None = None, sensor_shift_mm: float = 0) -> Optic:
    pinned_backend()
    errors = errors or BuildErrors()
    if (len(errors.radius_fraction), len(errors.center_thickness_mm), len(errors.air_gap_mm),
            len(errors.decenter_mm), len(errors.tilt_deg)) != (6, 3, 2, 3, 3):
        raise ValueError("build error array lengths do not match the three-element prescription")
    radii = [r * (1 + d) for r, d in zip(prescription.radii_mm, errors.radius_fraction, strict=True)]
    spacings = list(prescription.spacings_mm)
    for e, delta in enumerate(errors.center_thickness_mm):
        spacings[e * 2] += delta
    for e, delta in enumerate(errors.air_gap_mm):
        spacings[e * 2 + 1] += delta
    if not all(math.isfinite(v) for v in [*radii, *spacings, sensor_shift_mm]):
        raise InvalidBuild("geometry:nonfinite_parameters")
    if min(spacings) <= 0:
        raise InvalidBuild("geometry:nonpositive_center_thickness_or_air_gap")
    perturbed = prescription.model_copy(update={"radii_mm": tuple(radii), "spacings_mm": tuple(spacings)})
    geometry_summary(perturbed)
    sensor = prescription.sensor_z_mm + sensor_shift_mm
    if sensor <= sum(spacings):
        raise InvalidBuild("geometry:sensor_before_last_surface")
    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    for index in range(1, 7):
        element = (index - 1) // 2
        thickness = spacings[index - 1] if index < 6 else sensor - sum(spacings)
        aperture = prescription.clear_semidiameters_mm[element]
        if index == 4:
            aperture = min(aperture, prescription.stop_diameter_mm / 2)
        kwargs = dict(index=index, radius=radii[index - 1], thickness=thickness,
                      is_stop=index == 4, aperture=RadialAperture(r_max=aperture))
        if index % 2:
            kwargs["material"] = prescription.glasses[element]
        lens.surfaces.add(**kwargs)
    lens.surfaces.add(index=7)
    for e, index in enumerate((1, 3, 5)):
        material_hash = hashlib.sha256(Path(lens.surfaces[index].material_post.filename).read_bytes()).hexdigest()
        if material_hash != prescription.material_sha256[e]:
            raise ValueError("glass catalog data do not match the immutable prescription fingerprint")
    lens.set_aperture(aperture_type="float_by_stop_size", value=prescription.stop_diameter_mm)
    lens.fields.set_type(field_type="angle")
    for x, y in prescription.fields_deg:
        lens.fields.add(x=x, y=y)
    for wavelength in prescription.wavelengths_um:
        lens.wavelengths.add(value=wavelength, is_primary=wavelength == prescription.primary_wavelength_um)
    # First apply all axial stack errors. Then rotate the entire glass element
    # about its FRONT vertex. The back vertex rotates too: merely assigning the
    # same rx/ry to two fixed vertices would shear the lens instead of moving it.
    front_z = 0.0
    for e in range(3):
        rotation = rotation_matrix(errors.tilt_deg[e])
        displacement = np.array([*errors.decenter_mm[e], front_z])
        for offset, index in [(0.0, 2 * e + 1), (spacings[2 * e], 2 * e + 2)]:
            vertex = displacement + rotation @ np.array([0.0, 0.0, offset])
            cs = lens.surfaces[index].geometry.cs
            cs.x, cs.y, cs.z = vertex
            cs.rx, cs.ry = np.deg2rad(errors.tilt_deg[e])
        if e < 2:
            front_z += spacings[2 * e] + spacings[2 * e + 1]
    # Sensor is a single global plane fixed to the housing/first-vertex datum,
    # independent of all stack errors until the one allowed refocus is applied.
    lens.image_surface.geometry.cs.z = sensor
    return lens


def normalized_field(prescription: Prescription, field: tuple[float, float]) -> tuple[float, float]:
    if field not in prescription.fields_deg:
        raise ValueError(f"brief requests an unconfigured degree-valued field: {field}")
    maximum = max(math.hypot(x, y) for x, y in prescription.fields_deg)
    return (field[0] / maximum, field[1] / maximum) if maximum else (0.0, 0.0)


def mtf_value(calculator, field_index: int, orientation: str, frequency: float) -> tuple[float, dict]:
    # Verified against optiland 0.6.3 mtf/fft.py: T is pupil Y (index 0), S is
    # pupil X (index 1). Each has its OWN calibrated image-space cycles/mm axis.
    if orientation == "tangential":
        index, axis = 0, calculator.freq_tang[field_index]
    elif orientation == "sagittal":
        index, axis = 1, calculator.freq_sag[field_index]
    else:
        raise ValueError("unsupported MTF orientation")
    axis, values = np.asarray(axis), np.asarray(calculator.mtf[field_index][index])
    if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0):
        raise InvalidBuild("trace:invalid_MTF_frequency_axis")
    if frequency < axis[0] or frequency > axis[-1]:
        raise ValueError("requested frequency exceeds sampled MTF axis; increase pupil sampling")
    if not np.all(np.isfinite(values)) or values[0] <= 0:
        raise InvalidBuild("trace:no_finite_MTF_power")
    return float(np.interp(frequency, axis, values)), dict(
        orientation_index=index, frequency_axis_step_lp_per_mm=float(axis[1] - axis[0]),
        maximum_sampled_frequency_lp_per_mm=float(axis[-1]))


def measure(prescription: Prescription, brief: Brief, *, pupil_samples: int = 64,
            errors: BuildErrors | None = None, sensor_shift_mm: float = 0) -> tuple[NominalMetrics, dict]:
    lens = make_optic(prescription, errors, sensor_shift_mm)
    eps = 1e-4
    diagnostics = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ParaxialDomainWarning)
        # Small real-ray bundles measure differential on-axis optical power.
        # Unlike scalar f2(), these traces include actual tilted/decentered geometry.
        rays = lens.trace_generic(Hx=0.0, Hy=0.0, Px=np.array([eps, -eps, 0, 0]),
                                  Py=np.array([0, 0, eps, -eps]), wavelength=prescription.primary_wavelength_um)
        if not np.all(rays.i > 0) or not np.all(np.isfinite(rays.N)):
            raise InvalidBuild("trace:invalid_differential_focal_rays")
        height = eps * float(np.asarray(lens.paraxial.EPD()).item()) / 2
        powers = np.array([(rays.L[0] / rays.N[0] - rays.L[1] / rays.N[1]) / (2 * height),
                           (rays.M[2] / rays.N[2] - rays.M[3] / rays.N[3]) / (2 * height)])
        focal_axes = -1 / powers
        focal = float(np.mean(focal_axes))
        f_number = float(np.asarray(get_working_FNO(lens, (0, 0), prescription.primary_wavelength_um)).item())
        if not np.all(np.isfinite(focal_axes)) or min(focal_axes) <= 0 or not math.isfinite(f_number):
            raise InvalidBuild("trace:nonpositive_or_nonfinite_focal_length_or_f_number")
        measurements = []
        calculators = {}
        for target in brief.mtf_thresholds:
            if target.wavelength_um not in prescription.wavelengths_um:
                raise ValueError(f"unconfigured wavelength {target.wavelength_um} um")
            field = normalized_field(prescription, target.field_angle_deg)
            if target.wavelength_um not in calculators:
                fields = list(dict.fromkeys(normalized_field(prescription, p.field_angle_deg)
                                            for p in brief.mtf_thresholds if p.wavelength_um == target.wavelength_um))
                try:
                    calculator = FFTMTF(lens, fields=fields, wavelength=target.wavelength_um,
                                        num_rays=pupil_samples, grid_size=pupil_samples * 4,
                                        strategy="chief_ray", remove_tilt=False)
                except ValueError as exc:
                    if str(exc).startswith("No valid ray samples found for chief-ray wavefront"):
                        raise InvalidBuild("trace:no_valid_wavefront_rays") from exc
                    raise  # unknown library errors are software errors, not bad builds
                if calculator.resolved_fields != fields or calculator.resolved_wavelength != target.wavelength_um:
                    raise RuntimeError("pinned API field/wavelength resolution changed")
                calculators[target.wavelength_um] = calculator
            calculator = calculators[target.wavelength_um]
            index = calculator.resolved_fields.index(field)
            value, details = mtf_value(calculator, index, target.orientation, target.spatial_frequency_lp_per_mm)
            measurements.append(MTFMeasurement(**target.model_dump(exclude={"minimum_mtf_fraction"}), mtf_fraction=value))
            diagnostics.append(dict(field_deg=target.field_angle_deg, normalized_field=field,
                                    wavelength_um=target.wavelength_um, **details))
        advisory_codes = sorted({type(w.message).__name__ for w in caught})
    metrics = NominalMetrics(effective_focal_length_mm=focal, first_surface_to_sensor_mm=float(lens.image_surface.geometry.cs.z),
                             f_number=f_number, mtf=tuple(measurements))
    return metrics, dict(sensor_z_mm=float(lens.image_surface.geometry.cs.z), sensor_shift_mm=sensor_shift_mm,
                         differential_focal_axes_mm=focal_axes.tolist(), field_mapping=diagnostics,
                         optiland_advisory_types=advisory_codes, pupil_samples=pupil_samples,
                         fft_grid_size=pupil_samples * 4)


def constraint_failures(metrics: NominalMetrics, brief: Brief) -> list[dict]:
    failures = []
    if abs(metrics.effective_focal_length_mm - brief.target_focal_length_mm) > brief.focal_length_tolerance_mm:
        failures.append(dict(constraint="effective_focal_length_mm", measured=metrics.effective_focal_length_mm,
                             target=brief.target_focal_length_mm, tolerance=brief.focal_length_tolerance_mm))
    if abs(metrics.f_number - brief.target_f_number) > brief.f_number_tolerance:
        failures.append(dict(constraint="f_number", measured=metrics.f_number,
                             target=brief.target_f_number, tolerance=brief.f_number_tolerance))
    if metrics.first_surface_to_sensor_mm > brief.max_first_surface_to_sensor_mm:
        failures.append(dict(constraint="first_surface_to_sensor_mm", measured=metrics.first_surface_to_sensor_mm,
                             maximum=brief.max_first_surface_to_sensor_mm))
    if len(metrics.mtf) != len(brief.mtf_thresholds):
        raise ValueError("metrics do not cover every requested MTF threshold")
    for i, (measured, target) in enumerate(zip(metrics.mtf, brief.mtf_thresholds, strict=True)):
        if measured.model_dump(exclude={"mtf_fraction"}) != target.model_dump(exclude={"minimum_mtf_fraction"}):
            raise ValueError("MTF field, wavelength, frequency or orientation mismatch")
        if measured.mtf_fraction < target.minimum_mtf_fraction:
            failures.append(dict(constraint=f"mtf[{i}]", measured=measured.mtf_fraction,
                                 minimum=target.minimum_mtf_fraction, point=target.model_dump(mode="json")))
    return failures
