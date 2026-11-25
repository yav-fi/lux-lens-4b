"""Air-spaced, spherical, rotationally symmetric V2 lens designs."""

from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import math
from pathlib import Path
from statistics import NormalDist
from typing import Annotated
import warnings

import numpy as np
from pydantic import Field, model_validator

from lux_frontier.contracts import Brief, Contract, Fraction, NominalMetrics, Positive
from lux_frontier.optical_engine import (
    BuildErrors, InvalidBuild, constraint_failures, measure_optic, normalized_field,
    pinned_backend, rotation_matrix, sag,
)

from optiland.optic import Optic
from optiland.paraxial_path import ParaxialDomainWarning
from optiland.physical_apertures import RadialAperture
from optiland.samples.objectives import CookeTriplet, TripletTelescopeObjective
from optiland.samples.simple import TelescopeDoublet

OPTICS_VERSION = "lux-spherical-air-spaced-2.0"


class Lens(Contract):
    ancestry: str
    radii_mm: tuple[float, ...]
    spacings_mm: tuple[Positive, ...]
    sensor_z_mm: Positive
    clear_semidiameters_mm: tuple[Positive, ...]
    stop_surface: Annotated[int, Field(strict=True, gt=0)]
    stop_diameter_mm: Positive
    glasses: tuple[tuple[str, str], ...]
    material_sha256: tuple[str, ...]
    fields_deg: tuple[tuple[float, float], ...] = ((0, 0), (0, 1), (0, 3), (0, 7), (0, 14), (0, 20))
    wavelengths_um: tuple[Positive, ...] = (0.48, 0.55, 0.65)
    primary_wavelength_um: Positive = 0.55

    @model_validator(mode="after")
    def topology(self):
        n = len(self.glasses)
        if n not in (2, 3) or (len(self.radii_mm), len(self.spacings_mm),
                              len(self.clear_semidiameters_mm), len(self.material_sha256)) != (2*n, 2*n-1, n, n):
            raise ValueError("expected an air-spaced doublet or triplet")
        if self.stop_surface > 2*n or self.stop_diameter_mm / 2 > self.clear_semidiameters_mm[(self.stop_surface-1)//2]:
            raise ValueError("stop must lie on a glass surface within its clear aperture")
        if ((0, 0) not in self.fields_deg or any(x != 0 or not 0 <= y < 85 for x, y in self.fields_deg)
                or self.primary_wavelength_um not in self.wavelengths_um):
            raise ValueError("expected on-axis and +Y fields and a configured primary wavelength")
        clearance(self)
        return self


class Requirements(Brief):
    minimum_pupil_throughput: Fraction = 0.65


@dataclass(frozen=True)
class Profile:
    name: str
    radius_sigma: float
    thickness_bound_mm: float
    gap_bound_mm: float
    decenter_sigma_mm: float
    tilt_bound_deg: float
    focus_travel_mm: float
    focus_increment_mm: float

    def record(self):
        return dict(**asdict(self), provisional=True,
                    description="Independent per surface/element/gap and axis. Radius fractions and decenter use "
                    "zero-mean normal distributions truncated at 3 sigma. Center thickness, gap and tilt use "
                    "zero-centered uniform distributions. Both faces move rigidly about the front vertex. "
                    "The sensor is housing-fixed with only the stated axial travel and increment. "
                    "Focus minimizes the on-axis 0.55 µm geometric RMS spot on a finite pupil sample.")


PROFILES = {
    p.name: p for p in (
        Profile("precision-fixed", .00018, .003, .003, .002, .009, 0, .01),
        Profile("standard-adjusted", .0007, .01, .008, .007, .028, .12, .015),
        Profile("standard-fixed", .0007, .01, .008, .007, .028, 0, .015),
        Profile("economy-adjusted", .0016, .018, .012, .018, .065, .16, .04),
    )
}


def clearance(p):
    """Exact minimum coaxial axial separation across the shared clear disk.

    For two spheres of fixed radii the sag difference is monotone for r > 0.
    Its derivative difference can only vanish identically or at the vertex.
    """
    limits = np.repeat(p.clear_semidiameters_mm, 2).tolist()
    edge = [sag(r, a) for r, a in zip(p.radii_mm, limits, strict=True)]
    thicknesses = [min(t, t + edge[i+1] - edge[i]) for i, t in enumerate(p.spacings_mm) if i % 2 == 0]
    gaps = [min(t, t + sag(p.radii_mm[i+1], min(limits[i:i+2])) - sag(p.radii_mm[i], min(limits[i:i+2])))
            for i, t in enumerate(p.spacings_mm) if i % 2]
    back = p.sensor_z_mm - sum(p.spacings_mm) - max(0, edge[-1])
    if min(*thicknesses, *gaps, back) <= 0:
        raise InvalidBuild("geometry:nonpositive_thickness_or_clearance")
    return dict(minimum_glass_thickness_mm=min(thicknesses), minimum_air_clearance_mm=min(gaps),
                minimum_sensor_clearance_mm=back)


@lru_cache(maxsize=64)
def file_hash(filename):
    return sha256(Path(filename).read_bytes()).hexdigest()


def zero_errors(p):
    n = len(p.glasses)
    return BuildErrors((0.,)*(2*n), (0.,)*n, (0.,)*(n-1), ((0., 0.),)*n, ((0., 0.),)*n)


def build_optic(p: Lens, errors: BuildErrors | None = None, sensor_shift_mm=0.0):
    pinned_backend()
    errors = errors or zero_errors(p)
    n = len(p.glasses)
    if tuple(map(len, asdict(errors).values())) != (2*n, n, n-1, n, n):
        raise ValueError("manufacturing errors do not match prescription topology")
    radii = tuple(r*(1+d) for r, d in zip(p.radii_mm, errors.radius_fraction, strict=True))
    spacings = list(p.spacings_mm)
    spacings[::2] = [t+e for t, e in zip(spacings[::2], errors.center_thickness_mm, strict=True)]
    spacings[1::2] = [t+e for t, e in zip(spacings[1::2], errors.air_gap_mm, strict=True)]
    sensor = p.sensor_z_mm + sensor_shift_mm
    trial = p.model_copy(update=dict(radii_mm=radii, spacings_mm=tuple(spacings), sensor_z_mm=sensor))
    geometry = clearance(trial)
    # Conservative bound on lost air clearance from translating curved surfaces
    # and from rigid tilt. Use the maximum sag slope over the participating rims.
    for e in range(n-1):
        back, front = 2*e+1, 2*e+2
        aperture = p.clear_semidiameters_mm[e:e+2]
        tilt = [math.radians(math.hypot(*v)) for v in errors.tilt_deg[e:e+2]]
        shift = math.dist(*errors.decenter_mm[e:e+2]) + spacings[back-1]*math.sin(tilt[0])
        slopes = [a/math.sqrt(r*r-a*a) for a, r in zip(aperture, radii[back:front+1], strict=True)]
        sags = [abs(sag(r, a)) for a, r in zip(aperture, radii[back:front+1], strict=True)]
        shift += sum(a*(1-math.cos(t)) + s*math.sin(t) for a, s, t in zip(aperture, sags, tilt, strict=True))
        axial = sum(a*math.sin(t) + s*(1-math.cos(t)) for a, s, t in zip(aperture, sags, tilt, strict=True))
        axial += spacings[back-1]*(1-math.cos(tilt[0]))
        common = min(aperture)
        gap = min(spacings[back], spacings[back]+sag(radii[front], common)-sag(radii[back], common))
        guaranteed = gap - axial - max(slopes)*shift
        if guaranteed <= 0:
            raise InvalidBuild("geometry:perturbed_clearance_not_certified")
        geometry["minimum_air_clearance_mm"] = min(geometry["minimum_air_clearance_mm"], guaranteed)
    rear_tilt = math.radians(math.hypot(*errors.tilt_deg[-1]))
    geometry["minimum_sensor_clearance_mm"] -= (p.clear_semidiameters_mm[-1]*math.sin(rear_tilt)
        + abs(sag(radii[-1], p.clear_semidiameters_mm[-1]))*(1-math.cos(rear_tilt)))
    if geometry["minimum_sensor_clearance_mm"] <= 0:
        raise InvalidBuild("geometry:perturbed_sensor_clearance_not_certified")

    lens = Optic()
    lens.surfaces.add(index=0, thickness=math.inf)
    for k, radius in enumerate(radii):
        e = k // 2
        aperture = p.clear_semidiameters_mm[e]
        stop = k+1 == p.stop_surface
        kwargs = {"material": p.glasses[e]} if k % 2 == 0 else {}
        lens.surfaces.add(index=k+1, radius=radius, thickness=([*spacings, sensor-sum(spacings)])[k],
                          is_stop=stop, aperture=RadialAperture(r_max=p.stop_diameter_mm/2 if stop else aperture), **kwargs)
        if k % 2 == 0 and file_hash(lens.surfaces[k+1].material_post.filename) != p.material_sha256[e]:
            raise ValueError("glass catalog fingerprint differs")
    lens.surfaces.add(index=2*n+1)
    lens.set_aperture(aperture_type="float_by_stop_size", value=p.stop_diameter_mm)
    lens.fields.set_type(field_type="angle")
    for x, y in p.fields_deg:
        lens.fields.add(x=x, y=y)
    for wave in p.wavelengths_um:
        lens.wavelengths.add(value=wave, is_primary=wave == p.primary_wavelength_um)
    z = 0.
    for e, (decenter, tilt) in enumerate(zip(errors.decenter_mm, errors.tilt_deg, strict=True)):
        back_axis = rotation_matrix(tilt) @ [0., 0., spacings[2*e]]
        for offset, k in ((np.zeros(3), 2*e+1), (back_axis, 2*e+2)):
            cs = lens.surfaces[k].geometry.cs
            cs.x, cs.y, cs.z = np.asarray([*decenter, z]) + offset
            cs.rx, cs.ry = np.deg2rad(tilt)
        z += sum(spacings[2*e:2*e+2])
    lens.image_surface.geometry.cs.z = sensor
    return lens, geometry


REFERENCE_CLASSES = (CookeTriplet, TelescopeDoublet, TripletTelescopeObjective)


@lru_cache
def reference(name):
    pinned_backend()
    sample = next(cls for cls in REFERENCE_CLASSES if cls.__name__ == name)()
    surfaces = list(sample.surfaces)[1:-1]
    glass = surfaces[::2]
    stop = next(k+1 for k, s in enumerate(surfaces) if s.is_stop)
    y, _ = sample.paraxial.trace_generic(1., 0., -1, sample.primary_wavelength)
    diameter = abs(float(np.asarray(sample.paraxial.EPD()*y[stop]).item()))
    files = [Path(s.material_post.filename) for s in glass]
    # Front and rear clear apertures are development assumptions, documented in the research log.
    apertures = {"CookeTriplet": (9., 4.5, 8.), "TelescopeDoublet": (3.5, 3.5),
                 "TripletTelescopeObjective": (18., 18., 18.)}[name]
    if name == "TripletTelescopeObjective":
        diameter *= .9  # stop down the reference for this modeled aperture stack
    return Lens(ancestry=f"optiland-0.6.3:{name}", radii_mm=[float(s.geometry.radius) for s in surfaces],
                spacings_mm=[float(s.thickness) for s in surfaces[:-1]], sensor_z_mm=float(sample.image_surface.geometry.cs.z),
                clear_semidiameters_mm=apertures, stop_surface=stop, stop_diameter_mm=diameter,
                glasses=[(f.stem, f.parent.name) for f in files], material_sha256=[file_hash(f) for f in files])


def pupil(count):
    """Deterministic equal-area disk quadrature."""
    k = np.arange(count)
    angle = k * math.pi * (3 - math.sqrt(5))
    r = np.sqrt((k + .5) / count)
    return r*np.cos(angle), r*np.sin(angle)


def trace_field(lens, p, field, wave, count=61):
    hx, hy = normalized_field(p, field)
    px, py = pupil(count)
    # The grading measurement records this advisory once. These extra real-ray
    # bundles use the same paraxial initialization and rigid physical trace.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ParaxialDomainWarning)
        return lens.trace_generic(Hx=hx, Hy=hy, Px=px, Py=py, wavelength=wave)


def measurements(p, brief, samples, *, errors=None, focus=0., optic=None, curves=False):
    lens, geometry = optic if optic is not None else build_optic(p, errors, focus)
    metrics, details = measure_optic(lens, p, brief, pupil_samples=samples, sensor_shift_mm=focus, curves=curves)
    throughput = []
    for field, wave in dict.fromkeys((t.field_angle_deg, t.wavelength_um) for t in brief.mtf_thresholds):
        rays = trace_field(lens, p, field, wave)
        fraction = float(np.mean((rays.i > 0) & np.isfinite(rays.x) & np.isfinite(rays.y)))
        throughput.append(dict(field_deg=field, wavelength_um=wave, pupil_fraction=fraction))
    return dict(metrics=metrics.model_dump(mode="json"), **geometry, throughput=throughput, trace=details)


def failures(measured, brief):
    result = constraint_failures(NominalMetrics.model_validate(measured["metrics"]), brief)
    result.extend(dict(constraint="pupil_throughput", minimum=brief.minimum_pupil_throughput, **row)
                  for row in measured["throughput"] if row["pupil_fraction"] < brief.minimum_pupil_throughput)
    return result


def paraxial_matrix(p, lens):
    """Independent reduced-angle ABCD product with catalog indices from Optiland."""
    matrix = np.eye(2)
    previous = 1.
    for k, radius in enumerate(p.radii_mm):
        post = float(np.asarray(lens.surfaces[k+1].material_post.n(p.primary_wavelength_um)).item())
        matrix = np.array([[1., 0.], [-(post-previous)/radius, 1.]]) @ matrix
        if k < len(p.spacings_mm):
            matrix = np.array([[1., p.spacings_mm[k]/post], [0., 1.]]) @ matrix
        previous = post
    return dict(efl_mm=-1./matrix[1, 0], paraxial_bfl_mm=-matrix[0, 0]/matrix[1, 0],
                matrix=matrix.tolist())


def draw(p, profile, stream):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(stream)))
    n = len(p.glasses)
    # Fixed maximum-size blocks preserve common random numbers across topologies.
    normal = NormalDist()
    low, high = normal.cdf(-3), normal.cdf(3)
    def truncated(count, sigma):
        return tuple(sigma*normal.inv_cdf(v) for v in rng.uniform(low, high, count))
    radius = truncated(6, profile.radius_sigma)
    thickness = tuple(rng.uniform(-profile.thickness_bound_mm, profile.thickness_bound_mm, 3))
    gap = tuple(rng.uniform(-profile.gap_bound_mm, profile.gap_bound_mm, 2))
    decenter = np.reshape(truncated(6, profile.decenter_sigma_mm), (3, 2)).tolist()
    tilt = rng.uniform(-profile.tilt_bound_deg, profile.tilt_bound_deg, (3, 2)).tolist()
    return BuildErrors(radius[:2*n], thickness[:n], gap[:n-1], tuple(map(tuple, decenter[:n])), tuple(map(tuple, tilt[:n])))


def focus_position(lens, p, profile):
    if profile.focus_travel_mm == 0:
        return 0., 0.
    rays = trace_field(lens, p, (0, 0), p.primary_wavelength_um)
    keep = (rays.i > 0) & np.isfinite(rays.x) & np.isfinite(rays.y) & (rays.N != 0)
    if sum(keep) < 9:
        raise InvalidBuild("trace:too_few_focus_rays")
    pos = np.stack((rays.x[keep], rays.y[keep]))
    slope = np.stack((rays.L[keep]/rays.N[keep], rays.M[keep]/rays.N[keep]))
    pos -= pos.mean(axis=1, keepdims=True)
    slope -= slope.mean(axis=1, keepdims=True)
    power = np.sum(slope*slope)
    if power <= 0 or not np.isfinite(power):
        raise InvalidBuild("trace:invalid_focus_ray_slopes")
    ideal = float(-np.sum(pos*slope)/power)
    increment = profile.focus_increment_mm
    steps = math.floor(profile.focus_travel_mm/increment + 1e-12)
    quantized = increment*max(-steps, min(steps, round(ideal/increment)))
    return quantized, ideal


def simulate_build(p, brief, samples, profile, stream):
    errors = draw(p, profile, stream)
    focus, ideal = None, None
    try:
        lens, geometry = build_optic(p, errors)
        focus, ideal = focus_position(lens, p, profile)
        if focus:
            geometry["minimum_sensor_clearance_mm"] += focus
            if geometry["minimum_sensor_clearance_mm"] <= 0:
                raise InvalidBuild("geometry:perturbed_sensor_clearance_not_certified")
            lens.image_surface.geometry.cs.z = p.sensor_z_mm + focus
        measured = measurements(p, brief, samples, focus=focus, optic=(lens, geometry))
        reasons = failures(measured, brief)
    except InvalidBuild as exc:
        measured, reasons = None, [dict(constraint=str(exc))]
    return dict(stream=list(stream), errors=asdict(errors), sensor_shift_mm=focus, unconstrained_best_rms_shift_mm=ideal,
                passed=not reasons, measurements=measured, failures=reasons)
