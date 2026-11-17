"""CPU smoke check using Optiland's real CookeTriplet, never the synthetic case."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
import optiland.backend as be
from optiland.samples.objectives import CookeTriplet


@dataclass(frozen=True)
class TraceSummary:
    effective_focal_length_mm: float
    wavelength_um: float
    rays_traced: int
    rays_reached_image: int
    max_image_radius_mm: float


def trace_cooke_triplet() -> TraceSummary:
    be.set_backend("numpy")  # explicit CPU backend; no torch or GPU required
    lens = CookeTriplet()
    # Select an on-axis field already defined by the built-in fixture.
    hx, hy = next(field for field in lens.fields.get_field_coords() if field == (0, 0))
    wavelength = float(lens.wavelengths.primary_wavelength.value)
    efl = float(lens.paraxial.f2())
    # Unlike hexapolar's ring count, line_y's num_rays is the ray count.
    rays = lens.trace(Hx=hx, Hy=hy, wavelength=wavelength,
                      num_rays=9, distribution="line_y")
    image_z = float(lens.surfaces[-1].geometry.cs.z)
    reached = (np.isfinite(rays.x) & np.isfinite(rays.y) & np.isfinite(rays.z)
               & (rays.i > 0) & np.isclose(rays.z, image_z, rtol=0, atol=1e-8))
    count = int(np.count_nonzero(reached))
    if not isfinite(efl) or efl <= 0 or len(rays.x) != 9 or count != 9:
        raise RuntimeError(f"broken Cooke trace: EFL={efl}, image rays={count}/{len(rays.x)}")
    radius = float(np.max(np.hypot(rays.x[reached], rays.y[reached])))
    return TraceSummary(efl, wavelength, len(rays.x), count, radius)


def main() -> None:
    result = trace_cooke_triplet()
    print("Optiland CookeTriplet; backend=numpy (CPU); field=(0, 0) deg")
    print(f"Wavelength: {result.wavelength_um:.3f} um")
    print(f"Effective focal length: {result.effective_focal_length_mm:.6f} mm")
    print(f"Rays reaching image plane: {result.rays_reached_image}/{result.rays_traced}")
    print(f"Maximum image-plane intercept radius: {result.max_image_radius_mm:.9f} mm")


if __name__ == "__main__":
    main()
