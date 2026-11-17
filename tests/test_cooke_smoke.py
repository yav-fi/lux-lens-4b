from importlib.metadata import version

import pytest

from lux_frontier.cooke_smoke import trace_cooke_triplet


def test_real_cooke_triplet_reaches_image_plane():
    assert version("optiland") == "0.6.3"
    result = trace_cooke_triplet()
    assert result.wavelength_um == pytest.approx(0.55)
    assert result.effective_focal_length_mm == pytest.approx(50, abs=0.05)
    assert result.rays_reached_image == result.rays_traced == 9
    # A nonzero, small aberrated footprint catches missing refraction/propagation.
    assert 1e-6 < result.max_image_radius_mm < 0.1
