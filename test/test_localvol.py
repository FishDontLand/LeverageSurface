import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from localvol import expand_local_var, iv_to_price


def test_monte_carlo_price_matches_black_scholes_input():
    rng = np.random.default_rng(42)
    spot = 100.0
    strike = 100.0
    r = 0.01
    vol = 0.2
    expiry = 1.0

    n_paths = 200_000
    z = rng.standard_normal(n_paths)
    terminal_spot = spot * np.exp((r - 0.5 * vol * vol) * expiry + vol * np.sqrt(expiry) * z)
    mc_price = np.exp(-r * expiry) * np.maximum(terminal_spot - strike, 0.0).mean()

    bs_price = iv_to_price(
        spot=spot,
        r=r,
        strikes=np.array([strike]),
        ivs=np.array([vol]),
        expiries=np.array([expiry]),
    )[0, 0]

    assert abs(mc_price - bs_price) < 0.12


def test_expand_local_var_expands_in_time_and_interpolates_in_strike():
    partial_ts = np.array([0.5, 1.0])
    partial_xs = np.array([0.0, 1.0])
    partial_ys = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    full_ts = np.array([0.25, 0.5, 0.75, 1.0])
    full_xs = np.array([0.0, 0.5, 1.0])

    expanded = expand_local_var(partial_ys, partial_ts, partial_xs, full_ts, full_xs)

    expected = np.array([
        [1.0, 1.5, 2.0],
        [1.0, 1.5, 2.0],
        [3.0, 3.5, 4.0],
        [3.0, 3.5, 4.0],
    ])

    np.testing.assert_allclose(expanded, expected)
