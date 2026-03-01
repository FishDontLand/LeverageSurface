import numpy as np

from src.mcpricer import (
    MCStochasticVolModel,
    PiecewiseTermStructure,
    calibrate_params,
    compute_pvs,
    trans_params,
)
from src.localvol import iv_to_price


def _pricing_model(max_t, n_paths, n_steps, r_d, r_f, s0):
    base_params = np.zeros(13)
    rho_max, xi_max, lamb, kappa = trans_params(base_params)
    return MCStochasticVolModel(
        t_max=max_t,
        n_paths=n_paths,
        n_steps=n_steps,
        r_d=r_d,
        r_f=r_f,
        v0=1.0,
        s0=s0,
        kappa=kappa,
        max_rho_ts=PiecewiseTermStructure([1, 2, 3, 5], rho_max),
        max_xi_ts=PiecewiseTermStructure([1, 2, 3, 5], xi_max),
        lamb_ts=PiecewiseTermStructure([1, 2, 3, 5], lamb),
    )


def test_calibrate_params_reduces_pricing_error_on_synthetic_data():
    tenors = np.array([1, 2, 3])
    strikes = np.array([0.85, 1.00, 1.25, 1.30, 1.35, 1.40, 1.45])

    obs_ivs = np.array(
        [
            [0.125, 0.118, 0.112, 0.110, 0.112, 0.118, 0.130],
            [0.130, 0.122, 0.116, 0.114, 0.116, 0.122, 0.136],
            [0.138, 0.130, 0.124, 0.122, 0.124, 0.130, 0.146],
        ]
    )
    r_d = 0.03
    r_f = 0.01
    s0 = 1.0
    n_steps = 300
    n_paths = 10000

    observed_prices = iv_to_price(s0, r_f, r_d, strikes, obs_ivs, tenors)

    estimated_params, calibrated_model = calibrate_params(
        obs_price=observed_prices,
        tenors=tenors,
        strikes=strikes,
        n_paths=n_paths,
        n_steps=n_steps,
        r_d=r_d,
        r_f=r_f,
        s_0=s0,
        n_iters=60,
        verbose=True,
    )

    base_model = _pricing_model(
        max_t=float(tenors.max()),
        n_paths=n_paths,
        n_steps=n_steps,
        r_d=r_d,
        r_f=r_f,
        s0=s0,
    )
    base_paths, _ = base_model.simulate()
    base_prices = compute_pvs(base_paths, base_model.times, r_d, tenors, strikes)

    est_paths, _ = calibrated_model.simulate()
    estimated_prices = compute_pvs(est_paths, calibrated_model.times, r_d, tenors, strikes)

    calibrated_rmse = np.sqrt(np.mean((estimated_prices - observed_prices) ** 2))
    base_rmse = np.sqrt(np.mean((base_prices - observed_prices) ** 2))
    assert calibrated_rmse < base_rmse