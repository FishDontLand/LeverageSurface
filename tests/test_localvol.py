import numpy as np

from src.localvol import calibrate_local_vol, iv_to_price


def _simulate_prices_from_calibrated_local_var(
    local_var_obs,
    obs_tenors,
    obs_strikes,
    s0,
    r_d,
    r_f,
    *,
    n_paths=80000,
    steps_per_year=252,
    seed=123,
):
    """Monte Carlo repricer for a local-variance surface that is
    piecewise-constant in time (tenor buckets) and linear in strike via log-moneyness.
    """

    rng = np.random.default_rng(seed)
    max_t = float(obs_tenors[-1])
    n_steps = int(np.ceil(max_t * steps_per_year))
    dt = max_t / n_steps
    sqrt_dt = np.sqrt(dt)

    ys = np.log(obs_strikes / s0)

    # Each tenor is represented by a flat local-variance bucket on [prev_t, t].
    bucket_end_steps = np.clip(np.rint(obs_tenors / dt).astype(int), 1, n_steps)

    def local_var_for_state(tenor_bucket_idx, spot_state):
        y_state = np.log(spot_state / s0)
        return np.interp(
            y_state,
            ys,
            local_var_obs[tenor_bucket_idx],
            left=local_var_obs[tenor_bucket_idx, 0],
            right=local_var_obs[tenor_bucket_idx, -1],
        )

    spot = np.full(n_paths, s0, dtype=float)
    bucket_idx = 0
    saved_states = {}
    drift = r_d - r_f

    for step in range(1, n_steps + 1):
        var = local_var_for_state(bucket_idx, spot)
        vol = np.sqrt(np.maximum(var, 1e-12))
        z = rng.standard_normal(n_paths)
        spot *= np.exp((drift - 0.5 * vol * vol) * dt + vol * sqrt_dt * z)

        while bucket_idx < len(bucket_end_steps) and step == bucket_end_steps[bucket_idx]:
            saved_states[bucket_idx] = spot.copy()
            bucket_idx += 1

    mc_prices = np.zeros((len(obs_tenors), len(obs_strikes)))
    for i, t in enumerate(obs_tenors):
        st = saved_states[i]
        payoff = np.maximum(st[:, None] - obs_strikes[None, :], 0.0)
        mc_prices[i] = np.exp(-r_d * t) * np.mean(payoff, axis=0)

    return mc_prices


def test_calibrate_local_vol_mc_reprices_input_fx_grid():
    # FX context: EURUSD-like spot and strike/tenor ladder.
    s0 = 1.10
    r_d = 0.035
    r_f = 0.015

    obs_tenors = np.array([0.25, 0.5, 1.0])
    obs_strikes = np.array([0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30])

    # Input market prices only (derived from an FX-style implied-vol smile per tenor).
    # Calibration should infer local variance from these prices.
    obs_ivs = np.array(
        [
            [0.125, 0.118, 0.112, 0.110, 0.112, 0.118, 0.130],
            [0.130, 0.122, 0.116, 0.114, 0.116, 0.122, 0.136],
            [0.138, 0.130, 0.124, 0.122, 0.124, 0.130, 0.146],
        ]
    )
    observed_prices = iv_to_price(s0, r_d - r_f, obs_strikes, obs_ivs, obs_tenors)

    # Calibration config.
    y_min = np.log(0.60)
    y_max = np.log(1.70)
    tau_max = 1.0
    n_tau = 80
    n_y = 140

    calibrated_local_var = calibrate_local_vol(
        observed_prices,
        obs_tenors,
        obs_strikes,
        s0,
        r_d,
        r_f,
        y_min,
        y_max,
        tau_max,
        n_tau,
        n_y,
        benchmarking=True
    )

    mc_prices = _simulate_prices_from_calibrated_local_var(
        calibrated_local_var,
        obs_tenors,
        obs_strikes,
        s0,
        r_d,
        r_f,
    )

    # Repricing should match the original input quotes reasonably well.
    max_abs_error = np.max(np.abs(mc_prices - observed_prices))
    mean_abs_error = np.mean(np.abs(mc_prices - observed_prices))

    assert mean_abs_error < 0.008
    assert max_abs_error < 0.02