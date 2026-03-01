import numpy as np

from src.localvol import calibrate_local_vol, iv_to_price
import bisect

def _simulate_prices_from_calibrated_local_vol(
    local_vols,
    obs_tenors,
    obs_strikes,
    s0,
    r_d,
    r_f,
    n_paths=10000,
    n_steps=100,
    seed=123,
):
    np.random.seed(seed)
    max_t = float(obs_tenors[-1])
    dt = max_t / n_steps
    all_t = np.linspace(0, max_t, n_steps)
    sqrt_dt = np.sqrt(dt)
    drift = r_d - r_f

    dws = np.random.normal(0, sqrt_dt, (n_paths, n_steps))
    s = np.empty((n_paths, n_steps + 1))
    s[:, 0] = s0
    j = 0

    mc_prices = []

    for i in range(n_steps):
        t = dt * (i + 1)
        while j < len(obs_tenors) - 1 and obs_tenors[j] < t:
            j += 1
        local_vol_at_t = np.interp(s[:, i], obs_strikes, local_vols[j])
        s[:, i + 1] = s[:, i] + drift * s[:, i] * dt + dws[:, i] * local_vol_at_t * s[:, i]

    insert_idxs = np.searchsorted(all_t, obs_tenors)
    for idx, i in enumerate(insert_idxs):
        if i == 0:
            raise ValueError("observed tenors should be within the simulation time range")
        elif i == len(all_t):
            raise ValueError("observed tenors should be within the simulation time range")
        else:
            left = s[:, i - 1]
            right = s[:, i]
            left_t = all_t[i - 1]
            right_t = all_t[i]
            t = obs_tenors[idx]
            pvs = []
            for k in obs_strikes:
                left_payoff = np.maximum(left - k, 0.0).mean()
                right_payoff = np.maximum(right - k, 0.0).mean()
                right_w = (t - left_t) / (right_t - left_t)
                left_w = 1 - right_w
                expect_payoff = left_w * left_payoff + right_w * right_payoff
                pvs.append(expect_payoff * np.exp(-r_d * t))
            mc_prices.append(pvs)

    return np.array(mc_prices)




def test_calibrate_local_vol_mc_reprices_input_fx_grid():
    # FX context: EURUSD-like spot and strike/tenor ladder.
    s0 = 1.10
    r_d = 0.035
    r_f = 0.015

    obs_tenors = np.array([1, 2, 3])
    obs_strikes = np.array([0.85, 1.00, 1.25, 1.30, 1.35, 1.40, 1.45])

    # Input market prices only (derived from an FX-style implied-vol smile per tenor).
    # Calibration should infer local variance from these prices.
    obs_ivs = np.array(
        [
            [0.125, 0.118, 0.112, 0.110, 0.112, 0.118, 0.130],
            [0.130, 0.122, 0.116, 0.114, 0.116, 0.122, 0.136],
            [0.138, 0.130, 0.124, 0.122, 0.124, 0.130, 0.146],
        ]
    )
    observed_prices = iv_to_price(s0, r_f, r_d, obs_strikes, obs_ivs, obs_tenors)

    # Calibration config.
    y_min = np.log(0.50)
    y_max = np.log(1.80)
    tau_max = np.max(obs_tenors)
    n_tau = 300
    n_y = 140

    calibrated_local_vol = calibrate_local_vol(
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

    mc_prices = _simulate_prices_from_calibrated_local_vol(
        calibrated_local_vol,
        obs_tenors,
        obs_strikes,
        s0,
        r_d,
        r_f,
        n_steps=n_tau,
    )

    # Repricing should match the original input quotes reasonably well.
    max_abs_error = np.max(np.abs(mc_prices - observed_prices))
    mean_abs_error = np.mean(np.abs(mc_prices - observed_prices))

    assert mean_abs_error < 0.1
    assert max_abs_error < 0.1