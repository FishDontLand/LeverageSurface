import numpy as np
from src.mcpricer import (
    MCStochasticVolModel,
    PiecewiseTermStructure,
    compute_pvs,
    trans_params, calibrate_leverage_surface, simulate_with_leverage_surface, compute_barrier_pvs,
)
from src.localvol import iv_to_price, calibrate_local_vol


def _pricing_model(max_t, n_paths, n_steps, r_d, r_f, s0, v0):
    base_params = np.zeros(13)
    rho_max, xi_max, lamb, kappa = trans_params(base_params)
    return MCStochasticVolModel(
        t_max=max_t,
        n_paths=n_paths,
        n_steps=n_steps,
        r_d=r_d,
        r_f=r_f,
        v0=v0,
        s0=s0,
        kappa=kappa,
        max_rho_ts=PiecewiseTermStructure([1, 2, 3, 5], rho_max),
        max_xi_ts=PiecewiseTermStructure([1, 2, 3, 5], xi_max),
        lamb_ts=PiecewiseTermStructure([1, 2, 3, 5], lamb),
    )


def test_compute_pvs():
    r_d = 0.03
    r_f = 0.01
    s0 = 1.0
    sigma = 0.2
    tenors = np.array([1.0, 2.0])
    strikes = np.array([0.8, 1.0, 1.2])

    n_paths = 250_000
    rng = np.random.default_rng(123)

    sim_times = np.array([0.0, 1.0, 2.0])
    dt = np.diff(sim_times)
    brownian_increments = rng.standard_normal((n_paths, len(dt))) * np.sqrt(dt)

    s_sim = np.empty((n_paths, len(sim_times)))
    s_sim[:, 0] = s0

    for i, dt_i in enumerate(dt):
        drift = (r_d - r_f - 0.5 * sigma * sigma) * dt_i
        diffusion = sigma * brownian_increments[:, i]
        s_sim[:, i + 1] = s_sim[:, i] * np.exp(drift + diffusion)

    mc_prices = compute_pvs(s_sim, sim_times, r_d, tenors, strikes)
    bs_prices = iv_to_price(
        spot=s0,
        r_f=r_f,
        r_d=r_d,
        strikes=strikes,
        ivs=np.full((len(tenors), len(strikes)), sigma),
        expiries=tenors,
    )

    np.testing.assert_allclose(mc_prices, bs_prices, atol=1.5e-3, rtol=0.0)


def test_model_simulate1():
    r_d = 0.03
    r_f = 0.01
    s0 = 1.0
    tenors = np.array([1.0, 2.0])
    strikes = np.array([0.8, 1.0, 1.2])

    model = MCStochasticVolModel(
        t_max=2.0,
        n_paths=250_000,
        n_steps=200,
        r_d=r_d,
        r_f=r_f,
        v0=1.0,
        s0=s0,
        kappa=1.0,
        max_rho_ts=PiecewiseTermStructure([1, 2, 3, 5], np.array([-0.3, -0.3, -0.3, -0.3])),
        max_xi_ts=PiecewiseTermStructure([1, 2, 3, 5], np.array([0.7, 0.7, 0.7, 0.7])),
        lamb_ts=PiecewiseTermStructure([1, 2, 3, 5], np.zeros(4)),
        seed=123,
    )

    s_sim, _ = model.simulate()
    mc_prices = compute_pvs(s_sim, model.times, r_d, tenors, strikes)

    bs_prices = iv_to_price(
        spot=s0,
        r_f=r_f,
        r_d=r_d,
        strikes=strikes,
        ivs=np.full((len(tenors), len(strikes)), 1.0),
        expiries=tenors,
    )

    np.testing.assert_allclose(mc_prices, bs_prices, atol=3e-3, rtol=0.0)


def test_simulation_with_leverage():
    r_d = 0.03
    r_f = 0.01
    s0 = 1.0
    leverage = 0.6
    tenors = np.array([1.0, 2.0])
    strikes = np.array([0.8, 1.0, 1.2])

    model = MCStochasticVolModel(
        t_max=2.0,
        n_paths=250_000,
        n_steps=200,
        r_d=r_d,
        r_f=r_f,
        v0=1.0,
        s0=s0,
        kappa=1.0,
        max_rho_ts=PiecewiseTermStructure([1, 2, 3, 5], np.array([-0.3, -0.3, -0.3, -0.3])),
        max_xi_ts=PiecewiseTermStructure([1, 2, 3, 5], np.array([0.7, 0.7, 0.7, 0.7])),
        lamb_ts=PiecewiseTermStructure([1, 2, 3, 5], np.zeros(4)),
        seed=123,
    )

    leverage_surface = np.full((len(tenors), len(strikes)), leverage)
    s_sim, _ = simulate_with_leverage_surface(leverage_surface, tenors, strikes, model)
    mc_prices = compute_pvs(s_sim, model.times, r_d, tenors, strikes)

    bs_prices = iv_to_price(
        spot=s0,
        r_f=r_f,
        r_d=r_d,
        strikes=strikes,
        ivs=np.full((len(tenors), len(strikes)), leverage),
        expiries=tenors,
    )

    np.testing.assert_allclose(mc_prices, bs_prices, atol=2e-3, rtol=0.0)

def test_calibrate_leverage_surface():
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
    v0 = 1.1

    obs_prices = iv_to_price(s0, r_f, r_d, strikes, obs_ivs, tenors)

    kappa = 1.1286698125954764
    lambs = np.array([0.04566, 0.04934, 0.96427, 0.50000])
    max_rhos = np.array([-0.31832, -0.21295, -0.54724, -0.27500])
    max_xis = np.array([0.15903, 0.15190, 0.25732, 0.25000])

    leverage_surface, est_pvs = calibrate_leverage_surface(obs_prices, tenors, strikes, s0, v0, r_d, r_f,
                                                           kappa, max_xis, max_rhos, lambs,
                                                           300, 300, 30000, 300)

    assert np.sqrt(((est_pvs - obs_prices) ** 2).mean()) < 0.02


def test_compute_barrier_pvs1():
    r_d = 0.03
    r_f = 0.01
    s0 = 1.0
    sigma = 0.25
    tenors = np.array([1.0, 2.0])
    strikes = np.array([0.8, 1.0, 1.2])

    n_paths = 150_000
    n_steps = 200
    t_max = 2.0
    dt = t_max / n_steps
    sim_times = np.linspace(0.0, t_max, n_steps + 1)

    rng = np.random.default_rng(7)
    z = rng.standard_normal((n_paths, n_steps))

    s_sim = np.empty((n_paths, n_steps + 1))
    s_sim[:, 0] = s0
    drift = (r_d - r_f - 0.5 * sigma * sigma) * dt
    for i in range(n_steps):
        s_sim[:, i + 1] = s_sim[:, i] * np.exp(drift + sigma * np.sqrt(dt) * z[:, i])

    vanilla_pvs = compute_pvs(s_sim, sim_times, r_d, tenors, strikes)
    barrier_pvs = compute_barrier_pvs(s_sim, sim_times, r_d, tenors, strikes, B=100)

    np.testing.assert_allclose(barrier_pvs, vanilla_pvs, atol=1e-12, rtol=0.0)


def test_compute_barrier_pvs2():
    r_d = 0.03
    r_f = 0.01
    s0 = 1.0
    sigma = 0.35
    tenors = np.array([1.0, 2.0, 3, 5])
    strikes = np.array([0.8, 1.0, 1.2])

    n_paths = 150_000
    n_steps = 200
    t_max = 5.0
    dt = t_max / n_steps
    sim_times = np.linspace(0.0, t_max, n_steps + 1)

    rng = np.random.default_rng(9)
    z = rng.standard_normal((n_paths, n_steps))

    s_sim = np.empty((n_paths, n_steps + 1))
    s_sim[:, 0] = s0
    drift = (r_d - r_f - 0.5 * sigma * sigma) * dt
    for i in range(n_steps):
        s_sim[:, i + 1] = s_sim[:, i] * np.exp(drift + sigma * np.sqrt(dt) * z[:, i])

    high_barrier = compute_barrier_pvs(s_sim, sim_times, r_d, tenors, strikes, B=1.50)
    mid_barrier = compute_barrier_pvs(s_sim, sim_times, r_d, tenors, strikes, B=1.25)
    low_barrier = compute_barrier_pvs(s_sim, sim_times, r_d, tenors, strikes, B=1.10)

    assert np.all(high_barrier >= mid_barrier - 1e-12)
    assert np.all(mid_barrier >= low_barrier - 1e-12)
    assert np.any(high_barrier > low_barrier + 1e-4)