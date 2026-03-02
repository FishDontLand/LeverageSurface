from deprecated import deprecated

import numpy as np

from src.localvol import calibrate_local_vol, iv_to_price


class PiecewiseTermStructure:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def value_at_times(self, t: np.array):
        insert_idx = np.searchsorted(self.xs, t)
        insert_idx = np.clip(insert_idx, 0, len(self.xs) - 1)
        return self.ys[insert_idx]

    def value_at(self, t: float):
        insert_idx = np.searchsorted(self.xs, t)
        insert_idx = min(insert_idx, len(self.xs) - 1)
        return self.ys[insert_idx]

class MCStochasticVolModel:
    def __init__(self, t_max: float, n_paths: int, n_steps: int, r_d: float,
                 r_f: float, v0: float, s0: float, kappa: float, max_rho_ts: PiecewiseTermStructure,
                 max_xi_ts: PiecewiseTermStructure, lamb_ts: PiecewiseTermStructure,
                 seed=210, v_floor=1e-8):
        self.t_max = t_max
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = self.t_max / self.n_steps
        self.times = np.linspace(0, self.t_max, self.n_steps + 1)

        self.r_d = r_d
        self.r_f = r_f
        self.mu = self.r_d - self.r_f

        self.v0 = v0
        self.s0 = s0
        self.kappa = kappa
        self.max_rho_ts = max_rho_ts
        self.max_xi_ts = max_xi_ts
        self.lamb_ts = lamb_ts

        self.seed=seed
        self.v_floor = v_floor

        rng = np.random.default_rng(self.seed)
        self.sim_cache = rng.standard_normal([n_paths, n_steps, 2])

    def set_param(self, max_rhos, max_xis, lambs, kappa):
        self.kappa = kappa
        self.max_rho_ts.ys = max_rhos
        self.max_xi_ts.ys = max_xis
        self.lamb_ts.ys = lambs

    def simulate(self):
        v_sim = np.empty([self.n_paths, self.n_steps + 1])
        s_sim = np.empty([self.n_paths, self.n_steps + 1])
        v_sim[:, 0] = self.v0
        s_sim[:, 0] = self.s0
        sqrt_dt = np.sqrt(self.dt)

        for i in range(self.n_steps):
            t = i * self.dt
            xi = self.max_xi_ts.value_at(t) * self.lamb_ts.value_at(t)
            dw2 = self.sim_cache[:, i, 1] * sqrt_dt
            rho = self.max_rho_ts.value_at(t) * self.lamb_ts.value_at(t)
            dw1 =  self.sim_cache[:, i, 0] * sqrt_dt * np.sqrt(1 - rho * rho) + rho * dw2

            v_sim[:, i + 1] = v_sim[:, i] + self.kappa * (1.0 - v_sim[:, i]) * self.dt + \
                xi * v_sim[:, i] * dw2 + 0.5 * (xi * xi) * v_sim[:, i] * (dw2  * dw2- self.dt)

            v_sim[:, i + 1] = np.maximum(v_sim[:, i + 1], self.v_floor)
            s_sim[:, i + 1] = s_sim[:, i] * np.exp((self.mu - 0.5 * v_sim[:, i] * v_sim[:, i]) * self.dt + v_sim[:, i] * dw1)

        return s_sim, v_sim

def compute_barrier_pvs(s_sim: np.array, sim_times: np.array, r_d: float, tenors: np.array, strikes: np.array, B):
    insert_idx = np.searchsorted(sim_times, tenors)

    if min(insert_idx) == 0 or max(insert_idx) == len(sim_times):
        raise ValueError("tenors out of simulation range")

    assert np.abs(sim_times[insert_idx] - tenors).max() < 1e-8, "tenors not on simulation grid"

    all_pvs = []
    for i, idx in enumerate(insert_idx):
        max_s = s_sim[:, idx].max(axis=1)
        s_sim_adj = np.where(max_s < B, s_sim[:, idx], -1.0)
        payoff = np.maximum(s_sim_adj - strikes, 0.0)
        all_pvs.append(payoff.mean(axis=0) * np.exp(-r_d * tenors[i]))

    return np.stack(all_pvs, axis=0)

def compute_pvs(s_sim: np.array, sim_times: np.array, r_d: float, tenors: np.array, strikes: np.array):
    insert_idx = np.searchsorted(sim_times, tenors)

    if min(insert_idx) == 0 or max(insert_idx) == len(sim_times):
        raise ValueError("tenors out of simulation range")

    assert np.abs(sim_times[insert_idx] - tenors).max() < 1e-8, "tenors not on simulation grid"

    df = np.exp(-r_d * tenors)
    forward_price = np.maximum(s_sim[:, insert_idx,  None] - strikes, 0).mean(axis=0)
    pvs = forward_price * df[:, None]
    return pvs

def compute_grad(s_sim: np.array, stressed_s_sim: np.array, step_size: float,
                 sim_times: np.array, r_d: float, tenors: np.array, strikes: np.array):
    insert_idx = np.searchsorted(sim_times, tenors)

    if min(insert_idx) == 0 or max(insert_idx) == len(sim_times):
        raise ValueError("tenors out of simulation range")

    assert np.abs(sim_times[insert_idx] - tenors).max() < 1e-8, "tenors not on simulation grid"

    df = np.exp(-r_d * tenors)
    reg_payoffs = np.maximum(s_sim[:, insert_idx, None] - strikes, 0)
    stressed_payoffs = np.maximum(stressed_s_sim[:, insert_idx, None] - strikes, 0)
    pathwise_derivative = (stressed_payoffs - reg_payoffs) / step_size
    expected_deriv = pathwise_derivative.mean(axis=0) * df[:, None]
    return expected_deriv

def trans_params(param_trans):
    rho_max1_trans, rho_max2_trans, rho_max3_trans, rho_max5_trans = param_trans[:4]
    xi_max1_trans, xi_max2_trans, xi_max3_trans, xi_max5_trans = param_trans[4:8]
    lamb1_trans, lamb2_trans, lamb3_trans, lamb5_trans = param_trans[8:12]
    kappa = np.tanh(param_trans[-1]) * 2 + 3

    rho_max1 = np.tanh(rho_max1_trans) * 0.2 - 0.5
    rho_max2 = np.tanh(rho_max2_trans) * 0.2 - 0.4
    rho_max3 = np.tanh(rho_max3_trans) * 0.2 - 0.35
    rho_max5 = np.tanh(rho_max5_trans) * 0.15 - 0.275

    xi_max1 = np.tanh(xi_max1_trans) * 0.3 + 0.45
    xi_max2 = np.tanh(xi_max2_trans) * 0.25 + 0.375
    xi_max3 = np.tanh(xi_max3_trans) * 0.25 + 0.325
    xi_max5 = np.tanh(xi_max5_trans) * 0.2 + 0.25

    lamb1 = np.exp(lamb1_trans) / (np.exp(lamb1_trans) + 1)
    lamb2 = np.exp(lamb2_trans) / (np.exp(lamb2_trans) + 1)
    lamb3 = np.exp(lamb3_trans) / (np.exp(lamb3_trans) + 1)
    lamb5 = np.exp(lamb5_trans) / (np.exp(lamb5_trans) + 1)

    return np.array([rho_max1, rho_max2, rho_max3, rho_max5]), np.array([xi_max1, xi_max2, xi_max3, xi_max5]), np.array([lamb1, lamb2, lamb3, lamb5]), kappa

@deprecated
def calibrate_non_leverage_params(obs_price, tenors, strikes, n_paths, n_steps, r_d, r_f, s_0, v0,
                                  lr=1e-1, beta1=0.8, beta2=0.9, weight_decay=0.1, n_iters=1000,
                                  verbose=False):
    STEP_SIZE = 1e-2
    current_all_params = np.zeros(13)

    (current_rho_max,
     current_xi_max,
     current_lamb,
     current_kappa) = trans_params(current_all_params)

    model = MCStochasticVolModel(max(tenors), n_paths, n_steps, r_d, r_f, v0, s_0, current_kappa,
                                 PiecewiseTermStructure([1, 2, 3, 5], current_rho_max),
                                 PiecewiseTermStructure([1, 2, 3, 5], current_xi_max),
                                 PiecewiseTermStructure([1, 2, 3, 5], current_lamb))

    m = np.zeros(13)
    v = np.zeros(13)

    for t_idx in range(n_iters):
        (current_rho_max,
         current_xi_max,
         current_lamb,
         current_kappa) = trans_params(current_all_params)
        model.set_param(current_rho_max, current_xi_max, current_lamb, current_kappa)

        simulations, _ = model.simulate()
        pvs = compute_pvs(simulations, model.times, r_d, tenors, strikes)

        if verbose:
            loss = np.sqrt(((pvs - obs_price) ** 2).mean())
            print('Iteration ', t_idx + 1)
            print('loss: ', loss)
        g = np.zeros(len(current_all_params))
        for i in range(len(current_all_params)):
            stressed_trans_params = current_all_params.copy()
            stressed_trans_params[i] += STEP_SIZE

            stressed_rho_max, stressed_xi_max, stressed_lamb, stressed_kappa = trans_params(stressed_trans_params)
            model.set_param(stressed_rho_max, stressed_xi_max, stressed_lamb, stressed_kappa)
            stressed_simulations, _ = model.simulate()
            pv_gradients = compute_grad(simulations, stressed_simulations, STEP_SIZE, model.times, r_d, tenors, strikes)
            g[i] = ((pvs - obs_price) * pv_gradients).sum()

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        t = t_idx + 1
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        current_all_params = current_all_params - lr * m_hat / (np.sqrt(v_hat) + 1e-6)
        current_all_params = current_all_params - lr * weight_decay * current_all_params

    final_rho_max, final_xi_max, final_lamb, final_kappa = trans_params(current_all_params)
    model.set_param(final_rho_max, final_xi_max, final_lamb, final_kappa)

    return current_all_params, model

def simulate_with_leverage_surface(leverage_surface, tenors, strikes, base_model):
    v_sim = np.empty([base_model.n_paths, base_model.n_steps + 1])
    s_sim = np.empty([base_model.n_paths, base_model.n_steps + 1])
    v_sim[:, 0] = base_model.v0
    s_sim[:, 0] = base_model.s0
    sqrt_dt = np.sqrt(base_model.dt)

    j = 0
    for i in range(base_model.n_steps):
        t = i * base_model.dt
        while j < len(tenors) - 1 and t > tenors[j]:
            j += 1
        xi = base_model.max_xi_ts.value_at(t) * base_model.lamb_ts.value_at(t)
        dw2 = base_model.sim_cache[:, i, 1] * sqrt_dt
        rho = base_model.max_rho_ts.value_at(t) * base_model.lamb_ts.value_at(t)
        dw1 = base_model.sim_cache[:, i, 0] * sqrt_dt * np.sqrt(1 - rho * rho) + rho * dw2

        v_sim[:, i + 1] = v_sim[:, i] + base_model.kappa * (1.0 - v_sim[:, i]) * base_model.dt + \
                          xi * v_sim[:, i] * dw2 + 0.5 * (xi * xi) * v_sim[:, i] * (dw2 * dw2 - base_model.dt)

        v_sim[:, i + 1] = np.maximum(v_sim[:, i + 1], base_model.v_floor)
        leverage_surface_slice = np.interp(s_sim[:, i], strikes, leverage_surface[j, :])
        s_sim[:, i + 1] = s_sim[:, i] * np.exp(
            (base_model.mu - 0.5 * leverage_surface_slice * leverage_surface_slice * v_sim[:, i] * v_sim[:, i]) * base_model.dt + leverage_surface_slice * v_sim[:, i] * dw1)

    return s_sim, v_sim

def calc_stoch_var_cond_exp(s_sim, v_sim, sim_times, tenors, strikes):
    insert_idx = np.searchsorted(sim_times, tenors)

    if min(insert_idx) == 0 or max(insert_idx) == len(sim_times):
        raise ValueError('Leverage Surface tenors out of simulation range')

    assert np.max(np.abs(sim_times[insert_idx] - tenors)) < 1e-8, "leverage surface tenor not on simulation time grid"

    s_sim_at_tenor = s_sim[:, insert_idx]
    v_sim_at_tenor = v_sim[:, insert_idx]

    sim_strikes_max = np.max(s_sim_at_tenor, axis=0)
    sim_strikes_min = np.min(s_sim_at_tenor, axis=0)
    num_segments = 100
    radius = (sim_strikes_max - sim_strikes_min) / num_segments / 2

    conditional_expectations = []
    for idx, s in enumerate(strikes):
        condition = (s_sim_at_tenor > s - radius) & (s_sim_at_tenor < s + radius)
        vt_square_avg = np.nanmean(np.where(condition, v_sim_at_tenor * v_sim_at_tenor, np.nan), axis=0)
        conditional_expectations.append(vt_square_avg)

    return np.stack(conditional_expectations, axis=1)

def calibrate_leverage_surface_from_base(obs_prices, local_vol, base_model, tenors, strikes, num_iters,
                                         verbose=False):
    leverage_surface = np.ones([len(tenors), len(strikes)])
    lamb=5e-1
    diffs = []
    i = 0
    s_sim, v_sim = simulate_with_leverage_surface(leverage_surface, tenors, strikes, base_model)
    cond_exp = calc_stoch_var_cond_exp(s_sim, v_sim, base_model.times, tenors, strikes)
    if np.isnan(cond_exp).any():
        raise ValueError('bad initial condition')
    pvs = compute_pvs(s_sim, base_model.times, base_model.r_d, tenors, strikes)
    loss = np.sqrt(((pvs - obs_prices) ** 2).mean())
    while i < num_iters:
        if lamb < 1e-5:
            break
        new_leverage_surface = (1 - lamb) * leverage_surface + lamb * local_vol / np.maximum(np.sqrt(cond_exp), 1e-6)
        s_sim, v_sim = simulate_with_leverage_surface(new_leverage_surface, tenors, strikes, base_model)
        cond_exp = calc_stoch_var_cond_exp(s_sim, v_sim, base_model.times, tenors, strikes)
        if np.isnan(cond_exp).any():
            lamb /= 2
            continue
        pvs = compute_pvs(s_sim, base_model.times, base_model.r_d, tenors, strikes)
        new_loss = np.sqrt(((pvs - obs_prices) ** 2).mean())

        if new_loss > loss:
            lamb /= 2
            continue
        loss = new_loss
        if verbose:
            print(f"Iteration {i + 1}, loss: {loss:.4}")
        diffs.append(np.sqrt(((new_leverage_surface - leverage_surface) ** 2).mean()))
        leverage_surface = new_leverage_surface
        i += 1

    return leverage_surface, diffs

def calibrate_leverage_surface(obs_prices, tenors, strikes, s0, v0, r_d, r_f, kappa, max_xis, max_rhos, lambs,
                               n_pde_tau, n_pde_strike, n_sim_paths, n_sim_steps):
    y_min = np.log(np.maximum(np.min(strikes) / s0 - 0.3, 1e-6))
    y_max = np.log(np.max(strikes) + 0.3)

    calibrated_local_vol = calibrate_local_vol(obs_prices, tenors, strikes, s0,  r_d, r_f,
                                               y_min, y_max, max(tenors), n_pde_tau, n_pde_strike)

    base_model = MCStochasticVolModel(max(tenors), n_sim_paths, n_sim_steps, r_d, r_f, v0, s0, kappa,
                                      PiecewiseTermStructure([1,2,3,5], max_rhos),
                                      PiecewiseTermStructure([1,2,3,5], max_xis),
                                      PiecewiseTermStructure([1,2,3,5], lambs))

    leverage_surface, _ = calibrate_leverage_surface_from_base(obs_prices, calibrated_local_vol,
                                                               base_model, tenors, strikes, num_iters=100)

    s_sim, _ = simulate_with_leverage_surface(leverage_surface, tenors, strikes, base_model)

    est_pvs = compute_pvs(s_sim, base_model.times, r_d, tenors, strikes)

    return leverage_surface, est_pvs

def calculate_barrier_pv(base_model, leverage_surface, tenors, strikes):
    s_sim, v_sim = simulate_with_leverage_surface(leverage_surface, tenors, strikes, base_model)


if __name__ == '__main__':
    # for quick local test
    pass












