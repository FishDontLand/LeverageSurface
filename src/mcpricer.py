import numpy as np

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


def calibrate_params(obs_price, tenors, strikes, n_paths, n_steps, r_d, r_f, s_0,
                     lr=1e-3, beta1=0.9, beta2=0.999, weight_decay=0.1, n_iters=1000,
                     verbose=False):
    STEP_SIZE = 1e-2
    def trans_params(param_trans):
        rho_max1_trans, rho_max2_trans, rho_max3_trans, rho_max5_trans = param_trans[:4]
        xi_max1_trans, xi_max2_trans, xi_max3_trans, xi_max5_trans = param_trans[4:8]
        lamb1_trans, lamb2_trans, lamb3_trans, lamb5_trans = param_trans[8:12]
        kappa = np.exp(param_trans[-1])

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

    current_all_params = np.zeros(13)

    (current_rho_max,
     current_xi_max,
     current_lamb,
     current_kappa) = trans_params(current_all_params)

    model = MCStochasticVolModel(max(tenors), n_paths, n_steps, r_d, r_f, 1, s_0, current_kappa,
                                 PiecewiseTermStructure([1, 2, 3, 5], current_rho_max),
                                 PiecewiseTermStructure([1, 2, 3, 5], current_xi_max),
                                 PiecewiseTermStructure([1, 2, 3, 5], current_lamb))

    m = np.zeros(13)
    v = np.zeros(13)

    for t in range(n_iters):
        simulations, _ = model.simulate()
        pvs = compute_pvs(simulations, model.times, r_d, tenors, strikes)

        if verbose:
            print('Iteration %d' % t)
            print('loss: ', np.sqrt(((pvs - obs_price) ** 2).mean()))

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
        adj_lr = lr * np.sqrt(1 - beta2 ** t)
        current_all_params = current_all_params - adj_lr * m / (np.sqrt(v) + 1e-6)
        current_all_params = current_all_params - lr * weight_decay * current_all_params

    return current_all_params












