import timeit

import numpy as np
import bisect
from scipy.optimize import least_squares
from scipy.stats import norm

class DupireGrid:
    def __init__(self, T_max, y_min, y_max, n_tau, n_y, s_0):
        self.T_max = T_max
        self.y_min = y_min
        self.y_max = y_max
        self.n_tau = n_tau
        self.n_y = n_y
        self.s_0 = s_0

        self.dt = (self.T_max / self.n_tau)
        self.dy = (self.y_max - self.y_min) / self.n_y
        self.tau = np.arange(self.n_tau + 1) * self.dt
        self.y = y_min +  np.arange(self.n_y + 1) * self.dy
        self.K = self.s_0 * np.exp(self.y)


class LocalVariance:
    def __init__(self, grid, var):
        assert var.shape == (grid.n_tau + 1, grid.n_y + 1), "local variance values should match the size of the underlying grid"
        assert (var > 0).all()
        self.grid = grid
        self.var = var


def iv_to_price(spot: float, r_f:float, r_d:float, strikes: np.array, ivs: np.array, expiries:np.array) -> np.array:
    implied_total_vol = ivs * np.sqrt(expiries)[:, None]
    d1 = (np.log(spot / strikes) + (r_d - r_f + 0.5 * ivs * ivs) * expiries[:, None]) / implied_total_vol
    d2 = d1 - implied_total_vol
    return norm.cdf(d1) * spot * np.exp(-r_f * expiries)[:, None] - norm.cdf(d2) * strikes * np.exp(-r_d * expiries)[:, None]

def est_first_order_derivative(xs, ys):
    derivative = np.zeros_like(ys)

    step_size = xs[1:] - xs[:-1]
    current_step_size = step_size[1:]
    prev_step_size = step_size[:-1]

    coeff1 = prev_step_size / (prev_step_size + current_step_size) / current_step_size
    coeff2 = (current_step_size - prev_step_size) / prev_step_size / current_step_size
    coeff3 = current_step_size  / (prev_step_size + current_step_size) / prev_step_size

    derivative[:, 1:ys.shape[1] - 1] = coeff1 * ys[:, 2:] + coeff2 * ys[:, 1:-1] + coeff3 * ys[:, :-2]
    derivative[:, 0] = (ys[:,1] - ys[:, 0]) / step_size[0]
    derivative[:, -1] = (ys[:, -2] - ys[:, -1]) / -(step_size[-1])

    return derivative

def est_second_order_derivative(xs, ys):
    second_order_derivative = np.zeros_like(ys)
    step_size = xs[1:] - xs[:-1]
    current_step_size = step_size[1:]
    prev_step_size = step_size[:-1]

    coeff1 = 2 / current_step_size  / (current_step_size + prev_step_size)
    coeff2 = -2 / current_step_size / prev_step_size
    coeff3 = 2 / prev_step_size / (prev_step_size + current_step_size)

    second_order_derivative[:, 1:ys.shape[1] - 1] = coeff1 * ys[:, 2:] + coeff2 * ys[:, 1:-1] + coeff3 * ys[:, :-2]

    # for boundary point, approximate the ys with a 4th order Lagrange polynomial and calculate
    # its second order polynomial
    coeff1 = (6 * xs[0] - 2 * xs[1] - 2 * xs[2] - 2 * xs[3]) / (xs[0] - xs[1]) / (xs[0] - xs[2]) / (xs[0] - xs[3])
    coeff2 = (4 * xs[0] - 2 * xs[2] - 2 * xs[3]) / (xs[1] - xs[0]) / (xs[1] - xs[2]) / (xs[1] - xs[3])
    coeff3 = (4 * xs[0] - 2 * xs[1] - 2 * xs[3]) / (xs[2] - xs[0]) / (xs[2] - xs[1]) / (xs[2] - xs[3])
    coeff4 = (4 * xs[0] - 2 * xs[1] - 2 * xs[2]) / (xs[3] - xs[0]) / (xs[3] - xs[1]) / (xs[3] - xs[2])

    second_order_derivative[:, 0] = coeff1 * ys[:, 0] + coeff2 * ys[:, 1] + coeff3 * ys[:, 2] + coeff4 * ys[:, 3]

    coeff1 = (6 * xs[-1] - 2 * xs[-2] - 2 * xs[-3] - 2 * xs[-4]) / (xs[-1] - xs[-2]) / (xs[-1] - xs[-3]) / (xs[-1] - xs[-4])
    coeff2 = (4 * xs[-1] - 2 * xs[-3] - 2 * xs[-4]) / (xs[-2] - xs[-1]) / (xs[-2] - xs[-3]) / (xs[-2] - xs[-4])
    coeff3 = (4 * xs[-1] - 2 * xs[-2] - 2 * xs[-4]) / (xs[-3] - xs[-1]) / (xs[-3] - xs[-2]) / (xs[-3] - xs[-4])
    coeff4 = (4 * xs[-1] - 2 * xs[-2] - 2 * xs[-3]) / (xs[-4] - xs[-1]) / (xs[-4] - xs[-2]) / (xs[-4] - xs[-3])

    second_order_derivative[:, -1] = coeff1 * ys[:, -1] + coeff2 * ys[:, -2] + coeff3 * ys[:, -3] + coeff4 * ys[:, -4]
    return second_order_derivative

def naive_price_to_local_vol(r_d: float, r_f: float, price_grid: np.array, expiries: np.array, strikes: np.array) -> np.array:
    strike_derivative = est_first_order_derivative(strikes, price_grid)
    time_derivative = est_first_order_derivative(expiries, price_grid.T).T
    strike_curvature = est_second_order_derivative(strikes, price_grid)
    # dupire equation
    loc_var = (time_derivative + (r_d - r_f) * strikes * strike_derivative + r_f * price_grid) / (strikes * strikes * strike_curvature) * 2
    loc_var = np.maximum(loc_var, 1e-8)
    return np.sqrt(loc_var)


def get_step_forward_coeffs(current_u, current_a, next_a, b, delta_tau, delta_y, nextL, nextU):
    c1 = -1 / delta_tau
    c2 = (current_a + next_a) / 4 / delta_y / delta_y
    c2 = c2[1: -1]
    c3 = (current_a + next_a - 2 * b) / 4 / delta_y
    c3 = c3[1: - 1]
    c1 = np.ones_like(c2) * c1

    coeff_j = c1 - 2 * c2
    coeff_j_plus_1 = c2 - c3
    coeff_j_minus_1 = c2 + c3

    rhs = c1 * current_u[1:-1] - c2 * (current_u[2:] - 2 * current_u[1:-1] + current_u[:-2]) + c3 * (current_u[2:] - current_u[:-2])

    rhs[0] -= (c2 + c3)[0] * nextL
    rhs[-1] -= (c2 - c3)[-1] * nextU
    coeff_j_minus_1[0] = 0
    coeff_j_plus_1[-1] = 0

    return coeff_j_minus_1, coeff_j, coeff_j_plus_1, rhs

def solve_tridiag(prev_coeffs, current_coeffs, next_coeffs, rhs_arr):
    epsilon = 1e-10
    assert min(np.abs(current_coeffs)) > epsilon, "matrix singuarity found"
    adj_next_coeffs = []
    adj_rhs_arr = []

    for i in range(len(current_coeffs)):
        if i == 0:
            adj_next_coeffs.append(next_coeffs[0] / current_coeffs[0])
            adj_rhs_arr.append(rhs_arr[0] / current_coeffs[0])
        else:
            scalar = current_coeffs[i] - prev_coeffs[i] * adj_next_coeffs[-1]
            if abs(scalar) < epsilon:
                raise ValueError(f"singularity at row {i}")
            adj_next_coeffs.append(next_coeffs[i] / scalar)
            adj_rhs_arr.append((rhs_arr[i] - prev_coeffs[i] * adj_rhs_arr[-1]) / scalar)

    solution = np.zeros(len(current_coeffs))
    for i in range(len(current_coeffs) - 1, -1, -1):
        if i == len(current_coeffs) - 1:
            solution[i] = adj_rhs_arr[i]
        else:
            solution[i] = adj_rhs_arr[i] - adj_next_coeffs[i] * solution[i + 1]

    return solution

def step_forward(current_u, current_a, next_a, delta_tau, delta_y, nextL, nextU, b):
    coeff_lower, coeff_middle, coeff_upper, rhs = get_step_forward_coeffs(current_u, current_a, next_a, b,
                                                                          delta_tau, delta_y, nextL, nextU)

    next_u = np.zeros_like(current_u)
    next_u[0] = nextL
    next_u[-1] = nextU
    next_u[1:-1] = solve_tridiag(coeff_lower, coeff_middle, coeff_upper, rhs)

    return next_u

# crank-nelson to solve
def forward_solve(var_grid, tau_arr, delta_tau, delta_y, b, u_left, u_right, u_init):
    a_grid = var_grid / 2
    u_grid = np.zeros_like(var_grid)
    u_grid[:, 0] = u_left
    u_grid[:, -1] = u_right
    for i in range(len(tau_arr)):
        if i == 0:
            current_u = u_init.copy()
            current_u[0] = u_left[0]
            current_u[-1] = u_right[0]
        else:
            current_u = step_forward(current_u, a_grid[i - 1,:], a_grid[i, :], delta_tau, delta_y, u_left[i],
                                     u_right[i], b)
        u_grid[i, :] = current_u

    return u_grid

def _calc_est_price(u_grid, grid_tau, grid_y, s_0, obs_tenors, obs_strikes):
    obs_ys = np.log(obs_strikes / s_0)
    all_est_price = []
    for t in obs_tenors:
        insert_idx = bisect.bisect_left(grid_tau, t)
        if insert_idx == 0:
            est_price = np.interp(obs_ys, grid_y, u_grid[0, :])
        elif insert_idx == len(grid_tau):
            est_price = np.interp(obs_ys, grid_y, u_grid[-1, :])
        else:
            est_price1 = np.interp(obs_ys, grid_y, u_grid[insert_idx - 1, :])
            est_price2 = np.interp(obs_ys, grid_y, u_grid[insert_idx, :])
            est_price = (t - grid_tau[insert_idx]) / (grid_tau[insert_idx - 1] - grid_tau[insert_idx]) * est_price1 + \
                        (t - grid_tau[insert_idx - 1]) / (grid_tau[insert_idx] - grid_tau[insert_idx - 1]) * est_price2
        all_est_price.append(est_price)
    return np.concatenate(all_est_price)

def expand_partial_grid(partial_ys, partial_ts, partial_xs, full_ts, full_xs):
    full_strike_ys = []
    for i in range(len(partial_ts)):
        full_strike_ys.append(np.interp(full_xs, partial_xs, partial_ys[i]))

    j = 0
    full_ys = []
    for i in range(len(full_ts)):
        tenor = full_ts[i]
        while j < len(partial_ts) - 1 and partial_ts[j] < tenor:
            j += 1
        full_ys.append(full_strike_ys[j])

    return np.stack(full_ys, axis=0)

def calibrate_local_vol(obs_price, obs_tenors, obs_strikes, s_0, r_d, r_f, y_min, y_max, tau_max, n_tau, n_y,
                        benchmarking=False, reg=0.01):
    obs_price_flatten = obs_price.flatten()
    dupire_grid = DupireGrid(tau_max, y_min, y_max, n_tau, n_y, s_0)
    left_boundary = s_0 * np.exp(-r_f * dupire_grid.tau)
    right_boundary = np.zeros(len(dupire_grid.tau))
    price_at_expiry = s_0 * np.maximum(1.0 - np.exp(dupire_grid.y), 0.0)

    init_local_vol = naive_price_to_local_vol(r_d, r_f, obs_price, obs_tenors, obs_strikes)
    init_local_vol = np.maximum(init_local_vol, 1e-8).flatten()

    def residual(x):
        local_vol_surface = x.reshape(len(obs_tenors), len(obs_strikes))
        var_surface = expand_partial_grid(local_vol_surface * local_vol_surface, obs_tenors, obs_strikes, dupire_grid.tau, s_0 * np.exp(dupire_grid.y))
        full_price_grid = forward_solve(var_surface, dupire_grid.tau, dupire_grid.dt, dupire_grid.dy, r_d - r_f,
                                        u_left=left_boundary, u_right=right_boundary, u_init=price_at_expiry)
        est_price = _calc_est_price(full_price_grid, dupire_grid.tau, dupire_grid.y, s_0, obs_tenors, obs_strikes)
        return np.concatenate([est_price - obs_price_flatten, reg * (x - init_local_vol)])

    if benchmarking:
        start = timeit.default_timer()
    res = least_squares(
        fun=residual,
        x0=init_local_vol,
        method='lm',
        max_nfev=80
    )
    if benchmarking:
        end = timeit.default_timer()
        print("calibration time: ", end - start)

    return np.abs(res.x.reshape(len(obs_tenors), len(obs_strikes)))

if __name__ == '__main__':
    # used for local test
    expand_partial_grid(
        partial_ys=np.random.randn(4, 3),
        partial_ts=np.array([1,2,3,4]),
        partial_xs=np.array([0.5, 1, 1.5]),
        full_ts = np.arange(6),
        full_xs = np.arange(5) * 0.5
    )











