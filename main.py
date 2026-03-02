import argparse
import numpy as np
import pandas as pd

from src.localvol import iv_to_price
from src.mcpricer import calibrate_leverage_surface

def list_of_float(arg):
    n_split = arg.split(",")
    return [eval(e) for e in n_split]

def do_calibration(iv_file, s0, v0, r_d, r_f, kappa, max_xis, max_rhos, lambdas,
                   n_time_steps, n_sim_paths, n_strike_steps, output_file=None):
    iv_data = pd.read_csv(iv_file)
    iv_data.columns = [c.lower() for c in iv_data.columns]
    tenors = np.sort(iv_data['term'].unique())
    strikes = np.sort(iv_data['strike'].unique())

    iv_data = iv_data.sort_values(by=['term', 'strike'])
    assert len(iv_data) == len(tenors) * len(strikes), "iv surface must be a grid"
    obs_vols = iv_data['iv'].values.reshape([len(tenors), len(strikes)])
    obs_prices = iv_to_price(s0, r_f, r_d, strikes, obs_vols, tenors)

    leverage_surface, est_pvs = calibrate_leverage_surface(obs_prices, tenors, strikes, s0, v0, r_d, r_f, kappa, max_xis,
                                                           max_rhos, lambdas, n_time_steps, n_strike_steps, n_sim_paths,
                                                           n_strike_steps)
    rmse = np.sqrt(((est_pvs - obs_prices) ** 2).mean())
    print('rmse of fit: ', rmse)

    if output_file is not None:
        np.save(output_file, leverage_surface)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # calibrate_leverage_surface(obs_prices, tenors, strikes, s0, r_d, r_f, n_pde_tau, n_pde_strike, n_sim_paths, n_sim_steps)
    parser.add_argument('--iv_file', type=str, required=True)
    parser.add_argument('--s0', type=float, required=True)
    parser.add_argument('--v0', type=float, required=True)
    parser.add_argument('--r_d', type=float, required=True)
    parser.add_argument('--r_f', type=float, required=True)
    parser.add_argument('--kappa', type=float, required=True)
    parser.add_argument('--lambdas', type=list_of_float, required=True)
    parser.add_argument('--max_xis', type=list_of_float, required=True)
    parser.add_argument('--max_rhos', type=list_of_float, required=True)
    parser.add_argument('--n_time_steps', type=float, required=True)
    parser.add_argument('--n_sim_paths', type=int, required=True)
    parser.add_argument('--n_strike_steps', type=int, required=True)
    parser.add_argument('--output_folder', type=str, required=False)
    parser.add_argument('--K', type=float, required=True)
    parser.add_argument('--B', type=float, required=True)
    parser.add_argument('--T', type=float, required=True)

    args = parser.parse_args()
    do_calibration(**vars(args))

