import argparse
import numpy as np
import pandas as pd

from src.localvol import iv_to_price
from src.mcpricer import calibrate_leverage_surface


def do_calibration(iv_file, s0, r_d, r_f, n_time_steps, n_sim_paths, n_strike_steps, output_path=None):
    iv_data = pd.read_csv(iv_file)
    iv_data.columns = [c.lower() for c in iv_data.columns]
    tenors = np.sort(iv_data['term'].unique())
    strikes = np.sort(iv_data['strike'])

    iv_data = iv_data.sort_values(by=['term', 'strike'])
    assert len(iv_data) == len(tenors) * len(strikes), "iv surface must be a grid"
    obs_vols = iv_data['iv'].values.reshape([len(tenors), len(strikes)])
    obs_prices = iv_to_price(s0, r_f, r_d, strikes, obs_vols, tenors)
    # (obs_prices, tenors, strikes, s0, r_d, r_f, n_pde_tau, n_pde_strike, n_sim_paths, n_sim_steps)
    leverage_surface, est_pvs = calibrate_leverage_surface(obs_prices, tenors, strikes, s0, r_d, r_f,
                                                           n_time_steps, n_strike_steps, n_sim_paths, n_strike_steps)
    rmse = np.sqrt(((est_pvs - obs_prices) ** 2).mean())
    print('rmse of fit: ', rmse)

    if output_path is not None:
        np.save(output_path, leverage_surface)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # calibrate_leverage_surface(obs_prices, tenors, strikes, s0, r_d, r_f, n_pde_tau, n_pde_strike, n_sim_paths, n_sim_steps)
    parser.add_argument('--iv_file', type=str, required=True)
    parser.add_argument('--s0', type=float, required=True)
    parser.add_argument('--r_d', type=float, required=True)
    parser.add_argument('--r_f', type=float, required=True)
    parser.add_argument('--n_time_steps', type=float, required=True)
    parser.add_argument('--n_sim_paths', type=int, required=True)
    parser.add_argument('--n_strike_steps', type=int, required=True)
    parser.add_argument('--leverage_file', type=str, required=False)

    args = parser.parse_args()

