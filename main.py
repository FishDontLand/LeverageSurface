import argparse
import os

import numpy as np
import pandas as pd

from src.localvol import iv_to_price
from src.mcpricer import calibrate_leverage_surface, compute_barrier_pvs, MCStochasticVolModel, PiecewiseTermStructure, \
    simulate_with_leverage_surface


def list_of_float(arg):
    n_split = arg.split(",")
    return [eval(e) for e in n_split]

def _parse_user_query(query):
    parts = [p.strip() for p in query.split(",")]
    if len(parts) != 4:
        raise ValueError("expected: K,B,T,metric")

    k = float(parts[0])
    b = float(parts[1])
    t = float(parts[2])
    metric = parts[3].upper()
    return k, b, t, metric


def _interactive_pricing_loop(leverage_surface, base_model, r_d, tenors, strikes):
    print("Ready for pricing Barriers.")
    print("Input format: K,B,T,metric  (example: 1.1,1.4,2.0,PV)")
    print("Type 'exit' to quit.")

    while True:
        raw = input("> ").strip()
        if raw.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        try:
            k, b, t, metric = _parse_user_query(raw)
            if metric == "PV":
                s_sim, _ = simulate_with_leverage_surface(leverage_surface, tenors, strikes, base_model)
                pv = compute_barrier_pvs(
                    s_sim=s_sim,
                    sim_times=base_model.times,
                    r_d=r_d,
                    tenors=np.array([t]),
                    strikes=np.array([k]),
                    B=b,
                )[0, 0]
                print(f"PV(K={k}, B={b}, T={t}) = {pv:.10f}")
            elif metric == "DELTA":
                s_sim1, _ = simulate_with_leverage_surface(leverage_surface, tenors, strikes, base_model)
                pv1 = compute_barrier_pvs(
                    s_sim=s_sim1,
                    sim_times=base_model.times,
                    r_d=r_d,
                    tenors=np.array([t]),
                    strikes=np.array([k]),
                    B=b,
                )[0, 0]

                eps = 1e-2
                base_model.s0 += eps
                s_sim2, _ = simulate_with_leverage_surface(leverage_surface, tenors, strikes, base_model)
                pv2 = compute_barrier_pvs(
                    s_sim=s_sim2,
                    sim_times=base_model.times,
                    r_d=r_d,
                    tenors=np.array([t]),
                    strikes=np.array([k]),
                    B=b,
                )[0, 0]
                delta = (pv2 - pv1) / eps
                base_model.s0 -= eps
                print(f"Delta(K={k}, B={b}, T={t}) = {delta:.10f}")
            else:
                raise ValueError("unknown metric")


        except Exception as err:
            print(f"invalid input or query failed: {err}")

def do_calibration(iv_file, s0, v0, r_d, r_f, kappa, max_xis, max_rhos, lambdas,
                   n_time_steps, n_sim_paths, n_strike_steps, output_folder=None):
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

    if output_folder is not None:
        leverage_surface_frame = pd.DataFrame(leverage_surface, index=tenors, columns=strikes)
        leverage_surface_frame.to_csv(os.path.join(output_folder, "leverage_surface.csv"))

        obs_price_frame = pd.DataFrame(obs_prices, index=tenors, columns=strikes)
        est_pv_frame = pd.DataFrame(est_pvs, index=tenors, columns=strikes)

        obs_price_frame.to_csv(os.path.join(output_folder, "obs_prices.csv"))
        est_pv_frame.to_csv(os.path.join(output_folder, "est_pvs.csv"))

    base_model =  MCStochasticVolModel(max(tenors), n_sim_paths, n_strike_steps, r_d, r_f, v0, s0, kappa,
                                      PiecewiseTermStructure([1,2,3,5], max_rhos),
                                      PiecewiseTermStructure([1,2,3,5], max_xis),
                                      PiecewiseTermStructure([1,2,3,5], lambdas))

    s_sim, _ = simulate_with_leverage_surface(leverage_surface, tenors, strikes, base_model)
    _interactive_pricing_loop(leverage_surface, base_model, r_d, tenors, strikes)

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

    args = parser.parse_args()
    do_calibration(**vars(args))

