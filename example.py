import math
import numpy as np
import scipy.stats as st
import nlopt as nlo
from WASM import WASM


def example1():
    def g1(Xi, Xd, d):
        Z1, Z2 = Xd
        return Z1**2 * Z2 / 20.0 - 1.0

    def g2(Xi, Xd, d):
        Z1, Z2 = Xd
        return (Z1 + Z2 - 5.0) ** 2 / 30.0 + (Z1 - Z2 - 12.0) ** 2 / 120.0 - 1.0

    def g3(Xi, Xd, d):
        Z1, Z2 = Xd
        return 80.0 / (Z1**2 + 8.0 * Z2 + 5.0) - 1.0

    Xd_lbub = [
        (st.norm(2.0, 0.6), st.norm(5.0, 0.6)),
        (st.norm(2.0, 0.6), st.norm(5.0, 0.6)),
    ]
    limit_state_functions = [g1, g2, g3]

    w = WASM(Xd_lbub=Xd_lbub, n_samples=50000, inferior_superior_exponent=8)
    w.compute_limit_state_functions(
        limit_state_functions=limit_state_functions,
        system_definitions=[{"serial": [0, 1, 2]}],
    )

    def constraint(mu_z1, mu_z2):
        res = w.compute_Beta_Rashki([st.norm(mu_z1, 0.6), st.norm(mu_z2, 0.6)])
        return min(res.gXs_results.betas) - 2.0
        # return min(res.systems_results.betas) - 2.0

    def f_obj(mu_Xd, grad):
        mu_z1, mu_z2 = mu_Xd
        penalty = 1e8
        a = mu_z1 + mu_z2
        constraint_val = constraint(mu_z1, mu_z2)
        if constraint_val >= 0:
            return a
        else:
            return a + penalty

    n_Xd = len(Xd_lbub)
    global_opt = nlo.opt(nlo.GN_DIRECT, n_Xd)
    global_opt.set_min_objective(f_obj)
    global_opt.set_lower_bounds([2.0] * n_Xd)
    global_opt.set_upper_bounds([5.0] * n_Xd)
    global_opt.set_maxeval(50)
    res_opt = global_opt.optimize([np.NaN] * n_Xd)
    local_opt = nlo.opt(nlo.LN_SBPLX, n_Xd)
    local_opt.set_min_objective(f_obj)
    local_opt.set_lower_bounds([2.0] * n_Xd)
    local_opt.set_upper_bounds([5.0] * n_Xd)
    local_opt.set_xtol_rel(1e-6)
    res_opt = local_opt.optimize(res_opt)
    print(res_opt, sum(res_opt))
    mu_z1_opt, mu_z2_opt = res_opt
    res = w.compute_Beta_Rashki([st.norm(mu_z1_opt, 0.6), st.norm(mu_z2_opt, 0.6)])
    print(res.gXs_results.betas)
    print(res.systems_results.betas)


def example2():
    def g1(Xi, Xd, d):
        P, E, fy, kd, ki = Xi
        D, t = d
        return fy * math.pi * D * t - P

    def g2(Xi, Xd, d):
        P, E, fy, kd, ki = Xi
        D, t = d
        nu = 0.3
        thetab = kd / np.sqrt(1 + 0.005 * D / t)
        Sel = 2 * E * t / (D * np.sqrt(3 * (1 - nu**2.0)))
        lambdab = np.sqrt(fy / (thetab * Sel))
        lambdab = np.clip(lambdab, np.sqrt(1.0 / 2.0), np.sqrt(2.0))
        flb = (1.5 - lambdab / np.sqrt(2.0)) * fy
        return flb * math.pi * D * t - P

    def g3(Xi, Xd, d):
        P, E, fy, kd, ki = Xi
        D, t = d
        lambdae = 25.0 * np.sqrt(fy / E) / (0.35 * D * math.pi)
        gamma = (lambdae**2 + ki * (lambdae - 0.2) + 0.8) / (2 * lambdae**2)
        fgb = (gamma - np.sqrt(gamma**2 - 1 / lambdae**2)) * fy
        return fgb * math.pi * D * t - P

    Xi = [
        st.norm(10.0, 0.2 * 10.0),
        st.norm(2.1e5, 0.05 * 2.1e5),
        st.norm(650.0, 0.05 * 650.0),
        st.norm(0.54, 0.16 * 0.54),
        st.norm(0.49, 0.10 * 0.49),
    ]
    limit_state_functions = [g1, g2, g3]
    w = WASM(Xi=Xi, n_samples=10000, inferior_superior_exponent=9)

    def f_obj(d, grad):
        D, t = d
        penalty = 1e9
        w.compute_limit_state_functions(
            limit_state_functions=limit_state_functions,
            system_definitions=[{"serial": [0, 1, 2]}],
            d=d,
        )
        res = w.compute_Beta_Rashki()
        constraint_val = min(res.gXs_results.betas) - 4.0
        pf = max(res.systems_results.pfs)
        C1 = 20000.0
        Cf = 1e9
        h = 25.0
        cost = C1 * h * math.pi * D * t + Cf * pf
        if constraint_val >= 0:
            return cost
        else:
            return cost + penalty

    n_d = 2
    global_opt = nlo.opt(nlo.GN_DIRECT, n_d)
    global_opt.set_min_objective(f_obj)
    global_opt.set_lower_bounds([1.0, 5e-3])
    global_opt.set_upper_bounds([3.0, 15e-3])
    global_opt.set_maxeval(10)
    res_opt = global_opt.optimize([np.NaN] * n_d)
    local_opt = nlo.opt(nlo.LN_SBPLX, n_d)
    local_opt.set_min_objective(f_obj)
    local_opt.set_lower_bounds([1.0, 5e-3])
    local_opt.set_upper_bounds([3.0, 15e-3])
    local_opt.set_xtol_rel(1e-6)
    res_opt = local_opt.optimize(res_opt)
    print(res_opt)

    w.compute_limit_state_functions(
        limit_state_functions=limit_state_functions,
        system_definitions=[{"serial": [0, 1, 2]}],
        d=res_opt,
    )
    res = w.compute_Beta_Rashki()
    print(res.gXs_results.betas, res.systems_results.betas)
    pf = max(res.systems_results.pfs)
    C1 = 20000.0
    Cf = 1e9
    h = 25.0
    D, t = res_opt
    cost = C1 * h * math.pi * D * t + Cf * pf
    print(cost)


if __name__ == "__main__":
    example1()
    # example2()
