# RBDO_examples

The main objective of this repo is to provide examples on how to use [WASM](https://github.com/rcsalomao/WASM) to solve Reliability Based Design Optimization problems.

## Requirements

This repo makes use of the following libs:

- numpy
- scipy
- nlopt
- WASM

## Examples

### Nonlinear mathematical optimization problem.

$$
\begin{aligned}
\textrm{find: }& \mu_i^\* = \\{\mu_1^\*,\mu_2^\*\\}; \\
\textrm{that minimizes: }& f^{obj}(\mu_i) = \mu_1 + \mu_2; \\
\textrm{subject to: }& P[g_j(Z_i) \le 0] \le p_j^{fT} = \Phi[-\beta_j^T];\\
\textrm{with: }& \beta_j^T \ge 2.0; \\
& Z_i \sim N(\mu_i, 0.6); \\
& 0 \le \mu_i \le 10; \\
& g_1(Z_i) = \frac{Z_1^2Z_2}{20} - 1; \\
& g_2(Z_i) = \frac{(Z_1 + Z_2 - 5)^2}{30} + \frac{(Z_1 - Z_2 - 12)^2}{120} - 1; \\
& g_3(Z_i) = \frac{80}{(Z_1^2 + 8Z_2 + 5)} - 1; \\
\end{aligned}
$$

```python
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
```

| Reference          | $\mu_1$ | $\mu_2$ | $f^{obj}$ |
| ------------------ | ------- | ------- | --------- |
| [1]                | 3.660   | 3.609   | 7.269     |
| WASM + nlopt       | 3.657   | 3.615   | 7.272     |
| Relative diff. (%) | -0.082  | 0.166   | 0.041     |

| Reference          | $\beta_1$ | $\beta_2$ | $\beta_3$ |
| ------------------ | --------- | --------- | --------- |
| [1]                | 2.0       | 2.0       | 4.436     |
| WASM + nlopt       | 2.0       | 2.0       | 4.397     |
| Relative diff. (%) | 0.0       | 0.0       | -0.879    |

### Optimization of column considering buckling effect.

$$
\begin{aligned}
\textrm{find: }& d_n^\* = \\{D^\*, t^\*\\}; \\
\textrm{that minimizes: }& f^{obj}(d_n) = C^t(D,t) = C_1\pi hDt + C^fP^f; \\
\textrm{subject to: }& \beta_j(Xi_m,d_n) \ge \beta_j^T;\\
\textrm{with: }& \beta_j^T \ge 4.0; \\
& 1 \le D \le 3; \\
& 4 \le t \le 15; \\
& g_1(Xi_m,d_n) = g^{y}(Xi_m,d_n) = f^{y}\pi Dt - P; \\
& g_2(Xi_m,d_n) = g^{lb}(Xi_m,d_n) = f^{lb}\pi Dt - P; \\
& g_3(Xi_m,d_n) = g^{gb}(Xi_m,d_n) = f^{gb}\pi Dt - P; \\
\end{aligned}
$$

$$
\begin{aligned}
f^{lb} &= \left(1.5 - \frac{1}{\sqrt{2}} \lambda^b\right) \textrm{ with}\\
\lambda^b &= \sqrt{\frac{f^y}{\theta^b S^{el}}} \textrm{ for } \sqrt{\frac{1}{2}} \le \lambda^b \le \sqrt{2},\\
S^{el} &= \frac{2Et}{d\sqrt{3(1 - \nu^2)}} \textrm{ and }\\
\theta^b &= \frac{k^d}{\sqrt{1 + 0.005D/t}}
\end{aligned}
$$



$$
\begin{aligned}
f^{gb} &= \left(\gamma - \sqrt{\gamma^2 - 1/{\lambda^e}^2}\right)f^y \textrm{ with}\\
\gamma &= \frac{1}{2{\lambda^e}^2}({\lambda^e}^2 + k^i(\lambda^e - 0.2) + 0.8) \textrm{ and }\\
\lambda^e &= \frac{h}{0.35D\pi}\sqrt{\frac{f^y}{E}}
\end{aligned}
$$

| R. V. Name | Description            | R. V. Type | Mean  | Coef. of Var. (%) |
| ---------- | ---------------------- | ---------- | ----- | ----------------- |
| $P$        | Vertical load          | Normal     | 10.0  | 20                |
| $E$        | Young's modulus        | Normal     | 2.1e5 | 5                 |
| $f^y$      | Yield stress           | Normal     | 650.0 | 5                 |
| $k^d$      | Knock-down factor      | Normal     | 0.54  | 16                |
| $k^i$      | Imperfection parameter | Normal     | 0.49  | 10                |

```python
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
        generate_RV_2_param(st.norm, 2.1e5, 0.05 * 2.1e5),
        generate_RV_2_param(st.norm, 650.0, 0.05 * 650.0),
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
```

| Reference          | $D$   | $t$      | $f^{obj}$ |
| ------------------ | ----- | -------- | --------- |
| [2]                | 1.4   | 10.3e-3  | 23.0e3    |
| WASM + nlopt       | 1.366 | 10.56e-3 | 23.38e3   |
| Relative diff. (%) | -2.42 | 2.52     | 1.65      |

| Reference          | $\beta_1$ | $\beta_2$ | $\beta_3$ | $\beta_\textrm{serial system}$ |
| ------------------ | --------- | --------- | --------- | ------------------------------ |
| [2]                | 8.04      | 4.84      | 5.01      | 4.77                           |
| WASM + nlopt       | 8.16      | 4.93      | 4.96      | 4.81                           |
| Relative diff. (%) | 1.49      | 1.86      | -1.00     | 0.84                           |

## References

[1]: Youn BD, Choi KK. An investigation of nonlinearity of reliability-Based Design Optimization approaches. Journal of mechanical design (New York, N.Y.: 1990). 2004;126(3):403–411.

[2]: Enevoldsen I, Sørensen JD. Reliability-based optimization in structural engineering. Structural safety. 1994;15(3):169–196.
