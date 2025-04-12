## Minimal example

As a minimal example, we demonstrate a reproduction of the pure quartic soliton
shown in figure 2a of [Widjaja et al., <em>Absence of Galilean invariance for
pure-quartic solitons</em>, Phys. Rev. A <strong>104</strong> (2021)
043526](https://doi.org/10.1103/PhysRevA.104.043526), using the spectral
renormalization method (SRM) implemented in `SWtools`. A python script that
facilitates this is

```Python
import numpy as np
from SWtools import SRM

myS = SRM(
  np.linspace(-10, 10, 2**10),  # COMPUTATIONAL DOMAIN
  (0, 0, 0, -0.0917),           # NEVP - COEFFICIENTS OF LINEAR PART
  lambda I, xi: 4.07*I,         # NEVP - NONLINEAR FUNCTIONAL
  verbose = True                # NEVP - LIST DETAILS DURING ITERATION
)

myS.solve(
  np.exp(-myS.xi**2),           # SRM - INITIAL CONDITION
  1.76                          # SRM - EIGENVALUE OF SOUGHT-FOR SOLUTION
)

myS.show()
```

During iteration it provides details on the convergence of the solution 

```
Iter 000000: H = 3.262157, N = 1.253314, K = 2.602825, acc =  INF
Iter 000001: H = 1.551895, N = 0.884410, K = 1.754724, acc = 1.676E-01
Iter 000002: H = 1.554336, N = 0.885908, K = 1.754512, acc = 2.738E-03
Iter 000003: H = 1.561427, N = 0.888068, K = 1.758229, acc = 1.502E-03
Iter 000004: H = 1.563730, N = 0.888767, K = 1.759437, acc = 4.831E-04
Iter 000005: H = 1.564463, N = 0.888990, K = 1.759822, acc = 1.534E-04
Iter 000006: H = 1.564695, N = 0.889060, K = 1.759943, acc = 4.865E-05
Iter 000007: H = 1.564769, N = 0.889082, K = 1.759982, acc = 1.542E-05
Iter 000008: H = 1.564793, N = 0.889090, K = 1.759994, acc = 4.886E-06
Iter 000009: H = 1.564800, N = 0.889092, K = 1.759998, acc = 1.549E-06
Iter 000010: H = 1.564802, N = 0.889092, K = 1.759999, acc = 4.908E-07
Iter 000011: H = 1.564803, N = 0.889093, K = 1.760000, acc = 1.555E-07
Iter 000012: H = 1.564803, N = 0.889093, K = 1.760000, acc = 4.929E-08
Iter 000013: H = 1.564803, N = 0.889093, K = 1.760000, acc = 1.562E-08
Iter 000014: H = 1.564803, N = 0.889093, K = 1.760000, acc = 4.951E-09
Iter 000015: H = 1.564803, N = 0.889093, K = 1.760000, acc = 1.569E-09
Iter 000016: H = 1.564803, N = 0.889093, K = 1.760000, acc = 4.973E-10
Iter 000017: H = 1.564803, N = 0.889093, K = 1.760000, acc = 1.576E-10
Iter 000018: H = 1.564803, N = 0.889093, K = 1.760000, acc = 4.994E-11
Iter 000019: H = 1.564803, N = 0.889093, K = 1.760000, acc = 1.583E-11
Iter 000020: H = 1.564803, N = 0.889093, K = 1.760000, acc = 5.017E-12
Iter 000021: H = 1.564803, N = 0.889093, K = 1.760000, acc = 1.589E-12
Iter 000022: H = 1.564803, N = 0.889093, K = 1.760000, acc = 5.043E-13
Iter 000023: H = 1.564803, N = 0.889093, K = 1.760000, acc = 5.043E-13
# Iteration terminated successfully (num_iter = 23).
# Functional 1: N = 0.8890928240339375
# Functional 2: H = 1.5648033702992097
# Eigenvalue: K = 1.7599999999994147
# Local error: acc = 5.042727720296426e-13
```

Upon termination, it procudes the figure

| ![alt text](https://github.com/omelchert/SWtools/blob/main/results/numExp08_minimalExample/fig_minimalExample_v1.png)
|:--:|
|*Solution of the considered NEVP.  Left panel: Solution U. Shown are the real part (Re[U]), imaginary part (Im[U]), and modulus (|U|) of the solution.  Right panel: Variation of the accuracy upon iteration.*|
