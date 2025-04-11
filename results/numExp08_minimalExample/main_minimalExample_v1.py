import sys; sys.path.append('../../src/')
import numpy as np
from SWtools import SRM

myS = SRM(
  np.linspace(-10, 10, 2**10),  # COMPUTATIONAL DOMAIN
  (0, 0, 0, -0.0917),           # NEVP - COEFFS. LINEAR PART
  lambda I, xi: 4.07*I,         # NEVP - NONLINEAR FUNCTIONAL
  verbose = True                # NEVP - LIST DETAILS DURING ITERATION
)

myS.solve(
  np.exp(-myS.xi**2),           # SRM - INITIAL CONDITION
  1.76                          # SRM - EIGENVALUE OF SOUGHT-FOR SOLUTION
)

myS.show('fig_minimalExample_v1.png')
