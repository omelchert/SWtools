"""main_LogNSE_v1.py

This module demonstrates an instance of a gausson for a one-dimensional
nonlinear Schrödinger equation with a logarithmic nonlinearity, modeled after
[1]_.  This example is discussed as SWtools use-case in [2]_.

References
----------

.. [1] I. Bialynicki-Birula, J. Mycielski, Nonlinear Wave Mechanics, Annals of
Physics 100 (1976) 62, https://doi.org/10.1016/0003-4916(76)90057-9.

.. [2] O. Melchert, A. Demircan, https://doi.org/10.48550/arXiv.2504.10623.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from SWtools import NSOM

N0=1
beta=1

# -- INSTANTIATE NSOM
nevp = NSOM(
  # ... COMPUTATIONAL DOMAIN
  np.linspace(-15, 15, 2**10),
  # ... COEFFICIENTS OF LINEAR PART
  (0, -1, 0, 0),
  # ... NONLINEAR FUNCTIONAL
  lambda I, xi: beta*np.log(I),
  # ... OVERRELAXATION PARAMETER
  ORP = 1.7,
  verbose = False
)

# -- SOLUTION PROCEDURE
nevp.solve(np.exp(-nevp.xi**2), N0)

# -- POSTPROCESSING
xi, U, kap = nevp.xi, nevp.U, nevp.kap

# ... COMPARE TO EXACT RESULTS 
xi0 = np.trapz(xi*np.abs(U)**2, x=xi)
UG = np.exp(-(xi-xi0)**2/2)/np.pi**0.25
kapG = 2*np.log(np.pi**(-0.25))-1
err = np.sqrt(np.trapz(np.abs(U-UG)**2, x=xi))
print(f"U-err = {err}")
print(f"kap-err = {np.abs(kap-kapG)}")

# ... SAVE DATA
res = {
    'xi': xi,
    'beta': beta,
    'N0': N0,
    'nit': nevp.num_iter,
    'U': U,
    'kap': nevp.kap,
    'U_list': nevp.U_list,
    'N_list': nevp.N_list,
    'H_list': nevp.H_list,
    'kap_list': nevp.K_list,
    'acc_list': nevp.acc_list,
    'iter_list': nevp.iter_list
}

np.savez_compressed('./res_LNSE_NSOM_N0%lf_beta%lf'%(N0,beta), **res)

print(nevp); nevp.show()
