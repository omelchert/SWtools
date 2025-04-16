"""main_GNSE_periodicNlin_v2.py

This module demonstrates the calculation of a nonlinear bound-state in a medium
where the nonlinear refractive index is modulated along the transverse
direction, modeled after [1]_. This example is discussed as SWtools use-case in
[2]_.

References
----------

.. [1] G. Fibich, Y. Sivan, M. I. Weinstein, Bound states of nonlinear
Schrödinger equations with a periodic nonlinear microstructure, Physica
D 217 (2006) 31, http://dx.doi.org/10.1016/j.physd.2006.03.009.

.. [2] O. Melchert, A. Demircan, https://doi.org/10.48550/arXiv.2504.10623.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from SWtools import SRM

# -- SETUP AND INITIALIZATION 
xi = np.linspace(-20, 20, 2**12)
# ... NEVP INGREDIENTS 
cL = (0, -1., 0, 0)
F = lambda I, xi:\
  (1 - 0.8*np.cos(4*np.pi*xi))*I
kap = 1.0
# ... SRM INSTANTIATION
myS = SRM(xi, cL, F)

# -- SRM SOLUTION PROCEDURE
myS.solve(np.exp(-xi**2), kap)

# -- POSTPROCESSING
print(f"# max(U) = {np.max(myS.U):4.3F}")
print(f"# U(0)   = {myS.U[xi.size//2]:4.3F}")
print(myS); myS.show()
