"""main_HONSE_v1.py

This module demonstrates the calculation of a solitary wave solution for a
higher-order nonlinear Schrödinger equation, modeled after [1]_. This example
is discussed as SWtools use-case in [2]_.

References
----------

.. [1] O. Melchert, A. Demircan, Numerical investigation of solitary-wave
solutions for the nonlinear Schrödinger equation perturbed by third-order and
negative fourth-order dispersion, Phys. Rev. A 110 (2024) 043518,
https://doi.org/10.1103/PhysRevA.110.043518.

.. [2] O. Melchert, A. Demircan, https://doi.org/10.48550/arXiv.2504.10623.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from numpy.fft import fftfreq, ifft
from SWtools import SRM

# -- SETUP AND INITIALIZATION 
xi = np.linspace(-20, 20, 2**10)
# ... NEVP INGREDIENTS 
cL = (-0.07, -0.1, 0.0333, -0.0417)
F = lambda I, xi: I
kap = 0.867
# ... INITIAL CONDITION 
U0 = np.exp(-xi**2)
# ... SRM INSTANTIATION
myS = SRM(xi, cL, F, verbose=True)
# ... CUSTOM ACCURACY
acc = lambda xi, U, V: np.max(np.abs(U-V))

# -- SRM SOLUTION PROCEDURE
myS.solve(U0, kap, acc_fun = acc)

# -- POSTPROCESSING
k = fftfreq(xi.size, d=xi[1]-xi[0])*2*np.pi
Ik = np.abs(ifft(myS.U))**2
kc = np.trapz(k*Ik, x=k)/np.trapz(Ik, x=k)
print(f"max(U) = {np.max(myS.U):5.4F}")
print(f"kc     = {kc:5.4F}")
print(myS); myS.show()
