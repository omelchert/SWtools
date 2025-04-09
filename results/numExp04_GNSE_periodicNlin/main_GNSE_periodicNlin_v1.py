import sys; sys.path.append('../../src/')
import numpy as np
from SWtools import SRM

# -- SETUP AND INITIALIZATION 
xi = np.linspace(-20, 20, 2**12)
# ... NEVP INGREDIENTS 
cL = (0, -1., 0, 0)
m = lambda xi: -0.8*np.cos(4*np.pi*xi)
F = lambda I, xi: (1 + m(xi))*I
kap = 1.0
# ... INITIAL CONDITION
U0 = np.exp(-xi**2)
# ... SRM INSTANTIATION
myS = SRM(xi, cL, F)

# -- SRM SOLUTION PROCEDURE
myS.solve(U0, kap)

# -- POSTPROCESSING
U, N = myS.U.real, myS.N
print(f"# N      = {N:4.3F}")
print(f"# max(U) = {np.max(U):4.3F}")
print(f"# U(0)   = {U[xi.size//2]:4.3F}")
