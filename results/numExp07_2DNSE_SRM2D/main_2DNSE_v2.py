import sys; sys.path.append('../../src/')
import numpy as np
from SWtools_SRM2D import SRM2D

# -- SETUP AND INITIALIZATION 
x1 = np.linspace(-20, 20, 2**8)
x2 = np.linspace(-20, 20, 2**8)
X1, X2 = np.meshgrid(x1, x2, indexing='ij')
# ... NEVP INGREDIENTS 
cL = (-1., -1.)
V0=3.
cos2 = lambda x: np.cos(x)**2
V = lambda x: V0*(cos2(x[0]) + cos2(x[1]))
F = lambda I, xi: V(xi) + I
kap = 3.7045
# ... INITIAL CONDITION 
U0 = np.exp(-(X1**2+X2**2))
# ... SRM INSTANTIATION
myS = SRM2D((X1, X2), cL, F, tol=1e-12, verbose=True, nskip=100)

# -- 2DSRM SOLUTION PROCEDURE
myS.solve(U0, kap)

# -- POSTPROCESSING
U, H, N = myS.U, myS.H, myS.N
print(f"max(U) = {np.max(U):5.4F}")
print(f"H[U]   = {H:5.4F}")
print(f"N[U]   = {N:5.4F}")

results = {
    'xi': myS.xi,
    'U': myS.U,
    'iter_list': myS.iter_list,
    'acc_list': myS.acc_list,
}
np.savez_compressed('./res_2DNSE_SRM2D_kap%lf'%(kap), **results)

myS.show()
