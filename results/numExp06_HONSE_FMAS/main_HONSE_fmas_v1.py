import numpy as np
from SWtools import SRM, FT, IFT, FTFREQ
from fmas.solver import LEM
from fmas.tools import plot_evolution

# -- SET UP DOMAIN AND MODEL 
xi = np.linspace(-50, 50, 2**13)
k = FTFREQ(xi.size, d=xi[1]-xi[0])*2*np.pi
c1, c2, c3, c4 = 0.458, -1/2, 1/12, -1/24
F = lambda I, xi: I
N = lambda U: F(np.abs(U)**2,xi)*U
kap = 0.618

# -- SET UP AND SOLVE NEVP 
NEVP = SRM(xi, (c1,c2,c3,c4), F, verbose=True)
NEVP.solve(np.exp(-xi**2), kap)

# -- SET UP AND SOLVE IVP
u0 = FT(NEVP.U)*(1+np.exp(-5j*k+2.49826j))

Lk = 1j*(c2*k**2 + c3*k**3 + c4*k**4)
Nk = lambda u: 1j*FT(N(IFT(u)))
IVP = LEM(Lk, Nk, del_G=1e-8)
IVP.set_initial_condition(k, u0)
IVP.propagate(z_range = 90.,
              n_steps = 1000,
              n_skip = 1)

# -- SAVE DATA
res = {
    'xi':  xi,
    'U0':  NEVP.U,
    'it':  NEVP.iter_list,
    'acc': NEVP.acc_list,
    'eta': IVP.z,
    'U':   IVP.utz,
    'del': IVP._del_rle,
    'h':   IVP._dz_a
}
np.savez_compressed('res.npz',**res)
