import sys; sys.path.append('../../src/')
import numpy as np
from SWtools import SRM

# -- SETUP AND INITIALIZATION 
xi = np.linspace(-10, 10, 2**10)
# ... NEVP INGREDIENTS 
cL = (0., 0., 0., -2.2/24)
#cL = (-0.07, -0.2/2, 0.2/6, -1./24)
F = lambda I, xi: 4.07*I
kap = 1.76
# ... INITIAL CONDITION 
U0 = np.exp(-xi**2)
# ... SRM INSTANTIATION
myS = SRM(xi, cL, F, verbose=True)

# -- SRM SOLUTION PROCEDURE
myS.solve(U0, kap)

myS.show('fig_minimalExample_v1.png')
