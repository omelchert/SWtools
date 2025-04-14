import numpy as np
from SWtools import NSOM

# -- SETUP AND INITIALIZATION 
xi = np.linspace(-16, 16, 1025)
# ... NEVP INGREDIENTS 
cL = (0, -0.5, 0, 0)
beta = 31.371
F = lambda I, x: -(0.5*x**2 + beta*I)
N0 = 1.0
# ... INITIAL CONDITION 
U0 = np.exp(-xi**2/2)/np.pi**0.25
# ... NSOM INSTANTIATION
myS = NSOM(xi, cL, F, ORP=1.5, verbose=True, nskip=10)

# -- NSOM SOLUTION PROCEDURE
myS.solve(U0, N0)

# -- POSTPROCESSING (RMS WIDTH AND ENERGY)
U, mu = myS.U, -myS.kap
xi_rms = np.sqrt(np.trapz((xi*U)**2,x=xi))
E = mu - 0.5*beta*np.trapz(U**4,x=xi)
print(f"max(U) = {np.max(U):5.4F}")
print(f"xi_rms = {xi_rms:5.4F}")
print(f"E_beta = {E:5.4F}")
print(f"mu     = {mu:5.4F}")

myS.show()
