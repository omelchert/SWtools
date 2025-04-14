import numpy as np
from numpy.fft import fftfreq, ifft
from SWtools import SRM

# -- SETUP AND INITIALIZATION 
xi = np.linspace(-20, 20, 2**10)
# ... NEVP INGREDIENTS 
cL = (-0.07, -0.2/2, 0.2/6, -1./24)
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
U, N = myS.U, myS.N
k = fftfreq(xi.size, d=xi[1]-xi[0])*2*np.pi
Ik = np.abs(ifft(U))**2
kc = np.trapz(k*Ik, x=k)/np.trapz(Ik, x=k)
print(f"max(U) = {np.max(np.abs(U)):5.4F}")
print(f"N[U]   = {N:5.4F}")
print(f"kc     = {kc:5.4F}")
