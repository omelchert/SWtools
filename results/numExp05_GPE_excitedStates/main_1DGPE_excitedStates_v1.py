import sys; sys.path.append('../../src/')
import numpy as np
from SWtools import NSOM

# -- SETUP AND INITIALIZATION 
xi = np.linspace(-8, 8, 512)
# ... NEVP INGREDIENTS 
cL = (0,-0.5,0,0)
F = lambda I, xi: -0.5*xi**2 - I
N0 = 1.0
# ... INITIAL CONDITIONS 
UI0 = np.exp(-xi**2/2)/np.pi**0.25
UI1 = 2*xi*UI0/np.sqrt(2)
UI2 = (4*xi**2-2)*UI0/np.sqrt(8)
UI3 = (8*xi**3-12*xi)*UI0/np.sqrt(48)
# ... INSTANTIATE NSOM
myS = NSOM(xi, cL, F, tol=1e-12, ORP=1.5, maxiter=10000, nskip=10, verbose=True)

# -- NSOM SOLUTION PROCEDURE
# ... GROUND STATE
myS.solve(UI0, N0)
U0, mu0 = myS.U, -myS.kap
print(f"# mu0 = {mu0:4.3F}")
myS.show()
# ... 1ST EXCITED STATE
myS.solve(UI1, N0, ortho_set = [U0])
U1, mu1 = myS.U, -myS.kap
print(f"# mu1 = {mu1:4.3F}")
myS.show()
# ... 2ND EXCITED STATE
myS.solve(UI2, N0, ortho_set = [U0, U1])
U2, mu2 = myS.U, -myS.kap
print(f"# mu2 = {mu2:4.3F}")
myS.show()
# ... 3RD EXCITED STATE
myS.solve(UI3, N0, ortho_set = [U0, U1, U2])
U3, mu3 = myS.U, -myS.kap
print(f"# mu3 = {mu3:4.3F}")
myS.show()
