import sys; sys.path.append('../../src/')
import numpy as np
import numpy.fft as nfft
from SWtools import IterBase

FT = np.fft.ifftn
IFT = np.fft.fftn
FTFREQ = np.fft.fftfreq
SHIFT = np.fft.fftshift


class SRM2D(IterBase):
    def __init__(self, xi, coeffs, F, tol=1e-12, maxiter=10000, nskip=1, verbose=False):
        IterBase.__init__(self, xi, tol=tol, maxiter=maxiter, nskip=nskip, verbose=verbose)

        # -- SETUP AND INITIALIZATION
        xx, yy = xi
        # ... 2D VOLUME ELEMENT FOR INTEGRATION
        dx, dy = xx[1,0]-xx[0,0], yy[0,1]-yy[0,0]
        self.dV = dx*dy
        # ... FOURIER SAMPLES OF THE TRANSVERSE COORDINATE
        kx = FTFREQ(xx.shape[0], d=dx)*2*np.pi
        ky = FTFREQ(yy.shape[1], d=dy)*2*np.pi
        kxx, kyy = np.meshgrid(kx, ky, indexing='ij')
        # ... FOURIER REPRESENTATION OF THE LINEAR OPERATOR
        cx, cy = coeffs
        self.Lk = cx*kxx**2 + cy*kyy**2
        # ... NONLINEAR FUNCTIONAL
        self.F = F
        # ... con
        self.gam = 1.5


    def _N(self, U):
        return np.sum(np.abs(U)**2)*self.dV

    def _H(self, U):
        xi, Lk, F, dV = self.xi, self.Lk, self.F, self.dV
        return np.real(np.sum( np.conj(U)*IFT(Lk*FT(U)) + F(np.abs(U)**2, xi)*np.abs(U)**2)*dV)

    def _singleIteration(self, U, N, H, kap):
        # -- STRIP SELF KEYWORD
        xi, Lk, F, gam, dV = self.xi, self.Lk, self.F, self.gam, self.dV
        # -- USEFUL FUNCTIONS AND ABBREVIATIONS 
        _IP = lambda f,g: np.sum(np.conj(f)*g)*dV

        # -- SRM UPDATE
        # (1) PROPOSE UPDATED SOLUTION
        U_tmp = IFT(FT(F(np.abs(U)**2,xi)*U)/(kap - Lk))
        # (2) RESCALE SOLUTION SO IT SATISFIES THE DESIRED INTEGRAL IDENTITY
        s_tmp = np.abs(_IP(U,U)/_IP(U,U_tmp))
        U_tmp *= s_tmp**gam

        return U_tmp

    def show(self, f_name='none'):
        xi, U, it, acc = self.xi, self.U, self.iter_list, self.acc_list
        show_xyz(xi, U, it, acc, f_name=f_name)



def show_xyz(xi, U, it, acc, f_name='none'):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

    # -- FIGURE LAYOUT 
    fig = plt.figure(figsize=(7.,3.5))
    #fig = plt.figure(figsize=plt.figaspect(0.5)) #figsize=(6,3))

    # -- LEFT SUBPLOT
    xx, yy = xi
    U = np.real(U)
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax.plot_surface(xx, yy, U, edgecolor='C0', lw=0.5, rstride=5, cstride=5, alpha=0.3)
    ax.set_xlabel('Coordinate $x$')
    ax.set_ylabel('Coordinate $y$')
    ax.set_zlabel('Solution $U(x,y)$')

    # -- RIGHT SUBPLOT
    ax = fig.add_subplot(1,2,2)
    ax.plot(it, np.log10(acc), color='C0')
    ax.set_xlabel(r"Iteration $n$")
    ax.set_ylabel(r"log-accuracy $\log(\epsilon_n)$")

    # -- SHOW/SAVE FIGURE
    plt.subplots_adjust(left=0.02, right=0.96,  bottom=0.14, top=0.96, wspace=0.5)

    if f_name=='none':
        plt.show()
    else:
        plt.savefig(f_name)



