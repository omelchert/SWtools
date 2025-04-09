""" SWtools.py

AUTHOR: O. Melchert
DATE: 2023-2025

CHANGELOG:
2023-2025:
    - Refactoring of the research code this module is based upon
    - Implementation of the base class IterBase
    - Implementation of the subclass SpectralRenormalizationMethod (SRM)
    - Implementation of the subclass SuccessiveOverrelaxationMethod (NSOM)
2025-03-17:
    - implemented plotting function for one-dimensional problems
    - added simple plotting feature to SRM and NSOM
2025-03-18:
    - Cleanup of documentation strings
    - Refactoring
2025-03-20:
    - Refactored IterBase: _accuracy() is now local a function that can be
      modified using keyword-only argument acc_fun.

"""
import os
import sys
import numpy as np
import numpy.fft as nfft
import matplotlib.pyplot as plt


# -- FFT ABREVIATIONS
FT = nfft.ifft
IFT = nfft.fft
FTFREQ = nfft.fftfreq
SHIFT  = nfft.fftshift


class IterBase(object):
    """Base class for the implemented iteration methods.

    Attributes:
        xi (array):
            Discretized coordinate xi.
        tol (float):
            Convergence is assumed if relative error falls below this value
            (defaulf: tol=1e-12).
        maxiter (int):
            Maximum number of iterations to perform (default: maxiter=10000).
        nskip (float):
            Number of iterations to skip in between storing intermediate
            results (default: 1).
        verbose (bool):
            Set to True to print details during iteration (default: False).
        iter_list (array):
            Sequence of iteration steps at which intermediate results where
            stored (separated by nskip steps).
        U_list (array):
            List of intermediate solutions U.
        acc_list (array):
            List of intermediate accuracies (see class method _accuracy() below).
        N_list (array):
            List of intermediate values of the integral N[U].
        H_list (array):
            List of intermediate values of the integral H[U].
        K_list (array):
            List of intermediate values of the eigenvalue estimate K=H[U]/N[U].

    Methods:

    """
    def __init__(self, xi, tol=1e-12, maxiter=10000, nskip=1, verbose=True):
        self.xi = xi
        self.tol = tol
        self.maxiter = maxiter
        self.nskip = nskip
        self.verbose = verbose
        self.iter_list = []
        self.U_list = []
        self.acc_list = []
        self.N_list = []
        self.H_list = []
        self.K_list = []

    @property
    def U(self):
        '''Final solution U.'''
        return self.U_list[-1]

    @property
    def kap(self):
        '''Final value of the eigenvalue.'''
        return self.K_list[-1]

    @property
    def N(self):
        '''Value of the Functional N[U].'''
        return self.N_list[-1]

    @property
    def H(self):
        '''Value of the functional H[U].'''
        return self.H_list[-1]

    @property
    def acc(self):
        '''Terminal accuracy.'''
        return self.acc_list[-1]

    @property
    def num_iter(self):
        '''Number of iterations performed.'''
        return self.iter_list[-1]

    def _clean_up(self):
        '''Cleanup of attributes at beginning of each solution run.'''
        self.U_list = []
        self.acc_list = []
        self.iter_list = []
        self.N_list = []
        self.H_list = []
        self.K_list = []

    def _update_results(self, it, U, N, H, kap, err):
        '''Update attributes using the supplied intermediate results.'''
        self.iter_list.append(it)
        self.N_list.append(N)
        self.H_list.append(H)
        self.K_list.append(kap)
        self.U_list.append(np.copy(U))
        self.acc_list.append(err)
        if self.verbose:
            print(self)

    def solve(self, U, dum, **kwargs):
        '''Performs iteration solution of the nonlinear eigenvalue problem.

        Arguments:
            U (array):
                Initial condition.
            dum (float):
                Eigenvalue (SRM) or normalization constraint (NSOM).
            **kwargs (dict):
                Keyword-only arguments:

                ortho_set (array):
                    List of previously found (orthogonal) solutions.

                acc_fun (function):
                    Custom accuracy function with call signature
                    `fun(xi, Up, Uc)`, where the arguments are:
                    - xi (float): Coordinate xi.
                    - Up (array): Solution at previous iteration step.
                    - Uc (array): Solution at current iteration step.

        Returns: (U, acc, succ, msg)
            U (array):
                Final solutions.
            acc (float):
                Terminal accuracy.
            succ (bool):
                Boolean flag indicating if teh iteration procedure exited
                successfully.
            msg (str):
                Cause of the termination.
        '''
        # -- USEFUL FUNCTIONS AND ABBREVIATIONS
        # ... STRIP SELF KEYWORD
        xi, tol, maxiter, nskip = self.xi, self.tol, self.maxiter, self.nskip
        # ... LOCAL FUNCTIONS
        acc_fun_def = lambda xi, U, Un: np.sqrt(np.sum(np.abs(Un - U)**2)/np.sum(np.abs(U)**2))
        # -- LOGIC TO MAP OUT KEYWORD ARGUMENTS
        ortho_set = kwargs.get("ortho_set", [])
        acc_fun = kwargs.get("acc_fun", acc_fun_def)
        # -- CLEAN UP
        self._clean_up()

        # -- PREPARE/EVALUATE TRIAL SOLUTION
        N = self._N(U)
        H = self._H(U)
        # -- INITIALIZE ITERATION SCHEME
        it = 1
        acc = np.inf
        # -- KEEP INITIAL RESULTS
        self._update_results(0, U, N, H, H/N, acc)
        # -- ITERATION PROCEDURE
        while acc > tol and it < maxiter:
            # ... ORTHOGONALIZE WRT NOT-LIST 
            if ortho_set:
                U = self._orthogonalize(U, ortho_set)
            # ... UPDATE SOLUTION 
            U_new = self._singleIteration(U, N, H, dum)
            # ... ROOT-MEAN SQUARED DEVIATION W.R.T. PREVIOUS STEP 
            acc = acc_fun(xi, U, U_new)
            # ... ADVANCE UPDATE TO NEXT ITERATION STEP 
            U = U_new
            # ... RE-EVALUATE THE INTEGRALS FOR THE CURRENT SOLUTION 
            N = self._N(U_new)
            H = self._H(U_new)
            # ... KEEP INTERMEDIATE RESULTS
            if it%nskip==0:
                self._update_results(it, U, N, H, H/N, acc)
            it+=1

        # -- KEEP FINAL SOLUTION 
        # ... ORTHOGONALIZE WRT NOT-LIST 
        if ortho_set:
            U = self._orthogonalize(U, ortho_set)
        self._update_results(it, U, N, H, H/N, acc)

        # -- BASIC STATUS MESSAGE
        if it<maxiter:
            msg = f'Iteration terminated successfully (num_iter = {it}).'
        else:
            if it>=maxiter:
                msg = 'Maximum number of iterations reached.'
            if np.isnan(np.any(U)) or np.isnan(acc):
                msg = 'NaN result encountered.'

        if self.verbose:
            print(f"# {msg}")
            print(f"# Functional 1: N = {N}")
            print(f"# Functional 2: H = {H}")
            print(f"# Eigenvalue: K = {H/N}")
            print(f"# Local error: acc = {acc}")

        return self.U, self.acc, it<maxiter, msg

    def _accuracy(self, U, U_new):
        '''Accuracy.

        The accuracy is calculated as the root-mean-squared deviation from the
        previous iteration step.

        Note:
            The function can be overwritten to realize other convergence
            criteria [AM2009].

        References:
            [AM2009]
                M. J. Ablowitz and T.P. Horikis,
                "Solitons and spectral renormalization methods in nonlinear optics",
                Eur. Phys. J. Special Topics 173 (2009) 147–166.

        Arguments:
            U (array): Solution at iteration step n-1.
            U_new (array): Solution at iteration step n.

        Returns: (acc)
            acc (float): Accuracy
        '''
        return np.sqrt(np.sum(np.abs(U_new - U)**2)/np.sum(np.abs(U)**2))

    def _orthogonalize(self, U, ortho_set):
        '''Otrhogonalization procedure.

        Implements a straight forward projection construction that turns the
        solution U into a solution U_ortho, which is orthogonal to any of the
        functions in the set ortho_set.

        Arguments:
            U (array): Current solution.
            ortho_set (list): Orthogonal set of previously found solutions.

        Returns: (U_ortho)
            U_ortho (array): New solution which is orthogonal to any previously
            found solution.
        '''
        xi = self.xi
        # -- PROJECTION CONSTRUCTION 
        for V in ortho_set:
            # ... PROJECT OUT COMPONENT V 
            U -= V*np.sum(np.conj(V)*U)/np.sum(np.abs(V)**2)
        return U

    def _H(self, U):
        raise NotImplementedError

    def _N(self, U):
        raise NotImplementedError

    def _singleIteration(self, U, N, H, q):
        raise NotImplementedError

    def __str__(self):
        myStr = f"Iter {self.num_iter:06d}: H = {self.H:7.6F}, N = {self.N:7.6F}, K = {self.kap:7.6F}, acc = {self.acc:4.3E}"
        return myStr


class SpectralRenormalizationMethod(IterBase):
    def __init__(self, xi, coeffs, F, tol=1e-12, maxiter=10000, nskip=1, verbose=False):
        IterBase.__init__(self, xi, tol=tol, maxiter=maxiter, nskip=nskip, verbose=verbose)
        self.F = F
        self.Lw = self._set_Lk(xi, coeffs)
        self.gam = 1.5

    def _set_Lk(self, xi, coeffs):
        c1, c2, c3, c4 = coeffs
        k = FTFREQ(xi.size, d=xi[1]-xi[0])*2*np.pi
        return c1*k + c2*k**2 + c3*k**3 + c4*k**4

    def _N(self, U):
        return np.trapz(np.abs(U)**2, x=self.xi, axis=-1)

    def _H(self, U):
        xi, Lw, F = self.xi, self.Lw, self.F
        return np.real(np.trapz( np.conj(U)*IFT(Lw*FT(U)) + F(np.abs(U)**2, xi)*np.abs(U)**2, x=xi, axis=-1))

    def _singleIteration(self, U, N, H, kap):
        # -- STRIP SELF KEYWORD
        xi, Lw, F, gam = self.xi, self.Lw, self.F, self.gam
        # -- USEFUL FUNCTIONS AND ABBREVIATIONS 
        _IP = lambda f,g: np.trapz(np.conj(f)*g, x=xi)

        # -- SRM UPDATE
        # (1) PROPOSE UPDATED SOLUTION
        U_tmp = IFT(FT(F(np.abs(U)**2,xi)*U)/(kap - Lw))
        # (2) RESCALE SOLUTION SO IT SATISFIES THE DESIRED INTEGRAL IDENTITY
        s_tmp = np.abs(_IP(U,U)/_IP(U,U_tmp))
        U_tmp *= s_tmp**gam

        return U_tmp

    def show(self, f_name='none'):
        xi, U, it, acc = self.xi, self.U, self.iter_list, self.acc_list
        show_xy(xi, U, it, acc, f_name=f_name)



class SuccessiveOverrelaxationMethod(IterBase):
    """Nonlinear Successive Overrelaxation Method

    Implements a nonlinear successive overrelaxation method [LM2019] for a
    generalized nonlinear Schrödinger equation. The procedure uses a
    Gauss-Seidel method with overrelaxation to iteratively updata a
    user-provided initial condition in place [PTVF2007,LL2017].

    Arguments:
        xi (1D numpy array, floats):
            Discrete mesh for coordinate xi.
        cL (list, floats):
            Coefficients (c1, c2, c3, c4) for the linear operator.
        F (function):
            Nonlinear functional with function call signature
            `F(I, xi)`, where the arguments are:
            - I (float): Squared magnitude of the function A.
            - xi (float): Coordinate xi.
        ORP (float):
            Overrelaxation parameter (default: ORP = 1.0).
        tol (float):
            Iteration is stopped when the accuracy falls below this threshold
            (default: tol = 1e-12).
        maxiter (int):
            Maximum number of allowed iterations (default: maxiter = 10000).
        nskip (int):
            Number of iterations to skip in between kept intermediate results
            (default: nskip = 1).
        verbose (bool):
            Set to True to print details during iteration (default: False).

    References:
        [PTVF2007]
            W. Press, S. Teukolsky, W. Vetterling, B. Flannery,
            Numerical Recipes: The Art of Scientific Computing,
            Cambridge University Press (2007).
        [LL2017]
            H. P. Langtangen, S. Linge,
            Finite-Difference Computing with PDEs,
            Springer (2017).
        [LM2019]
            H. P. Langtangen, K.-A. Mardal,
            Introduction to Numerical Methods for Variational Problems,
            Springer (2019).
    """
    def __init__(self, xi, cL, F, ORP=1.0, tol=1e-12, maxiter=10000, nskip=1, verbose=False):
        IterBase.__init__(self, xi, tol=tol, maxiter=maxiter, nskip=nskip, verbose=verbose)
        self.dxi = xi[1]-xi[0]
        self.F = F
        self.coeffs = self._set_coeffs(cL)
        self.ORP = ORP

    def _set_coeffs(self,coeffs):
        dxi = self.dxi
        dxi2 = dxi**2
        c1, c2, c3, c4 = coeffs
        return [ 1j*dxi*c1/12 + c2/12 + 0.5j*c3/dxi + c4/dxi2,\
                -2j*dxi*c1/3 -c2*4/3 - 1j*c3/dxi - c4*4/dxi2,\
                -c2*5/2 - c4*6/dxi2,\
                 2j*dxi*c1/3 -c2*4/3 + 1j*c3/dxi - c4*4/dxi2,\
                -1j*dxi*c1/12 + c2/12 - 0.5j*c3/dxi + c4/dxi2]

    def _N(self, U):
        return np.trapz(np.abs(U)**2, x=self.xi)

    def _H(self, U):
        xi, dxi, F = self.xi, self.dxi, self.F
        am2, am1, a0, ap1, ap2 = self.coeffs
        HL = np.trapz( (am2*U[:-4] + am1*U[1:-3] - a0*U[2:-2] + ap1*U[3:-1] + ap2*U[4:] )*np.conj(U[2:-2]) , dx=1)/dxi
        HN = np.trapz(F(np.abs(U)**2, xi)*np.abs(U)**2, x=xi)
        return np.real(HL + HN)

    def _singleIteration(self, U, N, H, N0):
        # -- STRIP SELF KEYWORD
        xi, dxi, F, ORP = self.xi, self.dxi, self.F, self.ORP
        am2, am1, a0, ap1, ap2 = self.coeffs
        # -- LOGIC TO MAP OUT ARGUMENTS
        #N0 = kwargs.get("N0", 1.0)
        # -- USEFUL FUNCTIONS AND ABBREVIATIONS 
        _norm = lambda U, N0: U*np.sqrt(N0/np.trapz(np.abs(U)**2,x=xi))

        # -- GAUSS-SEIDEL RELAXATION PROCEDURE
        for i in range(2,xi.size-2):
            # ... KEEP OLD VALUE FOR REFERENCE
            Ui = U[i]
            xii = xi[i]
            # ... AVERAGE VALUE ACCROSS LOCAL NEIGHBORHOOD
            Ui_nbs_wgt = am2*U[i-2] + am1*U[i-1] + ap1*U[i+1] + ap2*U[i+2]
            # ... LOCAL POTENTIAL
            Vi = -F(np.abs(Ui)**2, xii)
            # ... UPDATE PIVOT
            num = Ui_nbs_wgt
            den =  a0 + (H/N + Vi)*dxi**2
            Ui_new = num/den
            # ... UPDATE SOLUTION USING OVERRELAXATION 
            U[i] += ORP*(Ui_new-Ui)

        # -- SCALE SOLUTION TO DESIRED ENERGY 
        U = _norm(U,N0)

        return U

    def show(self, f_name='none'):
        xi, U, it, acc = self.xi, self.U, self.iter_list, self.acc_list
        show_xy(xi, U, it, acc, f_name=f_name)


def show_xy(xi, U, it, acc, f_name='none'):
    # -- FIGURE LAYOUT 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))

    # -- LEFT SUBPLOT
    ax1.plot(xi, np.real(U), color='C0', dashes=[], label=r"${\rm{Re}}[U]$")
    ax1.plot(xi, np.imag(U), color='C0', dashes=[2,1], label=r"${\rm{Im}}[U]$")
    ax1.plot(xi, np.abs(U), color='gray', dashes=[1,1], label=r"$|U|$")
    ax1.set_xlabel(r"Coordinate $\xi$")
    ax1.set_ylabel(r"Solution $U$")
    legend = ax1.legend()

    # -- RIGHT SUBPLOT
    ax2.plot(it, np.log10(acc), color='C0')
    ax2.set_xlabel(r"Iteration $n$")
    ax2.set_ylabel(r"log-accuracy $\log(\epsilon_n)$")

    # -- SHOW/SAVE FIGURE
    plt.tight_layout()

    if f_name=='none':
        plt.show()
    else:
        plt.savefig(f_name)


NSOM = SuccessiveOverrelaxationMethod
SRM = SpectralRenormalizationMethod
