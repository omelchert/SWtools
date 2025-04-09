
# SWtools 

`SWtools.py` is a Python module containing data structures and algorithms that
allow a user to conveniently calculate solitary-wave solutions for a general
nonlinear Schrödinger-type wave equation.

In this regard, it provides an extendible framework for iterative methods that
allow a user to solve two variants of the associated nonlinear eigenvalue
problem (NEVP):

* A bare version of the NEVP;
* A constraint version of the NEVP wherein an additional normalization constraint for the solution is imposed.


To facilitate progress of science, we include many examples and workflows that
can help a user to quickly go from an idea to numerical experimentation to
results. In particular, we provide a verification test based on a known
analytical solution for a higher order nonliear Schrödinger equation, studied
in the literature.


## Prerequisites

`SWtools` is developed under python3 (version 3.9.7) and requires the
functionality of 

* numpy (1.21.2)
* scipy (1.7.0)

Further, the figure generation scripts included with the examples require the
functionality of

* matplotlib (3.4.3)

`SWtools` can be used as an extension module for
[py-fmas](https://github.com/omelchert/py-fmas) and
[GNLStools](https://github.com/omelchert/GNLStools.git), allowing a user to
take advantage of data structures and functions that permit pulse propagation
simulations for various nonlinear wave equations via fixed as well as variable
stepsize z-propagation algorithms. An example that uses `SWtools` in
conjunction with `py-fmas` is included in the repository.


## Availability of the software

The `SWtools` module presented here is derived from our research software and
is meant to work as a (system-)local software tool. There is no need to install
it once you got a local
[clone](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
of the repository, e.g. via

``$ git clone https://github.com/omelchert/SWtools``


## Further informaion

The `GNLStools' software package is described in 

> O. Melchert, A. Demircan, TBW. 

The presented software has been extensively used in our research work, and has
previously contributed to the process of scientific discovery in the field of
nonlinear optics

> O. Melchert and A. Demircan, "Numerical investigation of solitary-wave solutions for the nonlinear Schrödinger equation perturbed by third-order and negative fourth-order dispersion", Phys. Rev. A 110, 043518 (2024). 

> O. Melchert and A. Demircan, "Optical Solitary Wavelets," [arXiv.2410.06867](https://doi.org/10.48550/arXiv.2410.06867).


## License 

This project is licensed under the MIT License - see the
[LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

This work received funding from the Deutsche Forschungsgemeinschaft  (DFG)
under Germany’s Excellence Strategy within the Cluster of Excellence PhoenixD
(Photonics, Optics, and Engineering – Innovation Across Disciplines) (EXC 2122,
projectID 390833453).
