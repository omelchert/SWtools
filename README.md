# SWtools 

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`SWtools` is a Python module containing data structures and algorithms that
allow a user to conveniently calculate solitary-wave solutions for a general
nonlinear Schrödinger-type wave equation.

It provides an extendible framework for iterative methods that allow a user to
solve two variants of the associated nonlinear eigenvalue problem (NEVP):

* A bare version of the NEVP;
* A constraint version of the NEVP wherein an additional normalization constraint for the solution is imposed.

To facilitate progress of science, we include many examples and workflows that
can help a user to quickly go from an idea to numerical experimentation to
results. We also provide a verification test based on a known analytical
solution for a higher order nonlinear Schrödinger equation, studied in the
literature.


## Installation 

`SWtools` is developed under python3 (version 3.9.7) and requires

* numpy (1.21.2)
* scipy (1.7.0)
* matplotlib (3.4.3)

The software can be installed by
[cloning](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
the repository as

``$ git clone https://github.com/omelchert/SWtools``


## Further information

- **Source:** <https://github.com/omelchert/SWtools>

- **Documentation:** ARXIV ARTICLE FOLLOWING SOON.

- **Reference manual:** <https://omelchert.github.io//SWtools/doc/html/SWtools.html>

- **Minimal example:** [see here](doc/minimalExample.md)

- **Software integration:** `SWtools` can be used as an extension module for
  [py-fmas](https://doi.org/10.17632/7s2cv9kjfs.1) (see [Melchert and Demircan,
  Comp. Phys. Commun. 273 (2022)
  108257](https://doi.org/10.1016/j.cpc.2021.108257)) and
  [GNLStools](https://github.com/ElsevierSoftwareX/SOFTX-D-22-00165) (see
  [Melchert and Demircan, SoftwareX 20 (2022)
  101232](https://doi.org/10.1016/j.softx.2022.101232)), allowing a user to
  study the propagation dynamics of the obtained solutions. An example using
  `SWtools` in conjunction with `py-fmas` is included under
  `results\numExp06_HONSE_FMAS`.

- **Extendibility:** While the documented codebase assumes a one-dimenaional
  (d=1) transverse coordinate, extension to higher dimensions is straight
  forward. An example implementing a spectral renormalization method for d=2 is
  included under `results\numExp07_2DNSE_SRM2D`. 


## License 

This project is licensed under the MIT License - see the
[LICENSE.md](LICENSE.md) file for details.


## Acknowledgments

This work received funding from the Deutsche Forschungsgemeinschaft  (DFG)
under Germany’s Excellence Strategy within the Cluster of Excellence PhoenixD
(Photonics, Optics, and Engineering – Innovation Across Disciplines) (EXC 2122,
projectID 390833453).
