# SWtools 

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](https://docs.python.org/3/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2504.10623-b31b1b.svg)](https://doi.org/10.48550/arXiv.2504.10623)

`SWtools` is a Python package containing data structures and algorithms that
allow a user to conveniently calculate solitary-wave solutions for a general
nonlinear Schrödinger-type wave equation.

It provides an extendible framework for iterative methods that allow a user to
solve two variants of the associated nonlinear eigenvalue problem (NEVP):

* A bare version of the NEVP, where a solution with given eigenvalue is computed;
* A constraint version of the NEVP with <em>a priori</em> unknown eigenvalue, where a solution with given norm is computed.

To facilitate progress of science, we include many examples and workflows that
can help a user to quickly go from an idea to numerical experimentation to
results. We also provide a verification test based on a known analytical
solution for a higher order nonlinear Schrödinger equation, studied in the
literature (see the reference manual below).

The examples provided with `SWtools` include: Bose-Einstein condensate (BEC)
solutions for a Gross-Pitaevskii equation (GPE), traveling solitary wave
solutions of a higher-order nonlinear Schrödinger equation (HONSE), nonlinear
bound-states in a nonlinear Schrödinger equation (NSE) with periodic nonlinear
microstructure, excited-states for a GPE, solitary waves in a cubic-quintic
NSE, solitons in a saturable NSE, and solitary waves in a two-dimensional NSE.

## Installation 

`SWtools` is developed under python3 (version 3.9.7) and requires

* numpy (1.21.2)
* scipy (1.7.0)
* matplotlib (3.4.3)

The software can be installed by
[cloning](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
the repository as

``$ git clone https://github.com/omelchert/SWtools``

Within a Python script, add the path to your Python path and import `SWtools`:

```Python
import sys; sys.path.append('/path/to/SWtools-package')
import SWtools

```

As an alternative, working on the commandline, add the path by amending `.bash_profile` by the line

```bash 
export PYTHONPATH="${PYTHONPATH}:/path/to/SWtools-package"
```


## Further information

- **Source:** [https://github.com/omelchert/SWtools/SWtools](SWtools/)

- **Minimal example:** [https://github.com/omelchert/SWtools/blob/main/doc/minimalExample.md](doc/minimalExample.md)

- **Reference manual:** <https://doi.org/10.48550/arXiv.2504.10623>

- **Documentation:** <https://omelchert.github.io//SWtools/doc/html/SWtools_base.html>

- **Software integration:** `SWtools` can be used as an extension module for
  [py-fmas](https://doi.org/10.17632/7s2cv9kjfs.1) (Melchert and Demircan, [2022](https://doi.org/10.1016/j.cpc.2021.108257)) and
  [GNLStools](https://github.com/ElsevierSoftwareX/SOFTX-D-22-00165) (Melchert and Demircan, [2022](https://doi.org/10.1016/j.softx.2022.101232)), allowing a user to
  study the propagation dynamics of the obtained solutions. An example using
  `SWtools` in conjunction with `py-fmas` is included under
  `results/numExp06_HONSE_FMAS`.

- **Extendibility:** While the documented codebase assumes a one-dimenaional
  (d=1) transverse coordinate, extension to higher dimensions is straight
  forward. An example of an extension module [SWtools_ext_SRM2D](SWtools/SWtools_ext_SRM2D.py)
  (documented [here](<https://omelchert.github.io//SWtools/doc/html/SWtools_ext_SRM2D.html>)),
  implementing a spectral renormalization method for d=2 by subclassing
  `SWtools` base class `IterBase` is included with the example under
  `results/numExp07_2DNSE_SRM2D`. 

- **References:** [https://github.com/omelchert/SWtools/blob/main/doc/references.md](doc/references.md)

- **Code Ocean:** A code ocean compute capsule demonstrating the calculation of
  a soliton solution for a higher-order nonlinear Schrödinger equation via the
  spectral renormalization method is available under [TBW](). 


## License 

This project is licensed under the MIT License - see the
[LICENSE.md](LICENSE.md) file for details.


## Acknowledgments

This work received funding from the Deutsche Forschungsgemeinschaft  (DFG)
under Germany’s Excellence Strategy within the Cluster of Excellence PhoenixD
(Photonics, Optics, and Engineering – Innovation Across Disciplines) (EXC 2122,
projectID 390833453).
