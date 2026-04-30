# pyfhd
Fast Holographic Deconvolution in python

[![Python](https://img.shields.io/badge/Python-3.10--3.13-%231475b3?logo=python&logoColor=%23fff)](https://www.python.org/)
![GitHub last commit](https://img.shields.io/github/last-commit/EoRImaging/pyfhd?logo=github&color=blue&link=https%3A%2F%2Fgithub.com%2FEoRImaging%2Fpyfhd%2Fcommits%2Fmain%2F)
![GitHub License](https://img.shields.io/github/license/EoRImaging/pyfhd)

![GitHub branch check runs](https://img.shields.io/github/check-runs/EoRImaging/pyfhd/main)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/EoRImaging/pyfhd/test.yml?logo=python&logoColor=white&label=tests&link=https%3A%2F%2Feorimaging.github.io%2Fpyfhd%2Fpyfhd_report.html%3Fsort%3Dresult)
[![Documentation Status](https://readthedocs.org/projects/pyfhd/badge/?version=latest)](https://pyfhd.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![PyPI - Version](https://img.shields.io/pypi/v/pyfhd)
![Static Badge](https://img.shields.io/badge/DockerHub-skywa7ch3r%2Fpyfhd%3Alatest-blue?logo=docker&logoColor=white)


[![Static Badge](https://img.shields.io/badge/Test%20Data%20DOI-10.5281%2Fzenodo.15687722-grey?labelColor=blue)](https://doi.org/10.5281/zenodo.15687722)
[![Static Badge](https://img.shields.io/badge/pyfhd%20Repository%20DOI-10.5281%2Fzenodo.15720184-blue)](https://doi.org/10.5281/zenodo.15720184)




## FHD
FHD is an open-source imaging algorithm for radio interferometers, specifically tested on MWA Phase I, MWA Phase II, PAPER, and HERA. There are three main use-cases for FHD: efficient image deconvolution for general radio astronomy, fast-mode Epoch of Reionization analysis, and simulation.

pyfhd is the translated library of FHD from IDL to Python, it aims to get close to the same results as the original FHD project. Do expect some minor differences compared to the original FHD project due to the many differences between IDL and Python. These differences are often due to the difference in precision between IDL and Python with IDL being single-precision (accurate upto 1e-8) and Python being double-precision (1e-16). Some of the IDL functions are double-precision but most default to single-precision.

## Quick Start
```
pip install pyfhd
```

For full installation notes, including dependencies on pyfhd, check out the [ReadTheDocs installation page](https://pyfhd.readthedocs.io/en/latest/installation/installation.html).

To check if pyfhd is available on your path, run the following command:

```
pyfhd -v
```

You should see output that resembles something like this:

```
    ________________________________________________________________________
    |    ooooooooo.               oooooooooooo ooooo   ooooo oooooooooo.    |
    |    8888   `Y88.             8888       8 8888    888   888     Y8b    |
    |    888   .d88' oooo    ooo  888          888     888   888      888   |
    |    888ooo88P'   `88.  .8'   888oooo8     888ooooo888   888      888   |
    |    888           `88..8'    888          888     888   888      888   |
    |    888            `888'     888          888     888   888     d88'   |
    |    o888o            .8'     o888o        o888o   o888o o888bood8P'    |
    |                 .o..P'                                                |
    |                `Y8P'                                                  |
    |_______________________________________________________________________|
    
    Python Fast Holographic Deconvolution 

    Translated from IDL to Python as a collaboration between Astronomy Data and Computing Services (ADACS) and the Epoch of Reionisation (EoR) Team.

    Repository: https://github.com/EoRImaging/pyfhd

    Documentation: https://pyfhd.readthedocs.io/en/latest/

    Version: 1.0.1

    Git Commit Hash: aa3cddb69cb617d88cb95d8b3d177d934f1c5d01 (tutorial_adjustments)
```

To run the examples built into the repository and beyond, please find them here: [pyfhd Examples](https://pyfhd.readthedocs.io/en/latest/examples/examples.html)

## Useful Documentation Resources
 - [pyfhd documentation](https://pyfhd.readthedocs.io/en/latest/)
 - [MWA ASVO](https://asvo.mwatelescope.org/) - service to obtain MWA data
 - [FHD repository](https://github.com/EoRImaging/FHD) - the original IDL code
 - [FHD examples](https://github.com/EoRImaging/FHD/blob/master/examples.md) - examples on how to use the original IDL code
 - [FHD pipeline scripts](https://github.com/EoRImaging/pipeline_scripts) - pipeline scripts using the original IDL code

## Community Guidelines
We are an open-source community that interacts and discusses issues via GitHub. We encourage collaborative development. New users are encouraged to submit issues and pull requests and to create branches for new development and exploration. Comments and suggestions are welcome.

If you wish to contribute to pyfhd, first of all thank you, second please read the contribution guide which can be found here, [Contribution Guide](https://pyfhd.readthedocs.io/en/latest/develop/contribution_guide.html). The contribution will cover all you need to know for developing in pyfhd from adding features, formatting adding tests and some advice in translating IDL to Python.

### Citing pyfhd

If you use pyfhd for a paper, the way to cite pyfhd is using the DOI link:

[https://doi.org/10.5281/zenodo.15720184](https://doi.org/10.5281/zenodo.15720184)

From the Zenodo site, you can either copy or export the citation type you need (e.g. BibTeX).

TODO: A JOSS Paper is being done and will be submitted soon, put pre-print or JOSS paper itself here to also cite

## Maintainers
FHD was built by Ian Sullivan and the University of Washington radio astronomy team.
Maintenance is a group effort split across University of Washington and Brown
University, with contributions from University of Melbourne and Arizona State University. 

pyfhd was initiated by Nichole Barry and Astronomy Data and Computing Services
(ADACS) member Joel Dunstan. ADACS is a collaboration between the University of
Swinburne and Curtin Institute for Data Science (CIDS) located in Curtin University.
Ongoing development and maintenance is provided by Nichole Barry (University of
New South Wales Sydney) and Bryna Hazelton (University of Washington).

Thank you to the previous maintainers:
Jack Line - Astronomy Data and Computing Services (ADACS)

Acknowledgements to Bryna Hazelton and Paul Hancock for their advice and knowledge.

