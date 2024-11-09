# Hierarchical Manifold Modeling (HMM)

This package provides routines implementing hierarchical manifold modeling data analysis methods.

## Description

The current version 1.0 of this package provides the Hierarchical Linear Discriminant Analysis (HLDA) method which is a specific application of HMM for data with one dependent categorical variable, usually encountered in the context of a classification problem.

## Getting Started

### Dependencies

* This software has been developed on an Ubuntu Linux 18.04 system. While there should not be any platform-dependent code in the package, explicit testing of proper performance has not been done on any other platform.
* Python 3.5 or higher is required to run the program.

### Installing

No compilation or installation of additional software beyond downloading the package is required.

### Executing program

The provided programs can be executed as such from the main directory of the package. E.g., the program applying HLDA to Fisher's Iris data set can be executed through
```
python -m tests.iris
```
with the resulting plots saved in the tests/iris directory.

## Authors

### Primary author

Terhi Mäkinen [terhi.makinen@fmi.fi](mailto:terhi.makinen@fmi.fi)

### Contributing authors

Seppo Pulkkinen [seppo.pulkkinen@fmi.fi](mailto:seppo.pulkkinen@fmi.fi)
Bent Harnist [bent.harnist@fmi.fi](mailto:bent.harnist@fmi.fi)

## Version History

* 1.0
    * Initial Release

## License

This project is provided under the [MIT](https://choosealicense.com/licenses/mit/) License.

## Acknowledgments

The work on the HLDA method was supported by the Academy of Finland grant 341964 as well as funding by the Finnish Defence Forces' Research Programme 2021-2024.
