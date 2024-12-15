# mat-mat-ops
Custom C++ and CUDA operators for Matrix-Matrix Operations in PyTorch. Here I implemented Shared Memory Cache-Blocking and Block-tiling for both forward and backward kernels.

This [tutorial](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#setting-up-the-build-system) by PyTorch is all you need :)

## Requirements:
CUDA Toolkit 12.4
PyTorch 2.4+

## Supported operations so far:
- Mat-Mat Mul
- Mat-Mat L1

## To build:
```
pip install .
```

## To test:
the interactive option : `test/test.ipynb`
or
```
python test/test_extension.py
```

## Author

[Mehdi Moshtaghi](https://github.com/MMoshtaghi)
