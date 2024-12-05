# mat-mat-ops
CUDA implementation of some Matrix-Matrix Operation kernels. (Both forward and backward path)
This repo has both custom CPU and CUDA kernels.

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
