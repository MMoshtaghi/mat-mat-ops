import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import mat_mat_ops
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F


def reference_mat_mat_l1(a, b):
    batch_dims = a.shape[:-2]
    out_shape = (*batch_dims, a.shape[-2], b.shape[-1])

    mat1 = a.flatten(start_dim=0, end_dim=-3)
    mat2 = b.flatten(start_dim=0, end_dim=-3)

    B, M, K = mat1.shape
    B, K, N = mat2.shape
    mat2_tr = torch.transpose(input=mat2, dim0=1, dim1=2) # N,K
    my_out = torch.empty(size=(B,M,N), device=device, dtype=torch.float)
    for bs in range(B):
        for m in range(M):
            my_out[bs, m, :] = torch.abs(mat1[bs,m,:] - mat2_tr[bs,:,:]).sum(dim=-1) # (N)
    return my_out.reshape(out_shape)


class TestMatMatL1(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)

        return [
            [make_tensor(2,32,32), make_tensor(2,32,32)],
            [make_tensor(2,70,35), make_tensor(2,35,43)],
            [make_tensor(2,64,32), make_nondiff_tensor(2,32,32)],
            [make_nondiff_tensor(2,69, 32), make_tensor(2,32, 32)],
        ]

    def mat_mat_l1(self, a, b):
        torch._check(a.dim() == b.dim() , message=f'tensors a and b must have same number of dimensions')
        torch._check(a.dim() >= 3 , message=f'a.dim() must be at least 3 , but {a.dim()=}')
        for dim in range(a.dim() - 2):
            torch._check(a.shape[dim] == b.shape[dim] , message=f'tensors "a" and "b" must have same size at batch dimension: {dim=}')
        
        torch._check(a.shape[-1] == b.shape[-2] , message=f'For Matrix-Matrix operation satisfy : a.shape[-1]==b.shape[-2], but {a.shape[-1]=} and {b.shape[-2]=}')
        torch._check(a.dtype == torch.float , message=f'tensor "a" must be float, but {a.dtype=}')
        torch._check(b.dtype == torch.float , message=f'tensor "b" must be float, but {b.dtype=}')
        torch._check(a.device == b.device , message=f'tensors "a" and "b" must be on same device, but {a.device=} and {b.device=}')

        batch_dims = a.shape[:-2]
        out_shape = (*batch_dims, a.shape[-2], b.shape[-1])

        a = a.flatten(start_dim=0, end_dim=-3)
        b = b.flatten(start_dim=0, end_dim=-3)
        
        out = mat_mat_ops.ops.mat_mat_l1(a,b)

        out = out.reshape(out_shape)
        return out

    def _test_correctness(self, device):
        samples = self.sample_inputs(device)
        for args in samples:
            result = self.mat_mat_l1(*args)
            expected = reference_mat_mat_l1(*args)
            torch.testing.assert_close(result, expected)

    def test_correctness_cpu(self):
        self._test_correctness("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        self._test_correctness("cuda")

    def _test_gradients(self, device):
        samples = self.sample_inputs(device, requires_grad=True)
        for args in samples:
            diff_tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
            out = self.mat_mat_l1(*args)
            grad_out = torch.randn_like(out)
            result = torch.autograd.grad(out, diff_tensors, grad_out)

            out = reference_mat_mat_l1(*args)
            expected = torch.autograd.grad(out, diff_tensors, grad_out)

            torch.testing.assert_close(result, expected)

    def test_gradients_cpu(self):
        self._test_gradients("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_gradients_cuda(self):
        self._test_gradients("cuda")

    def _opcheck(self, device):
        # Use opcheck to check for incorrect usage of operator registration APIs
        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        for args in samples:
            opcheck(torch.ops.mat_mat_ops.mat_mat_l1.default, args)

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


def reference_muladd(a, b, c):
    return a * b + c

class TestMyMulAdd(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)

        return [
            [make_tensor(3), make_tensor(3), 1],
            [make_tensor(20), make_tensor(20), 3.14],
            [make_tensor(20), make_nondiff_tensor(20), -123],
            [make_nondiff_tensor(2, 3), make_tensor(2, 3), -0.3],
        ]

    def _test_correctness(self, device):
        samples = self.sample_inputs(device)
        for args in samples:
            result = mat_mat_ops.ops.mymuladd(*args)
            expected = reference_muladd(*args)
            torch.testing.assert_close(result, expected)

    def test_correctness_cpu(self):
        self._test_correctness("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        self._test_correctness("cuda")

    def _test_gradients(self, device):
        samples = self.sample_inputs(device, requires_grad=True)
        for args in samples:
            diff_tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
            out = mat_mat_ops.ops.mymuladd(*args)
            grad_out = torch.randn_like(out)
            result = torch.autograd.grad(out, diff_tensors, grad_out)

            out = reference_muladd(*args)
            expected = torch.autograd.grad(out, diff_tensors, grad_out)

            torch.testing.assert_close(result, expected)

    def test_gradients_cpu(self):
        self._test_gradients("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_gradients_cuda(self):
        self._test_gradients("cuda")

    def _opcheck(self, device):
        # Use opcheck to check for incorrect usage of operator registration APIs
        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        for args in samples:
            opcheck(torch.ops.mat_mat_ops.mymuladd.default, args)

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


class TestMyAddOut(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)

        return [
            [make_tensor(3), make_tensor(3), make_tensor(3)],
            [make_tensor(20), make_tensor(20), make_tensor(20)],
        ]

    def _test_correctness(self, device):
        samples = self.sample_inputs(device)
        for args in samples:
            result = args[-1]
            mat_mat_ops.ops.myadd_out(*args)
            expected = torch.add(*args[:2])
            torch.testing.assert_close(result, expected)

    def _opcheck(self, device):
        # Use opcheck to check for incorrect usage of operator registration APIs
        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        for args in samples:
            opcheck(torch.ops.mat_mat_ops.myadd_out.default, args)

    def test_opcheck_cpu(self):
        self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")


if __name__ == "__main__":
    unittest.main()
