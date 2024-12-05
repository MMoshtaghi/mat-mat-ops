import torch
from torch import Tensor

__all__ = ["mymuladd", "myadd_out", "mat_mat_l1"]

#########################################################################

def mymuladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.mat_mat_ops.mymuladd.default(a, b, c)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("mat_mat_ops::mymuladd")
def _(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def mymuladd_backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = torch.ops.mat_mat_ops.mymul.default(grad, b)
    if ctx.needs_input_grad[1]:
        grad_b = torch.ops.mat_mat_ops.mymul.default(grad, a)
    return grad_a, grad_b, None


def mymuladd_setup_context(ctx, inputs, output):
    a, b, c = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "mat_mat_ops::mymuladd", mymuladd_backward, setup_context=mymuladd_setup_context)

##########################################################################

@torch.library.register_fake("mat_mat_ops::mymul")
def _(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)

##########################################################################

def myadd_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    """Writes a + b into out"""
    torch.ops.mat_mat_ops.myadd_out.default(a, b, out)


##########################################################################


def mat_mat_ops_dimension_check(tensor1, tensor2, bias=None):
    # Get dimensions of the input tensors
    dim1 = tensor1.dim()
    dim2 = tensor2.dim()

    # Case 2: Both tensors are 2-dimensional (matrix-matrix product)
    if dim1 == 2 and dim2 == 2:
        if tensor1.size(1) != tensor2.size(0):
            raise ValueError(f"Size mismatch between tensor 1 and 2: {tensor1.size()} vs {tensor2.size()}")
    
        tensor1_broadcasted_shape = (1,) + tensor1.shape
        tensor2_broadcasted_shape = (1,) + tensor2.shape

        out_shape = ( tensor1.size(-2), tensor2.size(-1) )
        if bias:
            if bias.size(0) != tensor2.size(-1):
                raise ValueError(f"Size mismatch between bias and output: {bias.size()} vs {out_shape}")

        return tensor1_broadcasted_shape, tensor2_broadcasted_shape, out_shape

    # Case 5: Either tensor1 or tensor2 is N-dimensional (N > 2), batched matrix multiplication
    elif (dim1 >= 2 and dim2 >= 2):
        # Ensure broadcastability of the batch dimensions
        # Broadcast the batch dimensions
        try:
            broadcasted_shape = torch.broadcast_shapes(tensor1.shape[:-2], tensor2.shape[:-2])
        except RuntimeError as e:
            raise ValueError(f"Batch dimensions are not broadcastable: {tensor1.shape[:-2]} vs {tensor2.shape[:-2]}") from e

        # Ensure matrix dimensions match for multiplication
        if tensor1.size(-1) != tensor2.size(-2):
            raise ValueError(f"Size mismatch in matrix dimensions: {tensor1.size()} vs {tensor2.size()}")
        
        tensor1_broadcasted_shape = broadcasted_shape + tensor1.shape[-2:]
        tensor2_broadcasted_shape = broadcasted_shape + tensor2.shape[-2:]

        batch_dims = tensor1_broadcasted_shape[:-2]
        out_shape = (*batch_dims, tensor1.size(-2), tensor2.size(-1) )

        if bias:
            if bias.size(0) != tensor2.size(-1):
                raise ValueError(f"Size mismatch between bias and output: {bias.size()} vs {out_shape}")

        return tensor1_broadcasted_shape, tensor2_broadcasted_shape, out_shape

    else:
        raise ValueError("Invalid dimensions for batch_mat_mat_ops.")


def sum_over_broadcasted_dims(grad_output, original_shape, broadcasted_shape):
    """
    Compute the sum over broadcasted dimensions to match the original shape.
    
    Args:
        grad_output (Tensor): The gradient output tensor with broadcasted dimensions.
        original_shape (tuple): The shape of the original tensor before broadcasting.
        broadcasted_shape (tuple): The broadcasted shape after broadcasting.
    
    Returns:
        Tensor: The gradient tensor summed over broadcasted dimensions.
    """
    # Find the broadcasted dimensions by comparing original_shape and broadcasted_shape
    original_dims = len(original_shape)
    broadcast_dims = len(broadcasted_shape)

    original_shape_tmp = original_shape

    # Expand original_shape to match the length of broadcasted_shape (prepend 1s)
    if original_dims < broadcast_dims:
        original_shape_tmp = (1,) * (broadcast_dims - original_dims) + original_shape
    
    # Identify which dimensions were broadcasted (i.e., size 1 in the original_shape_tmp)
    broadcasted_dims = [i for i in range(broadcast_dims) if original_shape_tmp[i] == 1 and broadcasted_shape[i] != 1]
    
    # Sum over the broadcasted dimensions
    if broadcasted_dims:
        grad_output = torch.sum(grad_output, dim=tuple(broadcasted_dims), keepdim=False)
    
    # Squeeze the summed dimensions if necessary to match the original_shape
    return grad_output.view(original_shape)


########################################################################################################


def mat_mat_mul(a: Tensor, b: Tensor) -> Tensor:
    """Performs mat_mat_l1 in an efficient fused kernel"""
    a_broadcast_shape , b_broadcast_shape, out_shape = mat_mat_ops_dimension_check(a, b)

    torch._check(a.dtype == torch.float , message=f'tensor "a" must be float, but {a.dtype=}')
    torch._check(b.dtype == torch.float , message=f'tensor "b" must be float, but {b.dtype=}')
    torch._check(a.device == b.device , message=f'tensors "a" and "b" must be on same device, but {a.device=} and {b.device=}')
    
    out = torch.ops.mat_mat_ops.mat_mat_mul.default(a.expand(a_broadcast_shape).flatten(start_dim=0, end_dim=-3), 
                                                   b.expand(b_broadcast_shape).flatten(start_dim=0, end_dim=-3) )

    out = out.reshape(out_shape)
    return out
    


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("mat_mat_ops::mat_mat_mul")
def _(a, b):
    a_broadcast_shape , b_broadcast_shape, out_shape = mat_mat_ops_dimension_check(a, b)

    torch._check(a.shape[-1] == b.shape[-2] , message=f'For Matrix-Matrix operation satisfy : a.shape[-1]==b.shape[-2], but {a.shape[-1]=} and {b.shape[-2]=}')
    torch._check(a.dtype == torch.float , message=f'tensor "a" must be float, but {a.dtype=}')
    torch._check(b.dtype == torch.float , message=f'tensor "b" must be float, but {b.dtype=}')
    torch._check(a.device == b.device , message=f'tensors "a" and "b" must be on same device, but {a.device=} and {b.device=}')

    return torch.empty(size=out_shape, dtype=a.dtype, device=a.device)


def _mat_mat_mul_backward(ctx, grad):
    a, b = ctx.saved_tensors
    a_broadcast_shape , b_broadcast_shape, out_grad_shape = mat_mat_ops_dimension_check(a, b)

    torch._check( out_grad_shape==grad.shape )

    grad = grad.flatten(start_dim=0, end_dim=-3)
    a_broadcasted = a.expand(a_broadcast_shape).flatten(start_dim=0, end_dim=-3)
    b_broadcasted = b.expand(b_broadcast_shape).flatten(start_dim=0, end_dim=-3)


    grad_a, grad_b = None, None
    grad_a_broadcasted, grad_b_broadcasted = None, None
    
    if ctx.needs_input_grad[0]:
        grad_a_broadcasted = torch.ops.mat_mat_ops.mat_mat_mul.default(grad,
                                                                       b_broadcasted.transpose(dim0=-1 , dim1=-2))
    grad_a = sum_over_broadcasted_dims(grad_a_broadcasted, a.shape, a_broadcast_shape)
    
    if ctx.needs_input_grad[1]:
        grad_b_broadcasted = torch.ops.mat_mat_ops.mat_mat_mul.default(a_broadcasted.transpose(dim0=-1 , dim1=-2),
                                                                       grad)
    grad_b = sum_over_broadcasted_dims(grad_b_broadcasted, b.shape, b_broadcast_shape)
    
    return grad_a, grad_b, None


def _mat_mat_mul_setup_context(ctx, inputs, output):
    a, b = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "mat_mat_ops::mat_mat_mul", _mat_mat_mul_backward, setup_context=_mat_mat_mul_setup_context)


##########################################################################

def mat_mat_l1(a: Tensor, b: Tensor) -> Tensor:
    """Performs mat_mat_l1 in an efficient fused kernel"""
    
    a_broadcast_shape , b_broadcast_shape, out_shape = mat_mat_ops_dimension_check(a, b)

    torch._check(a.dtype == torch.float , message=f'tensor "a" must be float, but {a.dtype=}')
    torch._check(b.dtype == torch.float , message=f'tensor "b" must be float, but {b.dtype=}')
    torch._check(a.device == b.device , message=f'tensors "a" and "b" must be on same device, but {a.device=} and {b.device=}')
    
    out = torch.ops.mat_mat_ops.mat_mat_l1.default(a.expand(a_broadcast_shape).flatten(start_dim=0, end_dim=-3), 
                                                   b.expand(b_broadcast_shape).flatten(start_dim=0, end_dim=-3) )

    out = out.reshape(out_shape)
    return out
    


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("mat_mat_ops::mat_mat_l1")
def _(a, b):
    a_broadcast_shape , b_broadcast_shape, out_shape = mat_mat_ops_dimension_check(a, b)

    torch._check(a.shape[-1] == b.shape[-2] , message=f'For Matrix-Matrix operation satisfy : a.shape[-1]==b.shape[-2], but {a.shape[-1]=} and {b.shape[-2]=}')
    torch._check(a.dtype == torch.float , message=f'tensor "a" must be float, but {a.dtype=}')
    torch._check(b.dtype == torch.float , message=f'tensor "b" must be float, but {b.dtype=}')
    torch._check(a.device == b.device , message=f'tensors "a" and "b" must be on same device, but {a.device=} and {b.device=}')

    return torch.empty(size=out_shape, dtype=a.dtype, device=a.device)


def _mat_mat_l1_backward(ctx, grad):
    a, b = ctx.saved_tensors
    a_broadcast_shape , b_broadcast_shape, out_grad_shape = mat_mat_ops_dimension_check(a, b)

    # ! out and out_grad tensors must be the same as out_shape
    torch._check( out_grad_shape==grad.shape )

    grad = grad.flatten(start_dim=0, end_dim=-3)
    a_broadcasted = a.expand(a_broadcast_shape).flatten(start_dim=0, end_dim=-3)
    b_broadcasted = b.expand(b_broadcast_shape).flatten(start_dim=0, end_dim=-3)


    grad_a, grad_b = None, None
    grad_a_broadcasted, grad_b_broadcasted = None, None
    if ctx.needs_input_grad[0]:
        grad_for_A = True
        grad_a_broadcasted = torch.ops.mat_mat_ops.mat_mat_l1_grad.default(grad,
                                                                            b_broadcasted.transpose(dim0=-1,
                                                                                                    dim1=-2),
                                                                            a_broadcasted,
                                                                            grad_for_A)
    grad_a = sum_over_broadcasted_dims(grad_a_broadcasted, a.shape, a_broadcast_shape)
    
    if ctx.needs_input_grad[1]:
        grad_for_A = False
        grad_b_broadcasted = torch.ops.mat_mat_ops.mat_mat_l1_grad.default(a_broadcasted.transpose(dim0=-1,
                                                                                                   dim1=-2),
                                                                            grad,
                                                                            b_broadcasted,
                                                                            grad_for_A)
    grad_b = sum_over_broadcasted_dims(grad_b_broadcasted, b.shape, b_broadcast_shape)

    return grad_a, grad_b, None


def _mat_mat_l1_setup_context(ctx, inputs, output):
    a, b = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "mat_mat_ops::mat_mat_l1", _mat_mat_l1_backward, setup_context=_mat_mat_l1_setup_context)

##########################################################################

def mat_mat_l1_linear(a: Tensor, b: Tensor, bias:Tensor) -> Tensor:
    """Performs mat_mat_l1 in an efficient fused kernel"""
    
    a_broadcast_shape , b_broadcast_shape, out_shape = mat_mat_ops_dimension_check(a, b, bias)

    torch._check(a.dtype == torch.float , message=f'tensor "a" must be float, but {a.dtype=}')
    torch._check(b.dtype == torch.float , message=f'tensor "b" must be float, but {b.dtype=}')
    torch._check(a.device == b.device , message=f'tensors "a" and "b" must be on same device, but {a.device=} and {b.device=}')
    if bias:
        torch._check(bias.dtype == torch.float , message=f'tensor "bias" must be float, but {bias.dtype=}')
        torch._check(a.device == bias.device , message=f'tensors "a", "b" and "bias" must be on same device, but {a.device=} and {bias.device=}')
        
    out = torch.ops.mat_mat_ops.mat_mat_l1.default(a.expand(a_broadcast_shape).flatten(start_dim=0, end_dim=-3), 
                                                   b.expand(b_broadcast_shape).flatten(start_dim=0, end_dim=-3) )

    out = out.reshape(out_shape)
    return out
    


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("mat_mat_ops::mat_mat_l1_linear")
def _(a, b):
    a_broadcast_shape , b_broadcast_shape, out_shape = mat_mat_ops_dimension_check(a, b)

    torch._check(a.shape[-1] == b.shape[-2] , message=f'For Matrix-Matrix operation satisfy : a.shape[-1]==b.shape[-2], but {a.shape[-1]=} and {b.shape[-2]=}')
    torch._check(a.dtype == torch.float , message=f'tensor "a" must be float, but {a.dtype=}')
    torch._check(b.dtype == torch.float , message=f'tensor "b" must be float, but {b.dtype=}')
    torch._check(a.device == b.device , message=f'tensors "a" and "b" must be on same device, but {a.device=} and {b.device=}')

    return torch.empty(size=out_shape, dtype=a.dtype, device=a.device)


def _mat_mat_l1_linear_backward(ctx, grad):
    a, b = ctx.saved_tensors
    a_broadcast_shape , b_broadcast_shape, out_grad_shape = mat_mat_ops_dimension_check(a, b)

    # ! out and out_grad tensors must be the same as out_shape
    torch._check( out_grad_shape==grad.shape )

    grad = grad.flatten(start_dim=0, end_dim=-3)
    a_broadcasted = a.expand(a_broadcast_shape).flatten(start_dim=0, end_dim=-3)
    b_broadcasted = b.expand(b_broadcast_shape).flatten(start_dim=0, end_dim=-3)


    grad_a, grad_b = None, None
    grad_a_broadcasted, grad_b_broadcasted = None, None
    if ctx.needs_input_grad[0]:
        grad_for_A = True
        grad_a_broadcasted = torch.ops.mat_mat_ops.mat_mat_l1_grad.default(grad,
                                                                            b_broadcasted.transpose(dim0=-1,
                                                                                                    dim1=-2),
                                                                            a_broadcasted,
                                                                            grad_for_A)
    grad_a = sum_over_broadcasted_dims(grad_a_broadcasted, a.shape, a_broadcast_shape)
    
    if ctx.needs_input_grad[1]:
        grad_for_A = False
        grad_b_broadcasted = torch.ops.mat_mat_ops.mat_mat_l1_grad.default(a_broadcasted.transpose(dim0=-1,
                                                                                                   dim1=-2),
                                                                            grad,
                                                                            b_broadcasted,
                                                                            grad_for_A)
    grad_b = sum_over_broadcasted_dims(grad_b_broadcasted, b.shape, b_broadcast_shape)

    return grad_a, grad_b, None


def _mat_mat_l1_linear_setup_context(ctx, inputs, output):
    a, b = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "mat_mat_ops::mat_mat_l1_linear", _mat_mat_l1_linear_backward, setup_context=_mat_mat_l1_linear_setup_context)