import torch
from torch import nn
import torch.distributed as dist


class DDPNaive(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

        for param in module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for param in self.module.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()

class DDPBatch(DDPNaive):

    def finish_gradient_synchronization(self):
        gradients = [param.grad for param in self.module.parameters() if param.grad is not None]

        flat_grads = torch._utils._flatten_dense_tensors(gradients)
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads /= dist.get_world_size()

        synced_grads = torch._utils._unflatten_dense_tensors(flat_grads, gradients)

        for original_grad, synced_grad in zip(gradients, synced_grads):
            original_grad.copy_(synced_grad)


class DDPOverlapIndividualParameters(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.handles = []

        def async_all_reduce(param):
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)

        for param in module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(async_all_reduce)

        for param in module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

        for param in self.module.parameters():
            if param.requires_grad:
                param.grad /= dist.get_world_size()


