
import torch
from typing import Type, Any
import torch.distributed as dist


class OptimizerStateSharding(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        self.optimizer_cls = optimizer_cls
        self.optim = None
        self.param_to_rank = {}
        super().__init__(params, defaults=kwargs)


    def step(self, closure=None, **kwargs):
        self.optim.step(closure, **kwargs)
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    dist.broadcast(param.data, self.param_to_rank[param])


    def add_param_group(self, param_group: dict[str, Any]):
        # called by super().__init__(params, defaults=kwargs) internally
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        my_params = {'params': []}

        super().add_param_group(param_group)

        for idx, param in enumerate(param_group['params']):
            self.param_to_rank[param] = idx % world_size
            if idx % world_size == rank:
                my_params['params'].append(param)

        if not self.optim:
            self.optim = self.optimizer_cls([my_params], **self.defaults)
        else:
            self.optim.add_param_group(my_params)

