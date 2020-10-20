import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

from typing import Callable
import torch
import torch.nn.functional as F
import eagerpy as ep
from foolbox.models.base import Model


from foolbox.attacks import L2FastGradientAttack
class FGM_Conf(L2FastGradientAttack):
    def get_loss_fn(
        self, model: Model, labels: ep.Tensor
    ) -> Callable[[ep.Tensor], ep.Tensor]:
        # can be overridden by users
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            return type(inputs)(
                F.nll_loss(torch.log(logits.raw), labels.raw, reduction="none")
                ).sum()
            # return ep.crossentropy(logits, labels).sum()
        return loss_fn
    
    
from foolbox.attacks import L2BasicIterativeAttack
class L2BIM_Conf(L2BasicIterativeAttack):
    def get_loss_fn(
        self, model: Model, labels: ep.Tensor
    ) -> Callable[[ep.Tensor], ep.Tensor]:
        # can be overridden by users
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            return type(inputs)(
                F.nll_loss(torch.log(logits.raw), labels.raw, reduction="none")
                ).sum()
            # return ep.crossentropy(logits, labels).sum()
        return loss_fn
    
from foolbox.attacks import LinfBasicIterativeAttack
class LinfBIM_Conf(LinfBasicIterativeAttack):
    def get_loss_fn(
        self, model: Model, labels: ep.Tensor
    ) -> Callable[[ep.Tensor], ep.Tensor]:
        # can be overridden by users
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            return type(inputs)(
                F.nll_loss(torch.log(logits.raw), labels.raw, reduction="none")
                ).sum()
            # return ep.crossentropy(logits, labels).sum()
        return loss_fn

from foolbox.attacks import LinfFastGradientAttack
class FGSM_Conf(LinfFastGradientAttack):
    def get_loss_fn(
        self, model: Model, labels: ep.Tensor
    ) -> Callable[[ep.Tensor], ep.Tensor]:
        # can be overridden by users
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            return type(inputs)(
                F.nll_loss(torch.log(logits.raw), labels.raw, reduction="none")
                ).sum()
            # return ep.crossentropy(logits, labels).sum()
        return loss_fn


