import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

from typing import Callable, Tuple
import torch
import torch.nn.functional as F
import eagerpy as ep
from foolbox.models.base import Model


from foolbox.attacks import L2FastGradientAttack
# fixed epsilons attack
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

from foolbox.attacks import L2AdditiveGaussianNoiseAttack

def get_fixed_eps_attacks(loss):
    if loss == 'standard_ce':
        l2_fea = {}
        l2_fea["Gaussian Noise"] = L2AdditiveGaussianNoiseAttack()
        l2_fea["FGM"] = FGM_Conf()
        l2_fea["L2 BIM"] = L2BIM_Conf()
        linf_fea = {}
        linf_fea["Linf BIM"] =LinfBIM_Conf()
        linf_fea["FGSM"] = FGSM_Conf()
        return l2_fea, linf_fea
    if loss == 'pytorch_ce':
        l2_fea = {}
        l2_fea["Gaussian Noise"] = L2AdditiveGaussianNoiseAttack()
        l2_fea["FGM"] = LinfFastGradientAttack()
        l2_fea["L2 BIM"] = L2BasicIterativeAttack()
        linf_fea = {}
        linf_fea["Linf BIM"] = LinfBasicIterativeAttack()
        linf_fea["FGSM"] = LinfFastGradientAttack()
        return l2_fea, linf_fea
    
# minimization attack
from foolbox.attacks import BoundaryAttack, L2DeepFoolAttack, LinfDeepFoolAttack

class L2DeepFoolAttack_Conf(L2DeepFoolAttack):
    def _get_loss_fn(
        self, model: Model, classes: ep.Tensor,
    ) -> Callable[[ep.Tensor, int], Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]]:

        N = len(classes)
        rows = range(N)
        i0 = classes[:, 0]

        if self.loss == "logits":

            def loss_fun(
                x: ep.Tensor, k: int
            ) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
                logits = model(x)
                ik = classes[:, k]
                l0 = logits[rows, i0]
                lk = logits[rows, ik]
                loss = lk - l0
                return loss.sum(), (loss, logits)

        elif self.loss == "crossentropy":

            def loss_fun(
                x: ep.Tensor, k: int
            ) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
                logits = model(x)
                ik = classes[:, k]
                l0 = -ep.crossentropy(logits, i0)
                lk = -ep.crossentropy(logits, ik)
                loss = lk - l0
                return loss.sum(), (loss, logits)
            
        elif self.loss == "standard_ce":
            
            def loss_fun(
                x: ep.Tensor, k: int
            ) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
                logits = model(x)
                ik = classes[:, k]
                # l0 = -ep.crossentropy(logits, i0)
                l0 = type(x)(
                    -F.nll_loss(torch.log(logits.raw), i0.raw, reduction="none"))
                # lk = -ep.crossentropy(logits, ik)
                lk = type(x)(
                    -F.nll_loss(torch.log(logits.raw), ik.raw, reduction="none"))
                loss = lk - l0
                return loss.sum(), (loss, logits)
            
        else:
            raise ValueError(
                f"expected loss to be 'logits' or 'crossentropy', got '{self.loss}'"
            )

        return loss_fun

class LinfDeepFoolAttack_Conf(LinfDeepFoolAttack):
    def _get_loss_fn(
        self, model: Model, classes: ep.Tensor,
    ) -> Callable[[ep.Tensor, int], Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]]:

        N = len(classes)
        rows = range(N)
        i0 = classes[:, 0]

        if self.loss == "logits":

            def loss_fun(
                x: ep.Tensor, k: int
            ) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
                logits = model(x)
                ik = classes[:, k]
                l0 = logits[rows, i0]
                lk = logits[rows, ik]
                loss = lk - l0
                return loss.sum(), (loss, logits)

        elif self.loss == "crossentropy":

            def loss_fun(
                x: ep.Tensor, k: int
            ) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
                logits = model(x)
                ik = classes[:, k]
                l0 = -ep.crossentropy(logits, i0)
                lk = -ep.crossentropy(logits, ik)
                loss = lk - l0
                return loss.sum(), (loss, logits)
            
        elif self.loss == "standard_ce":
            
            def loss_fun(
                x: ep.Tensor, k: int
            ) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
                logits = model(x)
                ik = classes[:, k]
                # l0 = -ep.crossentropy(logits, i0)
                l0 = type(x)(
                    -F.nll_loss(torch.log(logits.raw), i0.raw, reduction="none"))
                # lk = -ep.crossentropy(logits, ik)
                lk = type(x)(
                    -F.nll_loss(torch.log(logits.raw), ik.raw, reduction="none"))
                loss = lk - l0
                return loss.sum(), (loss, logits)
            
        else:
            raise ValueError(
                f"expected loss to be 'logits' or 'crossentropy', got '{self.loss}'"
            )

        return loss_fun

from foolbox.attacks import SaltAndPepperNoiseAttack

def get_min_attacks(loss):
    if loss == 'standard_ce':
        l2_ma = {}
        # l2_ma['Boundary Attack'] = BoundaryAttack()
        l2_ma['L2 DeepFool'] = L2DeepFoolAttack_Conf(loss = 'standard_ce')
        linf_ma = {}
        linf_ma['Linf DeepFool'] = LinfDeepFoolAttack_Conf(loss = 'standard_ce')
        l0_ma = {}
        l0_ma['Salt&Pepper Noise'] = SaltAndPepperNoiseAttack()
    if loss == 'pytorch_ce':
        l2_ma = {}
        # l2_ma['Boundary Attack'] = BoundaryAttack()
        l2_ma['L2 DeepFool'] = L2DeepFoolAttack(loss = 'crossentropy')
        linf_ma = {}
        linf_ma['Linf DeepFool'] = LinfDeepFoolAttack(loss = 'crossentropy')
        l0_ma = {}
        l0_ma['Salt&Pepper Noise'] = SaltAndPepperNoiseAttack()
    return l2_ma, linf_ma, l0_ma