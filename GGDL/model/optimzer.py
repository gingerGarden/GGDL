from typing import Dict, Optional, Union, Any, Callable, List

import torch_optimizer, torch

from torch import optim
from torch import nn

from GGUtils.utils.method import get_method_parameters


class Optim:
    def __init__(self, name:str, hp_dict:Dict[str, Any]):
        """
        name에 설정된 optimizer를 설정한다.
cd .
        대상 optimizer 와 torch, torch_optimizer 예제 코드
        1. SGD (Stochastic Gradient Descent): 확률적 경사 하강법으로, 각 배치마다 모델의 매개변수를 업데이트합니다. 학습률과 모멘텀 등의 하이퍼파라미터를 조정하여 성능을 향상시킬 수 있습니다.
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        2. Adam (Adaptive Moment Estimation): 1차 및 2차 모멘트 추정을 활용하여 학습률을 적응적으로 조정하는 알고리즘입니다. 대부분의 경우 빠른 수렴 속도와 안정적인 성능을 제공합니다.
            >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        3. Adagrad (Adaptive Gradient Algorithm): 각 매개변수에 대해 학습률을 개별적으로 조정하여 드물게 나타나는 특징(feature)에 대해 학습을 촉진합니다. 그러나 학습률이 너무 작아질 수 있는 단점이 있습니다.
            >>> optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
        4. RMSprop (Root Mean Square Propagation): Adagrad의 학습률 감소 문제를 해결하기 위해 지수 이동 평균을 사용하여 학습률을 조정합니다.
            >>> optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        5. AdamW: Adam 옵티마이저의 변형으로, 가중치 감쇠(weight decay)를 올바르게 처리하여 일반화 성능을 향상시킵니다.
            >>> optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        6. Adadelta: Adagrad의 학습률 감소 문제를 해결하기 위해 학습률을 동적으로 조정하는 알고리즘입니다.
            >>> optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
        7. Adamax: Adam의 변형으로, 무한 노름(infinity norm)을 기반으로 학습률을 조정합니다.
            >>> optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        8. ASGD (Averaged Stochastic Gradient Descent): SGD의 변형으로, 모델의 일반화 성능을 향상시키기 위해 매개변수의 이동 평균을 사용합니다.
            >>> optimizer = torch.optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
        9. LBFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno): 2차 미분 정보를 활용하는 최적화 알고리즘으로, 메모리 사용을 최소화하면서도 빠른 수렴을 제공합니다.
            >>> optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
        10. NAdam (Nesterov-accelerated Adaptive Moment Estimation): Adam과 Nesterov 모멘텀을 결합한 알고리즘으로, 빠른 수렴과 안정성을 제공합니다.
            >>> optimizer = torch.optim.NAdam(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004)
        11. RAdam (Rectified Adam): Adam의 학습률 워밍업 문제를 해결하여 안정적인 학습을 지원합니다.
            >>> optimizer = torch.optim.RAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, degenerated_to_sgd=True)
        12. Rprop (Resilient Backpropagation): 각 매개변수에 대해 개별적인 학습률을 적용하여 학습 속도를 향상시킵니다.
            >>> optimizer = torch.optim.Rprop(model.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
        13. SparseAdam: 희소 텐서에 특화된 Adam 옵티마이저로, 희소 데이터의 효율적인 학습에 사용됩니다.
            >>> optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        14. Adafactor: 메모리 효율적인 Adam의 변형으로, 큰 모델 학습 시 유용합니다.
            >>> optimizer = torch_optimizer .Adafactor(model.parameters(), lr=1e-3, eps=(1e-30, 1e-3), clip_threshold=1.0, decay_rate=-0.8, beta1=None, weight_decay=0.0, scale_parameter=True, relative_step=True, warmup_init=False)
        15. AdamP: Adam의 변형으로, 과도한 가중치 증가를 억제하여 일반화 성능을 향상시킵니다.
            >>> optimizer = torch_optimizer.AdamP(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2)
        16. DiffGrad: 최근의 기울기 변화에 따라 학습률을 조정하는 알고리즘입니다.
            >>> optimizer = torch_optimizer.DiffGrad(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        17. Lamb (Layer-wise Adaptive Moments optimizer for Batch training): 대규모 분산 학습에 적합한 옵티마이저로, 레이어별 학습률 조정을 지원합니다.
            >>> optimizer = torch_optimizer.Lamb(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
        18. NovoGrad: 메모리 사용량을 줄이면서도 빠른 수렴을 제공하는 옵티마이저입니다.
            >>> optimizer = torch_optimizer.NovoGrad(model.parameters(), lr=1e-3, betas=(0.95, 0.98), eps=1e-8, weight_decay=0)
        19. SWATS (Switching from Adam to SGD): 학습 초기에 Adam을 사용하고, 이후 SGD로 전환하는 알고리즘입니다.
            >>> optimizer = torch_optimizer.SWATS(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        20. AdaBound: Adam 옵티마이저의 변형으로, 학습 후반부에 학습률을 조정하여 더 안정적인 수렴을 유도합니다.
            >>> optimizer = torch_optimizer.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)
        21. Yogi: Adam의 변형으로, 학습 과정에서의 과도한 가중치 증가를 억제하여 안정적인 학습을 지원합니다.
            >>> optimizer = torch_optimizer.Yogi(model.parameters(), lr=1e-3)
        22. Ranger: RAdam과 Lookahead 옵티마이저를 결합한 알고리즘으로, 빠른 수렴과 일반화 성능 향상을 목표로 합니다.
            >>> optimizer = torch_optimizer.Ranger(model.parameters(), lr=1e-3)

        Args:
            name (str): optimizer의 이름
            hp_dict (Dict[str, Any]): optimizer에 입력될 paramter가 포함된 hyper parameter의 dictionary
                - **kwargs를 사용하므로 optimizer의 parameter key와 hp_dict의 key가 일치해야함. 중복된 key를 사용하면 안됨.
        """
        self.methods = {
            'SGD':optim.SGD,
            'Adam':optim.Adam,
            'Adagrad':optim.Adagrad,
            'RMSprop':optim.RMSprop,
            'AdamW':optim.AdamW,
            'Adadelta':optim.Adadelta,
            'Adamax':optim.Adamax,
            'ASGD':optim.ASGD,
            'LBFGS':optim.LBFGS,
            'NAdam':optim.NAdam,
            'RAdam':optim.RAdam,
            'Rprop':optim.Rprop,
            'SparseAdam':optim.SparseAdam,
            'Adafactor':torch_optimizer.Adafactor,
            'AdamP':torch_optimizer.AdamP,
            'DiffGrad':torch_optimizer.DiffGrad,
            'Lamb':torch_optimizer.Lamb,
            'NovoGrad':torch_optimizer.NovoGrad,
            'SWATS':torch_optimizer.SWATS,
            'AdaBound':torch_optimizer.AdaBound,
            'Yogi':torch_optimizer.Yogi,
            'Ranger':torch_optimizer.Ranger,
        }
        self.method = self._get_method(name)
        self.hp_dict = self._filter_hp_dict(hp_dict)


    def __call__(self, param:nn.Module)->optim:
        return self.method(param, **self.hp_dict)


    def _get_method(self, name:str)->optim:
        """
        name에 해당하는 optimizer를 선택함.

        Args:
            name (str): optimizer의 이름

        Raises:
            ValueError: name이 해당하지 않는 이름이 선택된 경우

        Returns:
            torch.optim: optimizer 객체
        """
        method_keys = list(self.methods.keys())
        if name not in method_keys:
            raise ValueError(f"입력한 name: {name}은 해당 class에서 지원하지 않습니다. optimizer는 {method_keys}만 지원합니다.")
        else:
            return self.methods.get(name)
        

    def _filter_hp_dict(self, hp_dict:Dict[str, Any])->Dict[str, Any]:
        """
        입력된 hp_dict 중, optimizer에 포함된 paramter만 선택

        Args:
            hp_dict (Dict[str, Any]): 모든 Hyper parameter의 dictionary

        Raises:
            ValueError: hp_dict에 optimizer에 해당하는 parameter가 하나도 없는 경우

        Returns:
            Dict[str, Any]: optimizer에 해당하는 paramter만 선택
        """
        # method의 parameter들을 가져온다.
        params = get_method_parameters(fn=self.method)

        # hp_dict에서 유효 매개변수만 필터링
        filterd_hp_dict = {k: v for k, v in hp_dict.items() if k in params}
        # hp_dict이 크기가 0인 경우 출력
        if len(filterd_hp_dict) == 0:
            raise ValueError(f"입력된 hp_dict의 key가 사용하고자 하는 optimizer의 parameter와 일치하지 않습니다. 선택한 optimizer의 parameter는 {list(params)}와 같습니다.")
        else:
            return filterd_hp_dict



class LabelDtype:
    
    LONG_LOSS_FN = (nn.CrossEntropyLoss)
    FLOAT32_LOSS_FN = (
        nn.MSELoss, nn.L1Loss, nn.BCELoss, nn.BCEWithLogitsLoss, nn.KLDivLoss, nn.HuberLoss,
        nn.MarginRankingLoss, nn.CosineEmbeddingLoss, nn.HingeEmbeddingLoss
    )

    def __init__(self, loss_fn:nn.modules.loss=None, dtype:Optional[torch.dtype]=None):

        if dtype is None:
            self.dtype = self._check_by_loss_fn(loss_fn)
        else:
            self.dtype = dtype

        
    def __call__(self, labels):
        return labels.to(self.dtype)


    def _check_by_loss_fn(self, loss_fn):
        if isinstance(loss_fn, self.LONG_LOSS_FN):
            return torch.long
        elif isinstance(loss_fn, self.FLOAT32_LOSS_FN):
            return torch.float32
        else:
            raise ValueError(f"Unsupported loss function: {type(loss_fn).__name__}")