# Fine tuning 전략
# 1. Full Fine-tuning(전체 모델 Fine-tuning)
# 방법: 사전 학습된 모델의 모든 파라미터를 학습 가능 상태로 설정하고, 새로운 데이터셋에서 학습 진행
# 장점: 모델의 모든 파라미터를 새롭게 업데이트하여 데이터셋에 최적화할 수 있음.
# 단점: 계산 비용이 크고, 작은 데이터셋에서는 과적합의 위험이 큼.

# 2. Fixed Feature Extractor(고정된 Feature 추출기)
# 방법: 사전 학습된 모델의 대부분 레이어를 고정시키고, 마지막 분류기 레이어만 학습.
# 장점: 계산 비용이 낮으며, 빠르게 새로운 데이터셋에 맞출 수 있음.
# 단점: 사전 학습된 Feature에만 의존하게 되어, 만약 새로운 데이터셋의 Feature가 크게 다르면 성능이 저하될 수 있음.

# 3. Layer-wise Unfreezing(선택적 레이어 Unfreeze)
# 방법: 먼저 마지막 분류기 레이어만 학습하고, 이후 특정 레이어를 점진적으로 해제하며 모델 전체를 학습합니다. 일반적으로 하위 레이어보다는 상위 레이어를 먼저 학습합니다.
# 장점: 모델의 중요한 저차원 특징을 유지하며, 상위 레이어의 고수준 특징을 새롭게 학습할 수 있습니다.
# 단점: 조정이 잘못되면 학습이 불안정해질 수 있습니다.

# 4. Differential Learning Rate (차등 학습률)
# 방법: 모델의 각 레이어에 대해 다른 학습률을 설정합니다. 일반적으로 상위 레이어에는 더 높은 학습률을, 하위 레이어에는 낮은 학습률을 적용합니다.
# 장점: 사전 학습된 저수준 특징은 크게 수정하지 않으면서 고수준 특징을 효율적으로 조정할 수 있습니다.
# 단점: 각 레이어에 맞는 학습률을 설정하는 것이 복잡할 수 있습니다.

# 5. Learning Rate Scheduler (학습률 스케줄러)
# 방법: 학습 과정에서 학습률을 점차 줄여가며 모델이 더 안정적으로 수렴하도록 합니다. StepLR, CosineAnnealingLR와 같은 스케줄러가 대표적입니다.
# 장점: 학습 과정 동안 성능을 높이는 데 유리하며, 최종 수렴 시 과적합을 방지할 수 있습니다.
# 단점: 학습 초기의 학습률 설정에 따라 학습이 너무 느려질 수도 있습니다.

# 6. Transfer Learning with Pre-training on Domain-specific Data (도메인 맞춤형 전이 학습)
# 방법: 먼저 도메인에 적합한 데이터를 사용하여 추가적인 사전 학습을 진행한 후, 특정 작업에 맞게 Fine-tuning을 수행합니다.
# 장점: 모델이 일반적인 사전 학습된 지식을 도메인에 맞춰 조정한 후, 특정 작업에 대한 Fine-tuning을 하므로 성능이 높아질 가능성이 큽니다.
# 단점: 사전 학습 단계가 추가되어 시간이 더 소요됩니다.

# 7. Freeze and Unfreeze Strategy (고정과 해제 전략)
# 방법: 초기에는 대부분의 레이어를 고정시키고 분류기만 학습하다가, 점진적으로 레이어를 해제하면서 학습을 진행합니다. 이 방법은 특히 학습 초기 불안정함을 줄이기 위해 유용합니다.
# 장점: 학습이 안정적이며, 특정 단계마다 모델의 모든 부분을 최적화할 수 있습니다.
# 단점: 구현 복잡도가 높고, 반복적인 해제와 학습이 필요할 수 있습니다.

# 8. Task-specific Layer Addition (작업별 추가 레이어)
# 방법: 사전 학습된 모델에 몇 개의 새로운 레이어를 추가하여 특정 작업에 맞게 학습합니다. 일반적으로 새로운 레이어는 랜덤 초기화되고, 전체 모델을 함께 학습합니다.
# 장점: 특정 작업에 맞는 맞춤형 학습이 가능하며, 사전 학습된 특징을 효과적으로 활용할 수 있습니다.
# 단점: 새로 추가된 레이어가 복잡한 경우, 과적합의 위험이 있습니다.

# 9. Layer-wise Adaptive Learning Rates (레이어별 적응 학습률)
# 방법: 모델의 각 레이어에 대해 학습 중 성능에 따라 동적으로 학습률을 조정합니다.
# 장점: 모델이 특정 레이어에 대해 학습을 더 필요로 하는 경우 이를 반영할 수 있습니다.
# 단점: 학습률 조정 로직이 복잡하며, 계산 비용이 증가할 수 있습니다.

# 10. Regularization Techniques (정규화 기법)
# 방법: L2 정규화, Dropout 등을 적용하여 과적합을 방지하며 모델의 일반화 성능을 높일 수 있습니다.
# 장점: Fine-tuning 시 모델이 과적합되지 않도록 돕고, 일반화 성능을 높여줍니다.
# 단점: 정규화 파라미터를 잘못 설정하면 학습이 어려워질 수 있습니다.
from typing import Dict, Optional, Union
import re, torch



class ParameterIndex:
    
    def __init__(
            self, 
            model:torch.nn.Module, 
            weight_ptn:Optional[str]=None, 
            bias_ptn:Optional[str]=None,
            block_ptn:Optional[str]=None):
        """
        torch 기반 model의 model.named_parameters()를 기반으로 하여, parameter의 layer, block을 구분하는 index_dictionary를 생성한다.

        model의 parameter에 대하여 다음과 같은 방법을 통해 주요 metadata를 추출하여 index를 생성한다.
            1. paramter로부터 기본적인 정보를 가져온다.
            2. parameter를 weight, bias 존재 여부 확인
            3. parameter 순서와 weight, bias를 제외한 name이 이전과 동일한지 여부로 layer 구분
            4. 이전 layer와 현재 layer가 다르면서, self.block_ptn이 다른 경우, 다른 block으로 정의한다.

        Args:
            model (torch.nn.Module): torch 기반 모델
            weight_ptn (Optional[str], optional): 모델 parameter의 name에 대해서 weight의 정규식 패턴. Defaults to None.
                - None인 경우, torch model의 기본 형태인 `r'\\.weight$'`를 사용한다.
            bias_ptn (Optional[str], optional): 모델 parameter의 name에 대해서 bias의 정규식 패턴. Defaults to None.
                - None인 경우, torch model의 기본 형태인 `r'\\.bias$'`를 사용한다.
            block_ptn (Optional[str], optional): 모델 parameter의 name에 대해서 block의 정규식 패턴. Defaults to None.
                - None인 경우, torch model의 기본 형태인 `r'\\.\\d+\\.'`를 사용한다.
        """
        self.model = model
        self.weight_ptn = r"\.weight$" if weight_ptn is None else weight_ptn
        self.bias_ptn = r"\.bias$" if bias_ptn is None else bias_ptn
        self.block_ptn = r"\.\d+\." if block_ptn is None else block_ptn
        self.source = None


    def process(self)->Dict[int, Dict[str, Union[str, bool, int]]]:
        """
        실행 메서드
        """
        # parameter로부터 가장 기본적인 정보를 가져온다.
        self._make_source_dict()
        # parameter를 weight, bias로 구분
        self._add_weight_or_bias()
        # weight, bias를 기준으로 layer 구분
        self._add_layer()
        # layer, 정규식을 이용하여 block 구분
        self._add_block()
        return self.source


    def _make_source_dict(self):
        """
        parameter를 mapping하는 재료가 되는 dictionary 생성
        """
        stack_dict = dict()
        for i, (name, param) in enumerate(self.model.named_parameters()):
            stack_dict[i] = {
                "name":name,
                "grad":param.requires_grad,
            }
        self.source = stack_dict
        

    def _add_weight_or_bias(self):
        """
        name을 기준으로 weight, bias를 나눈다.
        """
        re_weight = re.compile(self.weight_ptn)
        re_bias = re.compile(self.bias_ptn)

        for _, value in self.source.items():
            name = value['name']
            value['weight'] = True if re_weight.search(name) else False
            value['bias'] = True if re_bias.search(name) else False


    def _add_layer(self):
        """
        weight, bias를 제외한 name이 이전과 동일하면 동일한 layer로 취급한다.
        """
        i = -1      # 0부터 시작하도록 함.
        before_id = ''
        for _, value in self.source.items():
            if value['weight']:
                current_id = re.sub(self.weight_ptn, '', value['name'])
            elif value['bias']:
                current_id = re.sub(self.bias_ptn, '', value['name'])
            else:
                current_id = value['name']
            
            # before_id랑 current_id가 다르면 i에 1을 더한다.
            if before_id != current_id:
                i += 1
                before_id = current_id
            value['layer'] = i


    def _add_block(self):
        """
        이전 layer와 현재 layer가 다르면서, self.block_ptn이 다른 경우, 다른 block으로 정의한다.
        """
        before_block = None
        before_layer = None
        i = -1
        regex = re.compile(pattern=self.block_ptn)
        for _, value in self.source.items():
            
            # layer가 다를 때만 정규식 기반 block 확인을 한다.
            current_layer = value['layer']
            if current_layer != before_layer:
                
                # 정규식 기반 block 확인
                searched = regex.search(string=value['name'])
                if searched:
                    current_block = searched.group()
                    if current_block != before_block:
                        before_block = current_block
                        i += 1
                else:
                    i += 1
            
            before_layer = current_layer    # before_layer 갱신
            value['block'] = i