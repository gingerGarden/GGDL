from typing import Optional
import torch



class BackPropagation:
    def __init__(
        self, optimizer:torch.optim, 
        use_amp:bool=False, grad_scaler:Optional[torch.GradScaler]=None, 
        use_clipping:bool=False, max_norm:Optional[int]=None
        ):
        self.optimizer = optimizer
        self.use_amp = use_amp
        self.grad_scaler = grad_scaler
        self.use_clipping = use_clipping
        self.max_norm = max_norm
        
        self.params = [p for group in optimizer.param_groups for p in group['params']]      # optimizer의 파라미터 추출
        
        
    def __call__(self, loss:torch.Tensor):
        self.optimizer.zero_grad()      # gradient 0으로 초기화 - 이전 단계의 변화도가 누적되지 않도록 방지
        if self.use_amp:
            self._amp_backward(loss)
        else:
            self._normal_backward(loss)
        
    
    def _normal_backward(self, loss:torch.Tensor):
        loss.backward()
        if self.use_clipping:       # clipping 적용 여부
            self._gradint_clipping()
        self.optimizer.step()
        
        
    def _amp_backward(self, loss:torch.Tensor):
        # 손실 값을 스케일링하여 역전파 수행
        self.grad_scaler.scale(loss).backward()
        # clipping 적용 여부
        if self.use_clipping:
            self._gradint_clipping()
        # 스케일링된 그래디언트를 사용하여 옵티마이저의 매개변수 업데이트
        self.grad_scaler.step(self.optimizer)
        # 다음 학습 단계를 위한 스케일 팩터 업데이트
        self.grad_scaler.update()
        
            
    def _gradint_clipping(self):
        # AMP가 적용되어 있는 경우, GradScaler를 역산 후 적용하여, 수치 안정성고가 임계값 유지를 한다.
        if self.use_amp:
            self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(parameters=self.params, max_norm=self.max_norm)
        