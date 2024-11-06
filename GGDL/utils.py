from typing import List, Dict, Any, Tuple, Callable, Union, Optional

import random, os, warnings, torch
import numpy as np
from torch.utils import data

from GGUtils.utils.path import new_dir_maker


def make_basic_directory(source:str='source', estop_dir:str='estop_point', log:str='log', result:str='result', make_new:bool=False):
    """
    GGDL을 사용하여 딥러닝 학습 시, 기본적인 디렉터리들을 생성한다.

    Args:
        source (str, optional): 학습의 기반이 되는 데이터들이 저장되는 디렉터리. Defaults to 'source'.
        log (str, optional): 학습 도중 발생하는 log들이 저장되는 디렉터리. Defaults to 'log'.
        result (str, optional): 학습 결과가 저장되는 디렉터리. Defaults to 'result'.
        make_new (bool, optional): 디렉터리들을 새로 생성할지 여부. Defaults to False.
    """
    early_stopping = f'{source}/{estop_dir}'
    new_dir_maker(source, makes_new=make_new)
    new_dir_maker(log, makes_new=make_new)
    new_dir_maker(result, makes_new=make_new)
    new_dir_maker(early_stopping, makes_new=make_new)


# set seed
def set_seed_everything(seed:int=1234, deterministic:bool=True, benchmakr:bool=False):
    """torch 학습 시, 무작위로 설정되는 난수들을 고정하여, 재학습 시 다른 결과가 출력되지 않도록 한다.

    * 적용 알고리즘
        1. ramdom 시드 고정
        2. numpy 시드 고정
        3. torch의 cpu 난수 생성기 시드 고정
        4. 모든 cuda 장치의 난수 생성기 시드 고정
        5. python의 hash 함수 시드 고정 - hash 기반 연산이 일관되게 함
        6. CuDNN 사용 시, 결정론적(Deterministic) 연산 수행 - 동일한 입력과 조건에서 항상 동일한 결과 출력
            - 결과 재현성에 유리함.
            - 성능이 저하될 수 있음(최적화 이슈).
        7. CuDNN의 연산 속도 최적화와 관련된 것으로, CuDBB에서 사용할 최적의 커널(알고리즘)을 자동으로 선택하도록 하는 설정
            - GPU에서 합성곱 연산 등 특정 연상 수행 시, 다양한 커널 중 성능이 가장 좋은 커널을 선택하여 속도 최적화 함.
            - 설정 시, 성능이 향상될 수 있음(입력 데이터의 크기가 고정되어 있는 경우)
            - 입력 데이터의 크기가 자주 바뀌는 경우, 매번 최적의 커널을 찾아야하므로 오히려 성능이 저하될 수 있음.
            - 결정론적 동작 방해

    Args:
        seed (int, optional): seed 값. Defaults to 1234.
        deterministic (bool, optional): 결정론적 연산 수행 여부. Defaults to True.
        benchmakr (bool, optional): CuDNN 밴치마킹 기능 비활성화 여부. Defaults to False.
    """
    random.seed(seed)                 # random 모듈 시드 고정
    np.random.seed(seed)              # numpy 라이브러리 난수 생성 시드 고정
    torch.manual_seed(seed)           # torch의 cpu 난수 생성기 시드 고정
    torch.cuda.manual_seed_all(seed)  # 모든 cuda 장치의 torch 난수 생성기 시드 고정
    os.environ['PYTHONHASHSEED'] = str(seed)  # python hash 함수 시드를 설정해 해시 기반 연산이 일관되게 수행되게 함.

    # CuDNN 사용 시, 결정론적 연산을 수행하도록 설정 - 재현성(O), 성능에 영향 줄 수 있음.
    torch.backends.cudnn.deterministic = deterministic
    # CuDNN 벤치마킹 기능 비활성화 - 동일 입력 크기에 대해 항상 동일 알고리즘을 사용하도록 함 - 재현성(O) 성능에 부정적일 수 있음.
    torch.backends.cudnn.benchmark = benchmakr


def img_to_tensor(img:np.ndarray, to_float:bool=True, max_pixel:float=255, dtype:torch.dtype=torch.float32)->torch.Tensor:
    """
    입력된 이미지를 Torch의 tensor 형식으로 변환
    >>> 일반 이미지: (Y, X, C)
        Torch 이미지: (C, Y, X)

    Args:
        img (np.ndarray): 원본 이미지
        to_float (bool, optional): 0에서 1사이의 실수로 변환 여부. Defaults to True.
        max_pixel (float, optional): 실수 변환을 위한 최대 픽셀 값. Defaults to 255.
            - RGB 또는 RGBA를 기본 이미지로 하므로, 기본값으로 255로 함.
        dtype (torch.dtype, optional): 실수의 dtype. Defaults to torch.float32.

    Returns:
        torch.Tensor: Torch 스타일로 변환된 이미지의 텐서
    """
    if to_float:
        img = img/max_pixel
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)  # x, y축만 갖는 이미지(흑백 이미지)인 경우, 1차원의 채널을 추가함.
    # array를 tensor로 변환
    torch_img = torch.as_tensor(img, dtype=dtype)
    return torch_img.permute(2, 0, 1)      # (C, Y, X) 형식으로 변환


def tensor_to_img(tensor_img:torch.Tensor, to_int:bool=True, max_pixel:int=255, dtype=np.uint8)->np.ndarray:
    """
    입력된 이미지 tensor를 numpy 배열 이미지로 변환
    >>> to_int가 True인 경우, max_pixel로 곱하여, 정수 이미지로 변환.

    Args:
        tensor_img (torch.Tensor): torch.Tensor type의 이미지.
        to_int (bool, optional): int로 변환 여부. Defaults to True.
        max_pixel (int, optional): int로 변환 시 곱하는 값. Defaults to 255.
        dtype (_type_, optional): int로 변환 된 이미지의 dtype. Defaults to np.uint8.

    Returns:
        np.ndarray: _description_
    """
    assert isinstance(tensor_img, torch.Tensor), "Input must be a torch.Tensor"

    if tensor_img.device != torch.device('cpu'):
        tensor_img = tensor_img.detach().cpu()
    
    np_img = tensor_img.numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    if to_int:
        return (np_img * max_pixel).astype(dtype)
    return np_img


def parameter_state_dict_to_numpy(state_dict:Dict[str, torch.Tensor])->Dict[str, np.ndarray]:
    """
    torch의 parameter에 대한 state_dict을 numpy array로 이루어진 dictionary로 변환한다.
    >>> 학습 중 해당 코드가 동작할 수 있으므로, detach하지 않음.

    Args:
        state_dict (Dict[torch.Tensor]): PyTorch 모델의 state_dict 파라미터

    Returns:
        Dict[np.ndarray]: numpy array로 변환된 state_dict 파라미터
    """
    # 학습 중 해당 코드가 동작하는 경우를 대비하여 detach하지 않음.
    return {key: parameters.cpu().numpy() for key, parameters in state_dict.items()}


def imgs_upload_to_device(imgs:List[torch.Tensor], device:torch.device)->List[torch.Tensor]:
    """
    torch.Tensor로 변환된 이미지들이 들어가 있는 List에 대하여, List 내 이미지들을 device에 업로드

    Args:
        imgs (List[torch.Tensor]): torch.Tensor로 변환된 이미지들이 들어가 있는 list
        device (str): img들을 업로드할 device명

    Returns:
        List[torch.Tensor]: device에 업로드 된 이미지들의 List
    """
    return [img.to(device) for img in imgs]


def targets_upload_to_device(targets:List[Dict[str, torch.Tensor]], device:torch.device)->List[Dict[str, torch.Tensor]]:
    """
    bbox(Bounding box)의 target이 torch.Tensor인 경우, device에 업로드한다.
    >>> torch.Tensor가 아닌 값들은 device에 업로드 하지 않으므로, bbox의 좌표만 디바이스 업로드 가능

    Args:
        targets (List[Dict[str, torch.Tensor]]): bbox가 포함되어 있는 target들의 list
        device (torch.device): 학습에 사용될 device

    Returns:
        List[Dict[str, torch.Tensor]]: bbox가 device에 업로드 된 list
    """
    return [{key:value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()} for target in targets]


class GetDevice:
    def __init__(self):
        """
        GetDevice class의 초기화 메서드
        GPU의 사용 가능 여부와 사용 가능한 GPU 장치의 개수를 확인하여, 각각 gpu_bool과 device_count에 저장
        """
        self.device_count = torch.cuda.device_count()
        self.gpu_bool = torch.cuda.is_available()
        
        
    def __call__(self, gpu_number:int=0)->str:
        """
        학습에 사용될 device 정의
        >>> gpu 사용이 불가능한 경우 cpu로 가지고 온다.
        >>> gpu_number가 device의 범위를 벗어나는 경우, 경고 문구를 출력하고 cuda:0으로 자동 설정한다.
        
        Args:
            gpu_number (int): gpu의 번호

        Returns:
            string: 'cuda:0'과 같은 torch cuda device
        """
        if self.gpu_bool:
            device = self._check_gpu_number_in_have(gpu_number)
        else:
            device = 'cpu'
            script = 'GPU usage is not available. Fetching devices with CPU.'
            warnings.warn(script, UserWarning)
        return  device
            
        
    def _check_gpu_number_in_have(self, gpu_number:int)->str:
        gpu_size = self.device_count - 1
        if gpu_number > gpu_size:
            script=f"""
            Warning! You have {self.device_count} GPUs, and the maximum selectable number is {gpu_size}!
            Since it's not a selectable GPU number, returning the default GPU 0 as the device.
            """
            warnings.warn(script, UserWarning)
            return 'cuda:0'
        
        return f'cuda:{gpu_number}'
        
        
    def summary(self):
        """
        현재 환경에서 device의 인식이 어떻게 되어있는지 요약.
        """
        if self.gpu_bool:
            print("CUDA is available.")
            print(f"GPU size: {self.device_count}")
            print("---"*20)
            for i in range(self.device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpue_cc = torch.cuda.get_device_capability(i)
                vram = round(torch.cuda.get_device_properties(i).total_memory/(1024**3))
                print(f"GPU number: {i}")
                print(f"Name: {gpu_name}")
                print(f"Computer capability: {gpue_cc[0]}.{gpue_cc[1]}")
                print(f"VRAM: {vram}GB")
                print("---"*20)
        else:
            print("CUDA is not available!")
            
            
            
class Option:
    def __init__(
        self, 
        
        process_id:int,
        
        model_name:str, pretrained:bool, device:str, dataset_class:data.Dataset,
        
        idx_dict:Dict[int, Dict[str, List[Dict[str, int]]]],
        img_channel:int=3, class_size:Optional[int]=None, custom_header:torch.nn.Module=None,
        
        img_size:int=224, resize_how:int=0, resize_how_list:List[int]=[2,3,4], resize_padding_color:str='black',
        augments:Optional[Callable]=None, batch_size:int=16, worker:int=0, 
        
        tuner_how:int=0
        ):
        
        self.process_id = process_id
        
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.idx_dict = idx_dict
        
        self.img_channel = img_channel
        self.class_size = class_size
        self.custom_header = custom_header
        
        self.dataset_class = dataset_class
        self.augments = augments
        self.img_size = img_size
        self.batch_size = batch_size
        self.worker = worker
        self.resize_how = resize_how
        self.resize_how_list = resize_how_list
        self.resize_padding_color = resize_padding_color

        self.tuner_how = tuner_how