from typing import List, Dict, Any, Tuple, Callable, Union, Optional, Literal


import torch
from torch.utils import data
import numpy as np
from GGUtils.img.img import get_rgb_img
from GGUtils.img.viewer import show_img
from GGImgMorph.geometric.geometric import Resize
from ..utils import img_to_tensor, tensor_to_img




def show_dataset_img(dataset:data.Dataset, index:int):
    """
    입력된 dataset에서 index에 해당하는 img와 img의 shape을 출력한다.

    Args:
        dataset (data.Dataset): imgDataset으로 인해 생성된 Dataset
        index (int): imgDataset의 이미지 index
    """
    img = tensor_to_img(dataset[index][0])
    show_img(img, title=f"index: {index}, img size: {img.shape}")



class ImgDataset(data.Dataset):
    def __init__(
            self, 
            key_list:List[Dict[str, Union[str, np.ndarray]]],
            augments:Optional[Callable]=None,
            path_key:str='path', label_key:str='label',
            # resize 관련 매개변수
            resize:Optional[int]=None, 
            resize_quality:str='low',
            resize_how:int=0, 
            resize_how_list:List[int]=[2, 3, 4],
            resize_padding_color:str="random"
    ):
        """
        image에 대한 가장 기본적인 Dataset
        >>> key_list: 이미지 경로와 label의 list
            - [{'path':'이미지의_절대_경로_1', 'label':0}, {'path':'이미지의_절대_경로_2', 'label':0}, ...] 의 형태
            - GGDL 의 img_dict 기본 형식
            - 다른 key 값을 쓰는 경우, path_key와 label_key를 수정하여 반영 가능
        >>> augments: 이미지 증강 방법 메서드
            - 이미지만을 매개 변수로 입력받아 증강된 이미지를 출력하는 모든 메서드에 사용 가능
                ex) aug_img = fn(img)
            - GGImgMorph의 image 증강 메서드를 고려하여 개발
        >>> resize: 이미지 크기 조정(이미지를 정방 행렬로 조정, resize_how를 1로 설정하는 경우, 이미지 비율 유지)
            - resize는 기본적으로 from GGImgMorph.geometric.geometric import Resize의 Resize 클래스를 통해 동작
            - 'resize', 'resize_quality', 'resize_how', 'resize_padding_color' 4개의 파라미터는 이미지 resize에 대한 파라미터
            - reisze는 이미지의 크기로 입력하지 않는 경우, resize 하지 않음.
            - resize_how: 1(비율 유지), 2(padding), 3(border replicate), 4(이미지 가로, 세로 비율을 무시하고 resize)

        Args:
            key_list (List[Dict[str, Union[str, np.ndarray]]]): 각 이미지의 path, label 2가지가 dictionary 형태로 입력된 list
                - GGDL의 idx_dict 형태의 가장 최소 단위 데이터
            augments (Optional[Callable], optional): 이미지 증강 알고리즘 메서드. Defaults to None.
            path_key (str, optional): key_list의 각 이미지 레코드의 이미지 경로의 key. Defaults to 'path'.
            label_key (str, optional): key_list의 각 이미지 레코드의 이미지 label의 key. Defaults to 'label'.
            resize (Optional[int], optional): 이미지 resize 시, 이미지의 크기. Defaults to None.
            resize_quality (str): 이미지 resize 방식의 보간 방법. Defaults to 'low'.
                - 'low', 'middle', 'how'
            resize_how (int): resize 방법. Defaults to 0.
                - resize 방식
                - 1: 비율 유지, 이미지의 넓이, 높이를 비율 유지하여 resize한다(장방 행렬이 될 수 있음)
                - 2: 비율 유지 - padding, 1로 resize하고 color로 padding
                - 3: 비율 유지 - border replicate, 1로 resize하고 가장자리의 pixel로 채움
                - 4: 비율 무시, 이미지의 가로, 세로 길이를 size에 맞게 resize
            resize_how_list (List[int]): resize_how=0 인 경우, 무작위 선택되는 resize 방법. Defaults to [2, 3, 4].
            resize_padding_color (str): _description_. Defaults to "random".
                - resize_how = 2인 경우, 패딩 영역의 색
        """
        self.key_list = key_list
        self.path_key = path_key
        self.label_key = label_key
        self.augments = augments
        
        # resize가 None이 아닌 경우, GGImgMorph를 이용하여 이미지를 resize 한다.
        if resize is not None:
            self.resize = Resize(
                size=resize, quality=resize_quality, 
                how=resize_how, how_list=resize_how_list, 
                color=resize_padding_color
            )
        else:
            self.resize = None


    def __len__(self):
        return len(self.key_list)


    def __getitem__(self, idx):
        path = self.key_list[idx][self.path_key]
        img = get_rgb_img(path)

        if self.augments is not None:
            img = self.augments(img)

        if self.resize is not None:
            img = self.resize(img)

        # 이미지를 실수 Tensor로 변환
        img = img_to_tensor(img)

        label = self.key_list[idx][self.label_key]
        return img, label
    


class GetLoader:
    def __init__(
            self,             
            dataset_class:data.Dataset,
            idx_dict:Dict[str, str], 
            augments:Optional[Callable]=None, 
            batch_size:int=16, 
            workers:int=0,          # Data 증강 알고리즘이 복잡할 수록 0에 가까워야 성능이 더 잘나온다.
            **kwargs
        ):
        """
        data.Dataset을 data.DataLoader로 만들어서 train, test, valid 3개의 DataLoader를 불러온다.
            - valid 데이터가 없는 경우, validLoader는 None으로 출력된다.

        Args:
            dataset_class (data.Dataset): dataset을 가져올 class
            idx_dict (Dict[str, str]): 'train', 'test', 'valid'(없을 수 있음) 3개의 key로 구성된 dictionary로 다음과 같은 형태로 구성
                - Dict[str, List[Dict[str, str]]]
                - 예시) 'train':[{'path':'/usr/img1.jpg', 'label':0}, {'path':'/usr/img2.jpg', 'label':1}, ...]
            augments (Optional[Callable], optional): 이미지 증강 알고리즘. Defaults to None.
                - None이 아닌 경우, Train Dataset만 대상으로 적용된다.
            batch_size (int, optional): DataLoader의 Batchsize. Defaults to 16.
            workers (int, optional): 데이터 로딩을 위한 병렬 처리 작업자 수. Defaults to 4.
        """
        self.dataset_class = dataset_class
        self.idx_dict = idx_dict
        self.augments = augments
        self.batch_size = batch_size
        self.workers = workers
        self.kwargs = kwargs

        # dataset 정의
        self.train_set = None
        self.test_set = None
        self.valid_set = None
        self._get_datasets()
        
        # Loader 정의
        self.train = None
        self.test = None
        self.valid = None
        self._get_loader()


    def _get_datasets(self):
        """
        train, test, valid set을 생성하여 instance 변수에 추가
        """
        self.train_set = self.dataset_class(
            key_list = self.idx_dict['train'],
            augments=self.augments,
            **self.kwargs
        )
        self.test_set = self.dataset_class(
            key_list = self.idx_dict['test'],
            **self.kwargs
        )
        if 'valid' in self.idx_dict:
            self.valid_set = self.dataset_class(
                key_list = self.idx_dict['valid'],
                **self.kwargs
            )


    def _get_loader(self)->Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
        """_summary_

        Returns:
            Tuple[data.DataLoader, data.DataLoader, data.DataLoader]: _description_
        """
        self.train = self._get_one_loader(dataset=self.train_set, key="train")
        self.test = self._get_one_loader(dataset=self.test_set, key="test")
        self.valid = self._get_one_loader(dataset=self.valid_set, key="valid")
    

    def _get_one_loader(self, dataset:data.Dataset, key:str="train")->data.DataLoader:
        shuffle, drop_last = (True, True) if key == "train" else (False, False)
        if dataset is not None:
            loader = data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                pin_memory=torch.cuda.is_available(),
                num_workers=self.workers,
                shuffle=shuffle,
                drop_last=drop_last
            )
        else:
            loader = None
        return loader
