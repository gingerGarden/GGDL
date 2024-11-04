from typing import List, Dict, Any, Tuple, Callable, Union, Optional
import numpy as np
import pandas as pd

from ..idx_dict.key_df import StratifiedSampling, KeyDF2KeyDict


_seed_list = [
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 1111,
    1101, 2101, 3101, 4101, 5101, 6101, 7101, 8101, 9101, 2222,
    1202, 2202, 3202, 4202, 5202, 6202, 7202, 8202, 9202, 3333,
    1303, 2303, 3303, 4303, 5303, 6303, 7303, 8303, 9303, 4444,
    1404, 2404, 3404, 4404, 5404, 6404, 7404, 8404, 9404, 5555,
    1504, 2504, 3504, 4504, 5504, 6504, 7504, 8504, 9504, 6666,
    1604, 2604, 3604, 4604, 5604, 6604, 7604, 8604, 9604, 7777,
    1704, 2704, 3704, 4704, 5704, 6704, 7704, 8704, 9704, 8888,
    1804, 2804, 3804, 4804, 5804, 6804, 7804, 8804, 9804, 9999,
    1904, 2904, 3904, 4904, 5904, 6904, 7904, 8904, 9904, 9990
    ]


def make_stratified_idx_dict(
        key_df:pd.DataFrame, stratified_columns:List[str], is_binary:bool,
        path_col:str, label_col:str, onehot_key:str='dummy',
        rounding:Callable=np.ceil, k_size:int=1, seed_list:List[int]=_seed_list,
        test_ratio:float=0.3, valid_ratio:Optional[float]=None
    )->Dict[int, Dict[str, List[Dict[str, Union[str, np.ndarray]]]]]:
    """
    Stratified k-fold cross validation:
    stratified_columns의 비율을 고려한 층화 표본 추출을 하여, train, test, validation(valid_ratio is not None) set을 나눈다.
    >>> seed_list: .make_dict._seed_list를 기본 seed로 설정하며, 유저가 원하는 seed_list 입력이 가능
        - seed_list의 크기는 반드시 k_size의 크기보다 크거나 같아야 하며, 만약 작은 경우 ValueError 발생
    >>> validation set: 전체 데이터에서 test set을 분할하고, 나머지 데이터에서 validation set을 분할
        - None인 경우, validation set을 분할하지 않음
    >>> onehot_key: 다중 분류 시, 새로 생성되는 컬럼에 추가되는 텍스트로 f"{column}_{onehot_key}_{class}"로 컬럼이 생성된다.
        - 기존 컬럼과 중복되는 경우, ValueError가 발생한다.

    Args:
        key_df (pd.DataFrame): path, label이 포함된 분할 대상이 되는 데이터 프레임
        stratified_columns (List[str]): 층화 표본 추출의 기준이 되는 column들의 list
        is_binary (bool): label이 이진 분류인지 여부
        path_col (str): 이미지 데이터의 경로가 들어 있는 key_df의 컬럼명
        label_col (str): label이 들어 있는 key_df의 컬럼명
        onehot_key (str, optional): 다중분류를 하는 경우 새로 생성되는 컬럼에 추가되는 텍스트. Defaults to 'dummy'.
        rounding (Callable, optional): train, test, validation set을 비율로 나눌 때, 크기의 반올림 방식. Defaults to np.ceil.
        k_size (int, optional): 몇 개의 idx_dict을 생성할지(k-fold stratified cross validation). Defaults to 1.
        seed_list (List[int], optional): seed들이 들어가 있는 list. Defaults to _seed_list.
        test_ratio (float, optional): test set의 비율. Defaults to 0.3.
        valid_ratio (Optional[float], optional): validation set의 비율(None인 경우, validation set은 생성되지 않는다.). Defaults to None.

    Returns:
        Dict[int, Dict[str, List[Dict[set, Union[str, np.ndarray]]]]]: k_size만큼의 Stratified validation 방식으로 분할된 key_dict이 들어가 있는 딕셔너리
    """
    # seed_list의 크기가 k_size보다 작은 경우, 경고 발생
    _k_size_and_seed_list_error_in_make_stratified_idx_dict(k_size, seed_list)

    result = dict()
    # 주요 instance 생성
    SAMPLINT_ins = StratifiedSampling(columns=stratified_columns, rounding=rounding)
    CONVERT_ins = KeyDF2KeyDict(path_col=path_col, label_col=label_col, is_binary=is_binary, onehot_key=onehot_key)
    # 데이터 분할
    for i, seed in enumerate(seed_list[:k_size]):
        result[i] = _sub_set_for_make_stratified_idx_dict(
            key_df=key_df, seed=seed, 
            test_ratio=test_ratio, valid_ratio=valid_ratio, 
            SAMPLINT_ins=SAMPLINT_ins, CONVERT_ins=CONVERT_ins
        )
    return result


def _sub_set_for_make_stratified_idx_dict(
        key_df:pd.DataFrame, seed:int,
        test_ratio:float, valid_ratio:Optional[float], 
        SAMPLINT_ins:Callable, CONVERT_ins:Callable
    )->Dict[str, List[Dict[set, Union[str, np.ndarray]]]]:
    """
    make_stratified_idx_dict의 하위 함수로, 'train', 'test', 'valid'(valid_ratio is not None)의 3(2)개의 key로 구성된 하위 idx_dict 생성
    """
    # train, valid set 분할
    test_key_df, rest_key_df = SAMPLINT_ins(df=key_df, ratio=test_ratio, seed=seed)
    result = {'test':CONVERT_ins(key_df=test_key_df)}
    if valid_ratio is not None:
        valid_key_df , train_key_df = SAMPLINT_ins(df=rest_key_df, ratio=valid_ratio, seed=seed)
        result['train'] = CONVERT_ins(key_df=train_key_df)
        result['valid'] = CONVERT_ins(key_df=valid_key_df)
    else:
        result['train'] = CONVERT_ins(key_df=rest_key_df)
    return result


def _k_size_and_seed_list_error_in_make_stratified_idx_dict(k_size:int, seed_list:List[int]):
    """
    k_size의 크기보다 seed_list의 원소의 갯수가 적은 경우, ValueError를 출력함.
    """
    seed_size = len(seed_list)
    if k_size > seed_size:
        raise ValueError(f"k_size: {k_size}, seed_list의 크기: {seed_size}입니다. seed_list의 크기는 k_size보다 반드시 크거나 같아야 합니다.")