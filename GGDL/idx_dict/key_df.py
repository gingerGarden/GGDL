from typing import List, Dict, Any, Tuple, Callable, Union, Optional
import re
import numpy as np
import pandas as pd


def make_basic_key_df(
    paths: Union[List[str], np.ndarray], 
    labels: Union[List[str], np.ndarray],
    use_specific_column:bool=True,
    **kwargs: Union[List[str], np.ndarray]
) -> pd.DataFrame:
    """가장 기본적인 형태의 key_df를 생성한다 - path, label 2개의 컬럼으로 구성
    >>> path: 이미지 파일의 절대 경로
        label: 이미지 파일의 label

    Args:
        paths (Union[List[str], np.ndarray]): 파일의 절대 경로들을 list 또는 array로 입력
        labels (Union[List[str], np.ndarray]): 파일의 label을 list 또는 array로 입력
        use_specific_column (bool, optional): kwargs로 추가되는 컬럼을 make_basic_key_df의 고유 컬럼명으로 추가할지 여부. Defaults to True.
        **kwargs (Union[List[str], np.ndarray]): 추가적으로 들어갈 class(데이터 분할 시 추가 속성)
            - use_specific_column=False인 경우, 지정한 파라미터명이 새로운 컬럼명이 된다.

    Returns:
        pd.DataFrame: path, label 및 kwargs로 전달된 추가 class 컬럼이 포함된 DataFrame
    """
    data = {'path': paths, 'label': labels}
    
    # kwargs로 전달된 데이터 추가
    for i, (column, value) in enumerate(kwargs.items()):
        if use_specific_column:
            data[f"class_{i}"] = value
        else:
            data[column] = value
    return pd.DataFrame(data)


def binary_label_convertor(array:Union[np.ndarray, pd.Series], positive_class:Union[str, int])->np.ndarray:
    """
    array를 positive_class를 1로, 나머지는 0으로 이진 변환한다.

    Args:
        array (Union[np.ndarray, pd.Series]): positive_class와 다른 class들이 포함된 시리즈 또는 array
        postive_class (Union[str, int]): 이진 분류 시 기준(1)이 될 element의 class

    Returns:
        np.ndarray: 0과 1로 이진화된 1차원 배열
    """
    return np.where(array == positive_class, 1, 0)


def add_multiclass_onehot_columns(df:pd.DataFrame, column:str, key:str='dummy', inplace:bool=False)->pd.DataFrame:
    """
    multiclass에 대하여 onehot embedding 결과를 기존 df에 추가하거나, 새로 출력함

    Args:
        df (pd.DataFrame): multiclass가 포함된 pd.DataFrame
        column (str): df에서 multiclass를 갖고 있는 column명
        key (str, optional): multiclass를 onehot column으로 생성 시, 추가되는 키(f"{column}_{key}"). Defaults to 'dummy'.
        inplace (bool, optional): 원본 DataFrame에 바로 반영할지 여부(False일 경우 복사본 반환). Defaults to False.

    Raises:
        ValueError: multiclass로 추가된 컬럼명({", ".join(duplicated_key)})이 기존 df에 이미 존재합니다. dummy의 값을 바꾸십시오.

    Returns:
        pd.DataFrame: multiclass를 onehot embedding하여 column으로 추가한 DataFrame(key_df에 추가 또는 복사)
    """
    df_copy = df.copy(deep=True) if not inplace else df     #inplace에 따라 df에 바로 추가할지 여부를 결정
    dummy_df = pd.get_dummies(df_copy[column], prefix=f'{column}_{key}', dtype=int)     # dummy 변수 생성

    # dummy_df의 column이 기존 column과 충돌나는 경우 ValueError 발생
    duplicated_key = set(dummy_df.columns) & set(df_copy.columns)
    if duplicated_key:
        raise ValueError(f"multiclass로 추가된 컬럼명({", ".join(duplicated_key)})이 기존 df에 이미 존재합니다. dummy의 값을 바꾸십시오.")

    # 컬럼 추가
    df_copy[dummy_df.columns] = dummy_df
    if not inplace:     # inplace가 False일 경우 df_copy를 따로 출력
        return df_copy
    


class KeyDF2KeyDict:

    def __init__(self, path_col:str='path', label_col:str='label', is_binary:bool=True, onehot_key:str='dummy'):
        """
        key_df의 각 레코드를 Dict으로 변환하며, 이 Dict들은 List 안에 들어가 있다.
        >>> callable class로 train, test, validation의 key_df를 대상으로 path와 label에 대하여 key_dict을 생성함
            - binary: 각 레코드에 대해 {"path": path, "label": 스칼라 값} 형태로 출력.
                ex) [
                        {"path":path/filename1.jpg, "label":0}, 
                        {"path":path/filename2.jpg, "label":1}
                    ]
            - multiclass: 각 레코드에 대해 {"path": path, "label": np.ndarray(원-핫 벡터)} 형태로 출력.
                    [
                        {"path":path/filename1.jpg, "label":np.ndarray([0, 0, 1, 0])}, 
                        {"path":path/filename1.jpg, "label":np.ndarray([1, 0, 0, 0])}
                    ]

        Args:
            path_col (str, optional): key_df의 path의 컬럼명. Defaults to 'path'.
            label_col (str, optional): key_df의 label의 컬럼명. Defaults to 'label'.
            is_binary (bool, optional): key_df가 이진 분류를 대상으로 하는지 여부. Defaults to True.
                >>> 이진 분류인 경우, label은 스칼라.
                    다중 분류인 경우, label은 np.ndarray(1차원 배열)로 변환.
            onehot_key (str, optional): 다중 분류의 경우, key_df에서 원-핫 인코딩된 컬럼들의 접두사를 정의. Defaults to 'dummy'.
                >>> 다중 분류 시, 컬럼명은 f"{column}_{key}_{class}"로 구성된다.
                >>> class는 레이블 값의 클래스에 해당함.
        """
        self.path_col = path_col
        self.label_col = label_col
        self.is_binary = is_binary
        self.onehot_pattern = f"^{label_col}_{onehot_key}"

    
    def __call__(self, key_df:pd.DataFrame)->List[Dict[set, Union[str, np.ndarray]]]:
        """
        클래스의 인스턴스를 함수처럼 호출할 수 있도록 정의된 메서드.
        key_df에 따라 이진 분류 또는 다중 분류에 맞는 key_dict 리스트를 반환한다.

        Args:
            key_df (pd.DataFrame): path와 label 정보를 포함한 DataFrame. 이 데이터프레임을 기반으로 key_dict 리스트를 생성한다.

        Returns:
            List[Dict[str, Union[str, np.ndarray]]]: 각 레코드를 딕셔너리로 변환한 리스트.
                - 이진 분류의 경우: {"path": 경로, "label": 스칼라 값} 형태.
                - 다중 분류의 경우: {"path": 경로, "label": np.ndarray(1차원 배열)} 형태.
        """
        return self._binary(key_df) if self.is_binary else self._multiclass(key_df)


    def _binary(self, key_df:pd.DataFrame)->List[Dict[str, str]]:
        """
        이진 분류 key_df를 대상으로 key_dict을 생성하는 경우
        """
        def record_to_dict(x):
            path = x[self.path_col]
            label = x[self.label_col]
            return {'path':path, 'label':label}
        return key_df.apply(record_to_dict, axis=1).to_list()
    

    def _multiclass(self, key_df:pd.DataFrame)->List[Dict[str, np.ndarray]]:
        """
        다중 분류 key_df를 대상으로 key_dict을 생성하는 경우
        """
        def record_to_dict(idx):
            path = key_df.loc[idx, self.path_col]
            label_array = label_mat[idx]
            return {'path':path, 'label':label_array}

        # self.onehot_pattern과 매칭되는 컬럼들을 list로 가지고 온다.
        label_columns = [i for i in key_df.columns if re.match(self.onehot_pattern, string=i)]
        # self.onehot_pattern이 존재하지 않는 경우 경로문 출력
        if not label_columns:
            raise ValueError(f"No columns match the pattern {self.onehot_pattern}")
        label_mat = key_df[label_columns].values
        return list(map(record_to_dict, key_df.index))
    


class StratifiedSampling:
    def __init__(self, columns:List[str], rounding:Callable=np.ceil):
        """
        key_df에 대하여 n개의 column에 대하여 층화 표본 추출 한다.
        
        >>> Callable class로 서로 다른 비율에 대하여, train, valdation, test 3개의 key_df를 생성하는 것을 주 목적으로 함.
            ex) # 전체 데이터의 20%를 test셋으로, 나머지의 10%를 validation set으로 하는 경우.
                this_ins = StratifiedSampling(columns=['label', 'class_0'])
                test_key_df, rest_key_df = this_ins(df=key_df, ratio=0.2)
                valid_key_df, train_key_df = this_ins(df=rest_key_df, ratio=0.1)

        Args:
            columns (List[str]): 층화 표본 추출할 대상이 되는 변수명
            rounding (Callable, optional): 추출할 데이터의 크기 산정 시, 반올림 방식. Defaults to np.ceil.
        """
        self.columns = columns
        self.rounding = rounding
        self.df = None
        self.freq_df = None


    def __call__(
            self, df:pd.DataFrame, ratio:float, seed:Optional[int]=None
        )->Tuple[pd.DataFrame, pd.DataFrame]:
        """
        df를 ratio의 비율로, seed의 난수로 완전 무작위 추출함(층화 표본 추출).

        Args:
            df (pd.DataFrame): 분할의 대상이 되는 key_df
            ratio (float): 분할의 비율
            seed (Optional[int], optional): 난수. Defaults to None.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 추출된 key_df, 나머지 key_df
        """
        self._instance_variable(df, ratio, seed)        # 초기 변수 설정
        # 계층별 비율을 고려한 무작위 index에 대한 df
        choosed_idx = self._choose_random_index()       # index 추출
        choosed_df = self.df.loc[choosed_idx].reset_index(drop=True)
        # 나머지 index에 대한 df
        rest_idx = list(set(self.df.index) - set(choosed_idx))
        rest_df = self.df.loc[rest_idx].reset_index(drop=True)
        return choosed_df, rest_df


    def _instance_variable(self, df:pd.DataFrame, ratio:float, seed:int):
        """
        instance내 주요 변수(self.df, self.freq_df)와 난수 설정
        """
        if seed is not None: np.random.seed(seed)
        self.df = df
        self._freq_table(ratio)     # 빈도표 생성


    def _freq_table(self, ratio:float):
        """
        무작위 추출할 크기(choose)가 추가된 빈도표 생성
        """
        self.freq_df = pd.DataFrame(self.df[self.columns].value_counts().sort_index())          # 빈도표 생성
        self.freq_df['choose'] = self.rounding(self.freq_df['count'] * ratio).astype(np.int64)  # ratio가 반영된 choose 컬럼 생성


    def _choose_random_index(self)->np.ndarray:
        """
        각 계층에 대하여, self.freq_df의 choose 값을 기반으로 key_df의 index를 완전 무작위 추출
        """
        stack = np.array([])
        for idx in self.freq_df.index:

            size = self.freq_df.loc[idx, 'choose']
            target_idx = self._get_target_idx(idx)      # freq_df의 index에 해당하는 df의 index를 가지고 온다
            if size > len(target_idx): size = len(target_idx)   # 올림으로 인해 size가 실제 크기를 초과할 경우, size 수정

            choosed_idx = np.random.choice(target_idx, size=size, replace=False)
            stack = np.concatenate((stack, choosed_idx))
        return stack


    def _get_target_idx(self, idx:Tuple[str, ...])->List[int]:
        """
        freq_df의 index(columns)에 해당하는 index를 선택한다.
        """
        # result = set(self.df.index)
        mask = np.ones(len(self.df), dtype=bool)
        for i, column in enumerate(self.freq_df.index.names):
            mask = mask & (self.df[column] == idx[i])
        return self.df[mask].index.tolist()