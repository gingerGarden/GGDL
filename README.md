## GGDL
딥러닝에 사용되는 다양한 패키지의 모음
* 본 프레임워크는 idx_dict 이라는 {path, label} 등의 묶음으로 이루어진 dictionary를 기반으로 학습 파이프라인이 동작합니다.
* 본 프레임워크는 torch, timm을 기반으로 Computer vision의 Classification, Detection, Segmentation 을 쉽게 하는 것을 목적으로 합니다.
* Text를 대상으로 하는 모델도 추가 예정에 있습니다.
* 대상 집단의 주요 Feature들을 대상으로 Stratified k-fold cross validation 하는 것을 기본으로 합니다.
* 현재 개발 진행 중인 코드로, 내부 코드들과 라이브러리 구조는 언제든지 수정될 수 있습니다.


</br>
</br>
</br>

* 해당 라이브러리는 다음과 같은 패키지들로 구성되어 있습니다.

</br>

## F. 외부 라이브러리 의존성
* 해당 코드는 현재 개발중인 단계로 CI/CD 를 위한 외부 라이브러리 의존성 버전 확인이 진행되지 않은 상태입니다.
* 현재 작성되어 있는 각 라이브러리의 버전은 개발 환경의 버전입니다.
> * opencv-python == 4.10.0
> * numpy == 1.26.4
> * torch == 2.3.1
> * timm == 1.0.9
> * GGUtils == 0.0.0
> * GGStatify == 0.0.0