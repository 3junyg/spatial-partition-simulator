# Spatial Partition Simulator

밀도·유입·유출·환경 특성을 반영해 불규칙한 평면을 대표 직사각형들로 근사하는 Python 데스크톱 실험 도구입니다.

## 실행

Python 3.10 이상이 필요합니다. Windows 공식 Python 설치의 Tkinter와 NumPy/SciPy를 사용합니다.

```bash
python -m pip install -r requirements.txt
python spatial_simulation.py
```

Windows에서는 의존성 설치 후 `run_simulation.bat`를 실행해도 됩니다.

## 사용 순서

1. 공간 유형과 구조·사람·유입/유출·환경 시드를 설정합니다.
2. `가상 관측 생성`을 눌러 4m² 격자별 정보를 만듭니다.
3. 상단에서 람다를 입력하거나 슬라이더·세밀/중간/단순 버튼으로 조절합니다.
4. `직사각형 계산`을 누르거나 `람다 3개 비교`에서 같은 원본의 결과를 비교합니다.
5. 구역별 평균·오차와 최적성 간격을 확인하고 필요하면 JSON으로 저장합니다.

## 모델

- 대표 상황을 한 번 분할하는 정적 모델입니다. 시간에 따라 사람이 이동하는 시뮬레이션이 아닙니다.
- 실제 공간 면적으로 나눈 정수 인원이 밀도이며, 유입·유출은 공간 연결 그래프에서 평활화합니다.
- 기본 특성 가중치는 밀도 60%, 유입 10%, 유출 10%, 환경 지표 20%입니다.
- 공간 밖 돌출을 허용하지만 외부를 실제 관측 데이터로 취급하지 않습니다.
- 정규화된 내부 오차, 직사각형 개수 비용 λM, 돌출 면적 비용을 합해 최소화합니다.
- 구역 수의 고정 하한·상한은 없습니다. 기본 λM은 0.08이며 작을수록 세밀한 분할을 선호합니다. 결과는 공간과 데이터에 따라 달라집니다.
- 최대 특성 차이는 경고로만 사용합니다.
- SciPy/HiGHS 0–1 정수최적화를 사용합니다. 시간 제한이 발생하면 현재 정수해 또는 유효 대체해와 최적성 상태를 표시합니다.

환경의 여러 변수를 공통 기준으로부터의 거리 하나로 축약하므로 다른 환경이 같은 지표를 가질 수 있습니다. 모든 장소 예시와 관측은 합성 데이터이며 실제 시설 도면·실측값이 아닙니다. 열·기류 방정식이나 검증된 열쾌적성 예측을 제공하지 않습니다. 상세 식과 가정은 [MODEL.md](MODEL.md)를 참고하세요.

## 파일

| 파일 | 역할 |
| --- | --- |
| `spatial_simulation.py` | Tkinter 화면, 람다 조절과 결과 비교 |
| `spatial_model.py` | 격자 특성, 평활화, 후보 생성, 정수최적화 |
| `scenario_generator.py` | 시드 기반 예시 공간·기대 밀도 생성 |
| `test_model.py` | 커버·겹침·구역 수 제한 해제 검증 |
| `MODEL.md` | 수식, 가정, 해석상 한계 |

## 검증

```bash
python test_model.py
```

## 참고 연구

Demiröz, Altınel, Akarun (2019), *Rectangle blanket problem: Binary integer linear programming formulation and solution algorithms*. https://doi.org/10.1016/j.ejor.2019.02.004

형태 근사 아이디어를 참고한 자체 확장 모델이며, 해당 논문의 branch-and-price 구현은 아닙니다.

## 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE)로 배포됩니다.
