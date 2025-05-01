### 설치 방법

#### 1. 저장소 클론
```bash
git clone https://github.com/your-username/l2-pronunciation-error-detection.git
cd l2-pronunciation-error-detection
```

#### 2. 자동 환경 설정 (권장)
포함된 `setup.sh` 스크립트를 사용하여 필요한 모든 환경을 자동으로 설정할 수 있습니다:

```bash
chmod +x setup.sh
./setup.sh
```

이 스크립트는 다음 작업을 수행합니다:
- Python 가상 환경 생성 및 활성화
- 필수 데이터 다운로드 (download.sh 실행)
- requirements.txt에서 필요한 모든 패키지 설치

#### 3. 수동 환경 설정 (대안)
자동 설정이 작동하지 않는 경우 다음 단계를 수행하세요:

1. Python 가상 환경 생성 및 활성화:
```bash
python3 -m venv .env
source .env/bin/activate  # Linux/Mac
# 또는
.env\Scripts\activate     # Windows
```

2. 필수 데이터 다운로드:
```bash
bash download.sh
```

3. 필수 패키지 설치:
```bash
pip install -r requirements.txt
```

### 환경 활성화
설치 후 나중에 가상 환경을 활성화해야 할 경우:

```bash
source .env/bin/activate  # Linux/Mac
# 또는
.env\Scripts\activate     # Windows
```