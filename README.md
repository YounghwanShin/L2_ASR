### 설치 방법

#### 1. 저장소 클론
```bash
git clone https://github.com/your-username/l2-pronunciation-error-detection.git
cd l2-pronunciation-error-detection
```

### 2. 가상 환경 설정
```bash
# 가상 환경 생성
python3 -m venv .env

# 가상 환경 활성화 (Linux/Mac)
source .env/bin/activate

# 가상 환경 활성화 (Windows)
.env\Scripts\activate
```

### 3. 필요 패키지 설치
```bash
pip install -r requirements.txt
```

## 데이터 다운로드

프로젝트에 필요한 데이터를 다운로드하기 위해 다음 스크립트를 실행하세요:

```bash
python download_drive.py
```