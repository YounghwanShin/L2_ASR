import os
import re
import sys
import subprocess
import time
import urllib.parse

def install_gdown_if_needed():
    try:
        import gdown
        print("gdown이 이미 설치되어 있습니다.")
    except ImportError:
        print("gdown이 설치되어 있지 않습니다. 설치 중...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        print("gdown 설치 완료!")

def extract_id_from_url(url):
    file_match = re.search(r'file/d/([^/]+)', url)
    if file_match:
        return file_match.group(1)
    
    folder_match = re.search(r'folders/([^/?]+)', url)
    if folder_match:
        return folder_match.group(1)
    
    return None

def download_file(file_url, output_dir):
    file_id = extract_id_from_url(file_url)
    if not file_id:
        print(f"경고: {file_url}에서 파일 ID를 추출할 수 없습니다. 건너뜁니다.")
        return False
    
    try:
        import gdown
        output_path = os.path.join(output_dir, f"file_{file_id}")
        
        if os.path.exists(output_path):
            print(f"파일이 이미 존재합니다: {output_path}")
            return True
        
        print(f"파일 다운로드 중: {file_id}")
        gdown.download(id=file_id, output=output_path, quiet=False)
        print(f"파일 다운로드 완료: {output_path}")
        return True
    except Exception as e:
        print(f"파일 다운로드 중 오류 발생: {e}")
        return False

def download_folder(folder_url, output_dir):
    folder_id = extract_id_from_url(folder_url)
    if not folder_id:
        print(f"경고: {folder_url}에서 폴더 ID를 추출할 수 없습니다. 건너뜁니다.")
        return False
    
    try:
        import gdown
        print(f"폴더 다운로드 중: {folder_id}")
        gdown.download_folder(id=folder_id, output=output_dir, quiet=False)
        print(f"폴더 다운로드 완료: {folder_id}")
        return True
    except Exception as e:
        print(f"폴더 다운로드 중 오류 발생: {e}")
        return False

def main():
    base_dir = "data"
    os.makedirs(base_dir, exist_ok=True)
    
    l2arctic_dir = os.path.join(base_dir, "l2arctic_dataset")
    os.makedirs(l2arctic_dir, exist_ok=True)
    
    install_gdown_if_needed()
    
    direct_files = [
        "https://drive.google.com/file/d/12X8eotMD3N9rq1savpdAzSC9XPL5AnZS/view?usp=sharing",
        "https://drive.google.com/file/d/1DbIckREiWy5aJ_uu3fNClZ-oKI75_pR0/view?usp=sharing",
        "https://drive.google.com/file/d/1T6bgWrBEGplHy4k-DTsgOhek9KuZqXbK/view?usp=sharing",
        "https://drive.google.com/file/d/1edurRF9LsVQ3RLP1s_bHs7PBvGM7ia6F/view?usp=sharing",
        "https://drive.google.com/file/d/1g2RCl7S27ZVkvRmDiwmAjACBGL6mSe7n/view?usp=sharing",
        "https://drive.google.com/file/d/1q0H3_cR0byKLLnWpziB8BKP_Uo24qk3i/view?usp=sharing",
        "https://drive.google.com/file/d/1s0EddcRq_iHlM0Yll4GVrIVzlGXNM8IU/view?usp=sharing",
    ]
    
    l2arctic_folders = [
        "https://drive.google.com/drive/folders/12P5XhSZi8QtygTP5fuN9J9pv3eZxI1sq?usp=sharing",
        "https://drive.google.com/drive/folders/179Z8-Mg_hfimLlI9t0GCSMW0isN46sGL?usp=sharing",
        "https://drive.google.com/drive/folders/19_XExVZv-CUltPIEWprOMArDowN1rmro?usp=sharing",
        "https://drive.google.com/drive/folders/1BiQijk2cQBE16mx8mr5xZt_PZmke_Emf?usp=sharing",
        "https://drive.google.com/drive/folders/1BpkiTkwbR9uDdzhBpH5BZ7amRtUXF4FE?usp=sharing",
        "https://drive.google.com/drive/folders/1ET6wFyRAxyOUjzYbX0rGVhY05rVfNLaU?usp=sharing",
        "https://drive.google.com/drive/folders/1JY2sAYpjYG3_3bXhhKY915wTNTaZvjfd?usp=sharing",
        "https://drive.google.com/drive/folders/1M6WkuLoqu_4fBPn6mym1E66YiOChA0YF?usp=sharing",
        "https://drive.google.com/drive/folders/1Nxryz5JuivVAd8iiXRcrfzpQUSjwC_GF?usp=sharing",
        "https://drive.google.com/drive/folders/1RbLM2YoIVt1WxBB5AQGBRjFoI3CRnSSN?usp=sharing",
        "https://drive.google.com/drive/folders/1UJBFdM5P_pTaDJHZXAjNXXul1niq7745?usp=sharing",
        "https://drive.google.com/drive/folders/1dj4NsZbL-rEoBFaPigJ41Ps-0dENZIap?usp=sharing",
        "https://drive.google.com/drive/folders/1gJYVVwtOIBVF9WS2-Gx5crIQsV6wLdc9?usp=sharing",
        "https://drive.google.com/drive/folders/1hZCHlvNsGY_cDKeHX_nj6J4XvF01LKpy?usp=sharing",
        "https://drive.google.com/drive/folders/1j9yBJMlVYj37-ag6RCJdsdFk1OzH3bTC?usp=sharing",
        "https://drive.google.com/drive/folders/1kUk-2anvICyT_w-8bTCJ3AP_22tyJarK?usp=sharing",
        "https://drive.google.com/drive/folders/1nTf6ZqbISnjYJASM5t7Opu5wWp0TPE9e?usp=sharing",
        "https://drive.google.com/drive/folders/1s3yvFBUV_2X6QhSniLPyiS0mLGpwkQFh?usp=sharing",
        "https://drive.google.com/drive/folders/1tYn5v-AvI9yP-HKDA3MNxEUViU-5uqAT?usp=sharing",
        "https://drive.google.com/drive/folders/1umct16vRMlegb7xv2Kgo0h4yGIGdcvtf?usp=sharing",
        "https://drive.google.com/drive/folders/1vKuOrC3VC_npzhp_yt-xG0qW1TFDbWeX?usp=sharing",
        "https://drive.google.com/drive/folders/1vg1f-IJHhc-h3eWIFFQ7SDONgB3RRkBZ?usp=sharing",
        "https://drive.google.com/drive/folders/1wA54T14zReIFblQ25uwz_dkUbA7QRkwJ?usp=sharing",
        "https://drive.google.com/drive/folders/1yjHPZE9FNq5mrj0Iru5sWtrduYX7w0iC?usp=sharing",
    ]
    
    print("===== 파일 다운로드 시작 =====")
    for url in direct_files:
        success = download_file(url, base_dir)
        if not success:
            print(f"파일 다운로드 실패: {url}")
        time.sleep(2)
    
    print("\n===== 폴더 다운로드 시작 =====")
    for url in l2arctic_folders:
        success = download_folder(url, l2arctic_dir)
        if not success:
            print(f"폴더 다운로드 실패: {url}")
        time.sleep(3)
    
    print("\n모든 다운로드가 완료되었습니다.")
    print(f"파일 위치: {base_dir}")
    print(f"L2Arctic 데이터셋 위치: {l2arctic_dir}")

if __name__ == "__main__":
    main()