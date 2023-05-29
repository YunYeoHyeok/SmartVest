import requests
import os
import hashlib
from dotenv import load_dotenv
from urllib.parse import urlencode, unquote

load_dotenv()
cse_id = os.environ.get("cse_id")
api_key = os.environ.get("api_key")

search_term = "포크레인"
image_down = 200

save_folder = "imagess"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

down_md5 = set()  # 중복된 이미지의 MD5 해시값을 저장할 집합
file_counter = 1  # 저장된 파일의 순서를 카운트할 변수

for i in range(image_down):
    start_index = i + 1
    url = f"https://www.googleapis.com/customsearch/v1?q={search_term}&start={start_index}&num=5&imgSize=large&searchType=image&cx={unquote(cse_id)}&key={unquote(api_key)}"
    # num=1 값을 num 적절하게 바꿔보기
    try:
        response = requests.get(url, verify=False)
        data = response.json()
        if "items" in data:
            for j, item in enumerate(data["items"]):
                response = requests.get(item["link"], verify=False)
                if response.status_code == 200:
                    md5_hash = hashlib.md5(response.content).hexdigest()
                    if md5_hash not in down_md5:  # 중복된 이미지인지 확인
                        file_name = f"{file_counter}ac.jpg"  # 파일이름
                        file_path = os.path.join(save_folder, file_name)
                        with open(file_path, "wb") as f:
                            f.write(response.content)
                        down_md5.add(md5_hash)
                        file_counter += 1
                    else:
                        continue
                else:
                    continue
                break
        else:
            print(f"Image index {i + 1}.")
    except Exception as e:
        print(f"Error image {i + 1}: {e}")
