import os
import boto3
from dotenv import load_dotenv
from urllib.parse import urlencode, unquote

load_dotenv()
regeion_name2 = os.environ.get("regeion_name")
aws_key_id = os.environ.get("aws_access_key_id")
aws_access_key = os.environ.get("aws_secret_access_key")


def s3_connection():
    try:
        s3 = boto3.client(
            service_name="s3",
            region_name=unquote(regeion_name2),
            aws_access_key_id=unquote(aws_key_id),
            aws_secret_access_key=unquote(aws_access_key),
        )
    except Exception as e:
        print(e)
    else:
        print("s3 bucket connected!")
        return s3


def upload_to_s3():
    s3 = s3_connection()

    folder_path = "C:/yyh/save"

    # 최신 파일 찾기
    latest_file = max(
        (f.path for f in os.scandir(folder_path) if f.is_file()), key=os.path.getctime
    )

    for filename in os.listdir(folder_path):
        if filename == os.path.basename(latest_file):
            continue

        try:
            with open(os.path.join(folder_path, filename), "rb") as f:
                s3.upload_fileobj(
                    f,
                    "project-s3-data",
                    filename,
                )
        except Exception as e:
            print(e)
        else:
            os.remove(os.path.join(folder_path, filename))
