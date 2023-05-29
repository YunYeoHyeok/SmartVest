import os
import boto3


def s3_connection():
    try:
        s3 = boto3.client(
            service_name="s3",
            region_name="ap-northeast-2",
            aws_access_key_id="AKIA3YT2IIE7I5ZWSM7H",
            aws_secret_access_key="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",
        )
    except Exception as e:
        print(e)
    else:
        print("s3 bucket connected!")
        return s3


s3 = s3_connection()

for filename in os.listdir("/home/pi/webapps/1teamProject/SAVE"):
    try:
        with open(
            os.path.join("/home/pi/webapps/1teamProject/SAVE", filename), "rb"
        ) as f:
            s3.upload_fileobj(
                f,
                "project-s3-data",
                filename,
            )
    except Exception as e:
        print(e)
