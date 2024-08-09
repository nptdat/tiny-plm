import subprocess
from logging import basicConfig, getLogger
from pathlib import Path

logger = getLogger(__name__)
basicConfig(
    level="INFO", format="%(asctime)s : %(levelname)s : %(name)s : %(message)s"
)


class SimpleS3:
    """Simple class to download from and upload to S3 via awscli."""

    def __init__(self, bucket_name: str) -> None:
        self.bucket_name = bucket_name

    def download(self, s3_key: str, local_path: str) -> None:
        s3_path = f"s3://{self.bucket_name}/{s3_key}"
        commands = ["aws", "s3", "cp", s3_path, local_path]
        last_name = s3_key.split("/")[-1]
        if len(last_name.split(".")) == 1:
            # has extension -> a file (TODO: handle file without extension)
            commands.append("--recursive")
        logger.info(f"{commands=}")
        logger.info(f"Downloading {s3_path} to {local_path}")
        subprocess.run(commands)

    def upload(self, local_path: str, s3_key: str) -> None:
        s3_path = f"s3://{self.bucket_name}/{s3_key}"
        logger.info(f"Uploading {local_path} to {s3_path}")

        commands = ["aws", "s3", "cp", local_path, s3_path]
        if Path(local_path).is_dir():
            commands.append("--recursive")
        subprocess.run(commands)
