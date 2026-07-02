from clearml import Dataset
import boto3
from botocore.config import Config


def main(prepared_dataset_id: str) -> bool:
    """
    Delete the prepared ClearML dataset (metadata + Cloudflare R2 files).
    Returns True on success, does not raise so the pipeline never fails because of cleanup.
    """

    try:
        dataset = Dataset.get(dataset_id=prepared_dataset_id)

        # The upload_uri used in prepare_dataset is:
        #   s3://8b56346322f98ed029a3c888fba38a69.r2.cloudflarestorage.com:443/ml-storage/processed-datasets/
        # ClearML stores files under  <upload_uri>/<dataset_id>/
        upload_uri = dataset.get_upload_uri() or ""
        print(f"[cleanup] upload_uri = {upload_uri}")
    except Exception as e:
        print(f"[cleanup] Could not fetch dataset metadata: {e}")
        return False

    try:
        import os
        uri = upload_uri.lstrip("s3://") if upload_uri.startswith("s3://") else upload_uri
        host_port, _, rest = uri.partition("/")
        bucket, _, key_prefix = rest.partition("/")

        folder_prefix = f"{key_prefix}{prepared_dataset_id}/".replace("//", "/")

        # R2 endpoint — derive from host_port
        r2_endpoint = f"https://{host_port}"

        r2_access_key = os.environ.get("R2_ACCESS_KEY_ID", "")
        r2_secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "")

        if not r2_access_key or not r2_secret_key:
            print("[cleanup] R2 credentials not set — skipping R2 deletion.")
        else:
            s3 = boto3.client(
                "s3",
                endpoint_url=r2_endpoint,
                aws_access_key_id=r2_access_key,
                aws_secret_access_key=r2_secret_key,
                config=Config(signature_version="s3v4"),
            )
            paginator = s3.get_paginator("list_objects_v2")
            deleted = 0
            for page in paginator.paginate(Bucket=bucket, Prefix=folder_prefix):
                objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
                if objects:
                    s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
                    deleted += len(objects)
            print(f"[cleanup] Deleted {deleted} R2 objects under s3://{bucket}/{folder_prefix}")
    except Exception as e:
        print(f"[cleanup] R2 deletion failed: {e}")
    try:
        Dataset.delete(dataset_id=prepared_dataset_id, force=True)
        print(f"[cleanup] ClearML dataset {prepared_dataset_id} deleted.")
    except Exception as e:
        print(f"[cleanup] ClearML dataset deletion failed: {e}")

    return True
