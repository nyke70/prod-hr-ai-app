"""Upload the already-built React website to the staging S3 bucket.

React produces files in frontend/dist. This script uploads those files and tells
CloudFront to refresh its cached home page. It does not build React or test it.
"""

import mimetypes
import os
import uuid
from pathlib import Path

import boto3


def main():
    # 1. Find the React files produced by npm run build.
    project_folder = Path(__file__).resolve().parents[1]
    website_folder = project_folder / "frontend" / "dist"
    if not (website_folder / "index.html").is_file():
        raise SystemExit("Build React before deploying the frontend.")

    # 2. Check that we are deploying to a classroom bucket.
    bucket_name = os.environ["HR_STAGING_FRONTEND_BUCKET"]
    if not bucket_name.startswith("prod-hr-ci-"):
        raise SystemExit("Use the lab bucket with an prod-hr-ci- prefix.")

    # 3. Upload each website file, with its browser content type.
    s3 = boto3.client("s3")
    for website_file in website_folder.rglob("*"):
        if not website_file.is_file():
            continue

        object_name = website_file.relative_to(website_folder).as_posix()
        content_type = mimetypes.guess_type(website_file.name)[0]
        if content_type is None:
            content_type = "application/octet-stream"

        # The home page changes between releases. Hashed assets can be cached.
        if object_name == "index.html":
            cache_control = "no-cache"
        else:
            cache_control = "public,max-age=31536000,immutable"

        s3.upload_file(
            str(website_file),
            bucket_name,
            object_name,
            ExtraArgs={"ContentType": content_type, "CacheControl": cache_control},
        )

    # 4. Ask CloudFront to stop serving an old home page.
    cloudfront = boto3.client("cloudfront")
    cloudfront.create_invalidation(
        DistributionId=os.environ["HR_STAGING_DISTRIBUTION_ID"],
        InvalidationBatch={
            "Paths": {"Quantity": 2, "Items": ["/", "/index.html"]},
            "CallerReference": uuid.uuid4().hex,
        },
    )
    print("Staging frontend uploaded. Browser and API checks are still required.")


if __name__ == "__main__":
    main()
