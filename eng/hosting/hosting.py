from huggingface_hub import HfApi
import os

api =HfApi(token=os.getenv("HFTOKEN"))
api.upload_folder(
    folder_path="eng/deployment",
    repo_id="grkavi0912/ENG",
    repo_type="space",
    path_in_repo="",
    commit_message="Update deployment files with latest app.py and requirements.txt"
)
