from huggingface_hub.utils import validate_repo_id, HfHubHTTPError,RepositoryNotFoundError
from huggingface_hub import HfApi
from huggingface_hub import create_repo
import os

repo_id = "grkavi0912/ENG"
repo_type = "dataset" # Corrected from "Datasets" to "dataset"
private = False
token = "HFTOKEN" # Placeholder, will be replaced by userdata.get() later
#Initialize API client
api = HfApi(token=os.getenv("HFTOKEN")) # Corrected missing parenthesis and used os.getenv as in original notebook

#step 1:check if the space exists
try:
    api.repo_info(repo_id=repo_id,repo_type=repo_type)
    print(f"Repo {repo_id} already exists")
except RepositoryNotFoundError:
    print(f"Repo {repo_id} not found")
    create_repo(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo {repo_id} created")

#api.upload_folder(
    #folder_path="eng/data",
    #repo_id=repo_id,
    #repo_type=repo_type,
#)
api.upload_folder(
    folder_path="eng/data",
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Initial dataset upload",
    ignore_patterns=["*.py", "__pycache__/*"]
)
