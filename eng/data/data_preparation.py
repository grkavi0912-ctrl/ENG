#for data manipulation
import pandas as pd
import sklearn
import os

#for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#for converting text data into numerical representation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

#for hugging face space authentication to upload files
from huggingface_hub import login,HfApi

api=HfApi(token=os.getenv("HFTOKEN"))
DATASET_PATH = "hf://datasets/grkavi0912/ENG/engine.csv"
df=pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully")

# --- DIAGNOSTIC PRINT ---
print(f"Columns initially loaded: {df.columns.tolist()}")

column_name_mapping = {
    'Engine rpm': 'engine_rpm',
    'Lub oil pressure': 'lub_oil_pressure',
    'Fuel pressure': 'fuel_pressure',
    'Coolant pressure': 'coolant_pressure',
    'lub oil temp': 'lub_oil_temp',
    'Coolant temp': 'coolant_temp',
    'Engine Condition': 'engine_condition'
}
# Check if original 'Engine Condition' is present before renaming
print(f"Before rename - 'Engine Condition' in df.columns: {'Engine Condition' in df.columns}")

# Apply the renaming to the DataFrame, reassigning to be explicit
df = df.rename(columns=column_name_mapping, errors='raise')
print("Columns renamed to snake_case.")

# --- DIAGNOSTIC PRINT ---
print(f"Columns after renaming operation: {df.columns.tolist()}")
# Check if new 'engine_condition' is present after renaming
print(f"After rename - 'engine_condition' in df.columns: {'engine_condition' in df.columns}")


target = "engine_condition"

# --- DIAGNOSTIC CHECK BEFORE DROP ---
if target not in df.columns:
    print(f"ERROR: Target column '{target}' not found in DataFrame columns after renaming. Current columns: {df.columns.tolist()}")
    raise KeyError(f"Target column '{target}' missing after rename. Available: {df.columns.tolist()}")
else:
    print(f"Target column '{target}' found. Proceeding with split.")

X = df.drop(columns=[target])
y = df[target]
print("Features (X) and target (y) split successfully.")

xtrain,xtest,ytrain,ytest = train_test_split(
    X,y, test_size=0.2,random_state=42
)
print("Data split into train and test sets.")

xtrain.to_csv("xtrain.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
xtest.to_csv("xtest.csv",index=False)
ytest.to_csv("ytest.csv",index=False)
print("Train/test sets saved to CSV files.")

files = ("xtrain.csv","xtest.csv","ytrain.csv","ytest.csv")

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="grkavi0912/ENG",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to Hugging Face.")
