import pandas as pd
import pickle
import warnings
import os
import shutil
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier 

warnings.filterwarnings("ignore")

# 1. Load Raw Dataset
try:
    df = pd.read_csv("career_dataset.csv")
    print(f"Loaded dataset with {len(df)} records.")
except FileNotFoundError:
    print("Error: career_dataset.csv not found!")
    exit(1)

df.dropna(how='all', inplace=True) 

# 2. Preprocessing & Cleaning
X = df.drop("job_role", axis=1)
y = df["job_role"]

# Fill missing numericals
for col in ["cgpa", "graduation_year"]:
    mean_val = X[col].mean()
    X[col] = X[col].fillna(mean_val)

# Fill missing categoricals
for col in ["degree", "specialization", "certifications", "internship_experience"]:
    mode_val = X[col].mode()[0]
    X[col] = X[col].fillna(mode_val)

# Standardize Internship (Yes/No -> 1/0)
def clean_internship(val):
    return 1 if str(val).lower() in ['true', '1', 'yes'] else 0

X["internship_experience"] = X["internship_experience"].apply(clean_internship)

# Scale Numericals
scaler = StandardScaler()
X[["cgpa", "graduation_year"]] = scaler.fit_transform(X[["cgpa", "graduation_year"]])

# 3. Intelligent Skills Parsing (Noise Reduction)
X["skills"] = X["skills"].fillna("None")
STOP_WORDS = {"communication", "problem solving", "critical thinking", "teamwork", "leadership"}

def parse_skills(s):
    if pd.isna(s) or s == "None" or s == "":
        return []
    skills = [skill.strip() for skill in str(s).split(",")]
    valid_skills = [sk for sk in skills if sk.lower() not in STOP_WORDS]
    return valid_skills

# Apply parsing
cleaned_skills = X["skills"].apply(parse_skills)

# Create MultiLabelBinarizer
mlb = MultiLabelBinarizer()
skills_encoded = pd.DataFrame(mlb.fit_transform(cleaned_skills), columns=mlb.classes_, index=X.index)

# Drop raw skills and join encoded ones
X = pd.concat([X.drop("skills", axis=1), skills_encoded], axis=1)

# 4. Encode Categorical Features
label_encoders = {}
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
label_encoders["job_role"] = target_encoder

# Categorical columns to encode
cat_cols = ["degree", "specialization", "certifications"]
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# 6. Train XGBoost Model 
print("⏳ Training XGBoost Model...")
model = XGBClassifier(
    objective='multi:softprob',  
    n_estimators=300,            
    learning_rate=0.05,          
    max_depth=6,               
    subsample=0.8,               
    random_state=42,
    n_jobs=-1                    
)

model.fit(X_train, y_train)

# 7. Evaluation
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)
weighted_f1 = f1_score(y_test, test_preds, average='weighted')
print(f"\n📊 Training Accuracy: {train_acc:.4f}")
print(f"🚀 Test Accuracy:     {test_acc:.4f}")
print(f"⚖️ Weighted F1 Score: {weighted_f1:.4f}")
print("\n🔍 Classification Report:")
all_labels = range(len(target_encoder.classes_))
report = classification_report(
    y_test, 
    test_preds,
    labels=all_labels, 
    target_names=target_encoder.classes_,
    zero_division=0 
)
print(report)


try:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join("backups", timestamp) 
    # Create backup directory if it doesn't exist
    if not os.path.exists("backups"):
        os.makedirs("backups")      
    # Check if artifacts already exist and move them to backup folder
    artifacts_to_backup = [
        "career_model.pkl", "label_encoders.pkl", "scaler.pkl", 
        "skills_mlb.pkl", "feature_names.pkl", "metadata.pkl"
    ]
    
    # Only create a specific timestamp folder if we actually find files to backup
    files_found = [f for f in artifacts_to_backup if os.path.exists(f)]
    
    if files_found:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"📂 Backing up existing models to: {backup_dir}")
        for filename in files_found:
            shutil.copy(filename, os.path.join(backup_dir, filename))

    # Save new artifacts
    with open("career_model.pkl", "wb") as f: pickle.dump(model, f)
    with open("label_encoders.pkl", "wb") as f: pickle.dump(label_encoders, f)
    with open("scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    with open("skills_mlb.pkl", "wb") as f: pickle.dump(mlb, f)
    with open("feature_names.pkl", "wb") as f: pickle.dump(list(X.columns), f)
    with open("feature_selector.pkl", "wb") as f: pickle.dump(None, f)
    
    # Metadata Generation
    raw_df = pd.read_csv("career_dataset.csv")
    degree_map = raw_df.groupby('degree')['specialization'].unique().apply(list).to_dict()
    
    def get_unique_certs(series):
        certs = set(str(x).strip() for x in series.unique())
        for bad_val in ["None", "nan", "NaN", "[object Object]"]:
            if bad_val in certs: certs.remove(bad_val)
        return sorted(list(certs))
        
    cert_map = raw_df.groupby('specialization')['certifications'].apply(get_unique_certs).to_dict()
    metadata = {
        "degree_map": degree_map,
        "cert_map": cert_map,
        "skills": list(mlb.classes_)
    }
    
    with open("metadata.pkl", "wb") as f: pickle.dump(metadata, f)     
    print("All artifacts saved successfully!")
    
except Exception as e:
    print(f"Error saving artifacts: {e}")