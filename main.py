import pandas as pd
import joblib

df = pd.read_csv("loan.csv", encoding="latin1")

# ---------------- CLEANING ---------------- #

# Drop sparse columns
threshold = len(df) * 0.5
df = df.dropna(axis=1, thresh=threshold)

# Target creation
def label_default(status):
    if status in ["Charged Off", "Default"]:
        return 1
    elif status == "Fully Paid":
        return 0
    else:
        return None

df["target"] = df["loan_status"].apply(label_default)
df = df.dropna(subset=["target"])

# Drop useless + leakage columns
drop_cols = ["loan_status", "id", "member_id", "url"]
leakage_cols = [
    "recoveries","collection_recovery_fee","last_pymnt_amnt",
    "total_rec_prncp","total_rec_int","out_prncp",
    "out_prncp_inv","total_pymnt","total_pymnt_inv",
    "total_rec_late_fee","last_paymnt_d",
    "last_credit_pull_d","zip_code","title","emp_title"
]

df = df.drop(columns=drop_cols + leakage_cols, errors="ignore")

# ---------------- FEATURE FIXES ---------------- #

# Convert term
# Convert term safely
if 'term' in df.columns:
    df['term'] = df['term'].astype(str).str.extract('(\d+)')[0]
    df['term'] = pd.to_numeric(df['term'], errors='coerce')

# Convert interest rate safely
if 'int_rate' in df.columns:
    df['int_rate'] = df['int_rate'].astype(str).str.replace('%','')
    df['int_rate'] = pd.to_numeric(df['int_rate'], errors='coerce')

# Clean DTI
if 'dti' in df.columns:
    df['dti'] = df['dti'].replace(-1, None)
    df['dti'] = df['dti'].clip(upper=100)  # cap unrealistic values

# Loan to Income ratio
df['annual_inc'] = df['annual_inc'].replace(0, 1)  # avoid division crash
df['loan_to_income'] = df['loan_amnt'] / df['annual_inc']
df['loan_to_income'] = df['loan_to_income'].clip(upper=100)
# ---------------- SPLIT ---------------- #

selected_features = [
    "loan_amnt",
    "annual_inc",
    "int_rate",
    "term",
    "grade",
    "dti"
]

X = df[selected_features].copy()

# Derived feature
X['loan_to_income'] = X['loan_amnt'] / X['annual_inc']
y = df["target"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- PIPELINE ---------------- #

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Handle imbalance safely
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = neg / pos if pos != 0 else 1

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    reg_alpha = 1,
    reg_lambda = 2
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# ---------------- TRAIN ---------------- #

pipeline.fit(X_train, y_train)

# ---------------- EVALUATE ---------------- #

from sklearn.metrics import roc_auc_score

y_prob = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC:", auc)

# ---------------- SAVE ---------------- #

joblib.dump(pipeline, "loan_pipeline.pkl")
print("Pipeline saved successfully ✅")
# print(y_prob.min(), y_prob.max())

print(df.columns)
print(df['dti'].head())
print(df['dti'].describe())
print(y_prob.min(), y_prob.max())