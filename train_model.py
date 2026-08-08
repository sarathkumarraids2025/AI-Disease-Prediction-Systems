import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv("DiseaseAndSymptoms.csv")
df.fillna("", inplace=True)

# --------------------------
# Get All Unique Symptoms
# --------------------------
all_symptoms = set()

for col in df.columns[1:]:
    all_symptoms.update(df[col].unique())

all_symptoms.discard("")

all_symptoms = sorted(list(all_symptoms))

print("Total Symptoms:", len(all_symptoms))

# --------------------------
# Create Binary Matrix
# --------------------------
X = pd.DataFrame(0, index=df.index, columns=all_symptoms)

for i, row in df.iterrows():
    for col in df.columns[1:]:
        symptom = row[col]
        if symptom != "":
            X.loc[i, symptom] = 1

# --------------------------
# Encode Disease
# --------------------------
le = LabelEncoder()
y = le.fit_transform(df["Disease"])

# --------------------------
# Train Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------
# Train Model
# --------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------
# Accuracy
# --------------------------
pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, pred))

# --------------------------
# Save Files
# --------------------------
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(all_symptoms, "symptom_list.pkl")

print("model.pkl Saved")
print("label_encoder.pkl Saved")
print("symptom_list.pkl Saved")