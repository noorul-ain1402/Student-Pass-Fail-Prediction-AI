import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("student_data.csv")

data["Result"] = data["Result"].map({
    "Fail": 0,
    "Pass": 1
})

X = data[["StudyHours", "Attendance"]]
y = data["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy * 100, "%")

study_hours = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance: "))

new_data = pd.DataFrame({
    "StudyHours": [study_hours],
    "Attendance": [attendance]
})

result = model.predict(new_data)

if result[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")