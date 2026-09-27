import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Teaching dataset (same as your Colab notebook)
data = pd.DataFrame({
    "max_temp":[38,40,43,45,36,42,39,44,41,46,37,43,45,39,41,47,38,44,42,46],
    "min_temp":[27,28,30,31,26,29,27,30,28,32,26,30,31,27,29,33,27,31,29,32],
    "humidity":[65,60,55,50,70,58,68,52,62,48,72,56,49,66,59,45,69,51,57,47],
    "wind_speed":[12,10,8,7,15,9,11,8,10,6,14,8,7,12,9,5,13,7,9,6],
    "previous_temp":[37,39,42,44,35,41,38,43,40,45,36,42,44,38,40,46,37,43,41,45],
    "heatwave":[0,0,1,1,0,1,0,1,0,1,0,1,1,0,0,1,0,1,1,1]
})

X = data[["max_temp","min_temp","humidity","wind_speed","previous_temp"]]
y = data["heatwave"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "model/heatwave_model.pkl")
print("Model saved to model/heatwave_model.pkl")