from fastapi import FastAPI
import joblib
import uvicorn
from pydantic import BaseModel

diabetes_app = FastAPI()
log_model = joblib.load('log_model.pkl')
scaler = joblib.load('scaler.pkl')
tree_model = joblib.load('tree_model.pkl')

class DiabetesSchema(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


@diabetes_app.post('/predict/')
async def predict(diabetes: DiabetesSchema):
    diabetes_dict = diabetes.dict()

    features = list(diabetes_dict.values())
    scaled_data = scaler.transform([features])
    pred = log_model.predict(scaled_data)[0]
    prob = log_model.predict_proba(scaled_data)[0][1]

    return {'approved': bool(pred), 'probability': round(prob, 2)}

if __name__ == '__main__':
    uvicorn.run(diabetes_app, host='127.0.0.1', port=8002)
