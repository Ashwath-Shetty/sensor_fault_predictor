import joblib
import uvicorn
from fastapi import FastAPI,Request
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List

class ClientData(BaseModel):
    '''
    json Data i/p for the API call.
    '''
    sensor1: float
    sensor2: float
    sensor3: float
    sensor4: float


app = FastAPI()
@app.get('/')
def get_root():
    return {'message': 'Welcome to the fault detection API'}

@app.post("/is-fault")
async  def predict_fraud(item :ClientData):
  '''
  objective: final prediction api.
  return: returns isFault True/False.
  '''
  # Getting the JSON from the body of the request
  h=item.dict()
  col=['sensor1','sensor2','sensor3','sensor4']
  data=pd.DataFrame([h],columns=col)
  scale=joblib.load("./exports/scale.joblib", mmap_mode=None)
  final_model=joblib.load("./exports/model_rf.joblib", mmap_mode=None)
  df=scale.transform(data)
  model=joblib.load("./models/model2.joblib", mmap_mode=None)
  pred = model.predict(df)[0]
  if pred==0:
    return {'isFault':False}
  else:
    return {'isFault':True}