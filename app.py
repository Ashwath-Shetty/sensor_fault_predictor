import streamlit as st
import joblib
import pandas as pd

st.title('Sensor fault prediction')
st.write("Since i'm using the free cloud prediction app will take some time to predict, please don't click predict button repeatedly")

sensor1= st.number_input("Enter the sensor1 value")
sensor2= st.number_input("Enter the sensor2 value")
sensor3= st.number_input("Enter the sensor3 value")
sensor4= st.number_input("Enter the sensor4 value")

scale=joblib.load("./exports/scale.joblib", mmap_mode=None)
final_model=joblib.load("./exports/model_rf.joblib", mmap_mode=None)
col=['sensor1','sensor2','sensor3','sensor4']

val=[[sensor1,sensor2,sensor3,sensor4]]
data=pd.DataFrame(val, columns=col)
df=scale.transform(data)
print(df)


if (st.button('predict')):
    op=final_model.predict(df)
    print("---",op)
    if op[0]==1:
        st.write("prediction result-------> faulty")
    else:
        st.write("prediction result-------> healthy")


