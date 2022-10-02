# Sensor fault Predictor
## Quick Links
<br><li> Live Demo ---> https://ashwath-shetty-sensor-fault-predictor-app-14nv4z.streamlitapp.com/
<br><li> Documentation ---> https://docs.google.com/document/d/1-5_IPQ8sZ2q4Bz-_Ujh3WedVi0-YIum22EKCr8oD8ZA/edit#
<br><li> Jupyter Notebook --> https://www.kaggle.com/code/ashwathshetty/samsung-assignment/notebook

## Problem Statement
Design and develop an ML / DL based approach to identify whether the given data is healthy
or faulty for the sensor data of the industrial gearbox machine.

## Folder Structure

.
├── data                     # replace with original Data-> faulty and healthy data files.
├── exports                   # exported models and scaler object.
├── notebooks                # Jupyter notebooks.
├── src                     # Source files 
│    ├── config.py          # configuration files
│    ├── train.py           # training code
│    ├── model_dispatcher.py  # has all the used models details
├── app.py                   # deployed main app code for streamlit
├── requirements.txt         # requirements
├── Dockerfile               # to Dockerize
├── fast_api.py              # FastAPI for deployment
├── api_test.py              # api test file for fast api
└── README.md


<b>How can i train the model?</b>
<br>step 1: clone the repository using https://github.com/Ashwath-Shetty/sensor_fault_predictor.git 
<br>step 2: add the data inside data folder and change the files path to your path in the config file(which is inside src folder).
<br>step 3: go to command prompt and navigate to the project folder(cd project/folder/path)
<br>step 4: enter pip install - r requirements.txt
<br>step 5 : navigate to src folder (cd path/src)
<br>step 6: type python train.py and hit enter.
<br>after training model will be saved in export folder.

<b>How to test the App/API on local host?</b>
<br> if you just want to check the deployed app, you can skip this and check the next section
<br><li>to test streamlit UI based application -> go to command prompt and navigate to project folder and enter streamlit run app.py
<br><li> to check FastAPI go to command prompt and navigate to project folder and enter uvicorn fast_api:app --reload
  <br> go to browser and visit http://127.0.0.1:8000/docs

## Testing the Deployed FastAPI (in the future if deployed on any cloud)
<br> <li>all you need is api_test.py file and python installed in the system. 
<br> <li>just run the file using python api_test.py and you will get the response printed on the console.
<br><li> replace the url with your url(https://your url/is-fault) (you can check the json format in api_test.py)
<br> if you want to change the data go to line 11 inside api_test.py where you can see json={} and change the data you want to.

## Deployment
<b>Streamlit Deployment</b>
<li>Application has been deployed to streamlit cloud and connected github to streamlit for continuous deployment. every commit to the github will automatically deploy to the streamlit.

## Tools and languages used
1. Python
2. Scikit learn
3. pandas
4. numpy
5. joblib
6. Randomized Search CV
7. streamlit for development
8. Various ML Modelling and preprocessing techniques.
9. streamlit cloud for deployment
10. Fast API for API Development.
11. Heroku for Fast API deployment.
12. uvicorn and gunicorn.

## Reach out to me here
check out my [portfolio](https://ashwathshetty.netlify.app/)

