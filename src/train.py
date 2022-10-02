import config
import model_dispatcher
import numpy as np 
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib


def loss(actual,pred):
    '''
    objective: f1 score calculation.
    prediction: precited values
    actual: actual values
    return: f1 score for the model.
    '''
    return f1_score(actual, pred)


def data_read(faulty_file_path, healthy_file_path):
    '''
    objective: reading and combining the data.
    return: final data frame(faulty+healhy)
    '''
    col=['sensor1','sensor2','sensor3','sensor4']
    df_faulty=pd.read_csv(faulty_file_path,header=None,names=col)
    df_faulty['y']=1
    
    df_healthy=pd.read_csv(healthy_file_path,header=None,names=col)
    df_healthy['y']=0
    
    combined_df=pd.concat([df_faulty,df_healthy],axis=0,ignore_index=True)
    return combined_df



def preprocess(combined_df):
    '''
    objective: preprocess the data frame for modelling
    combined_df: combined dataframe(faulty+healthy)
    return: X & Y - train, test data set.
    
    '''
    col_x=combined_df.columns[~combined_df.columns.isin(['y'])]
    X=combined_df[col_x]
    y=combined_df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=40)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    joblib.dump(scaler, "./exports/scale.joblib", 9) # export the scaler for inference
    
    return X_train,X_test, y_train,y_test


def validation(X_train,y_train,model):
    '''
    objective: validation loop to validate the model performace.
    X_train: independent variables of training data.
    y_train: dependent/target variable of training data.
    
    '''
    print("---------",model,"------------")
    kfold=StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    train_los=0
    test_los=0
    for fold,(train_idx, val_idx) in enumerate(kfold.split(X_train,y_train)):
        train_x=X_train[train_idx]
        train_y=y_train[train_idx]
        test_x=X_train[val_idx]
        test_y=y_train[val_idx]
        model.fit(train_x,train_y)
        pred_test=model.predict(test_x)
        pred_train=model.predict(train_x)
        train_los+=loss(train_y,pred_train)
        test_los+=loss(test_y,pred_test)
        print(fold,'train loss------->',loss(train_y,pred_train))
        print(fold,'test loss------->',loss(test_y,pred_test))
    print('train',train_los/5)
    print("test",test_los/5)
    return "----------end--------------"

def train_test(model, X_train, y_train, X_test,y_test):
    '''
    objective: model performances on train and test set.
    X_train: independent variables of training data.
    y_train: dependent/target variable of training data.
    X_test: independent variables of test data.
    y_test: dependent/target variable of test data.
    
    '''
    print("----------", model,"-----------")
    model.fit(X_train, y_train)
    pred=model.predict(X_train)
    print("train_accuracy",f1_score(y_train, pred))
    pred_test=model.predict(X_test)
    print("test_accuracy",f1_score(y_test, pred_test))

def main():
    combined_df=data_read(config.faulty_file_path, config.healthy_file_path)
    X_train,X_test, y_train,y_test=preprocess(combined_df)
    rf=model_dispatcher.models['rf']

    validation(X_train,y_train,rf) #validation set result for the model
    #train and test for the model
    rf.fit(X_train, y_train)
    pred_train=rf.predict(X_train)
    pred_test=rf.predict(X_test)
    print("train accuracy",loss(pred_train,y_train))
    print("test accuracy",loss(pred_test,y_test))

    joblib.dump(rf, "./exports/model_rf.joblib", 9) #model export

if __name__=='__main__':
    main()










