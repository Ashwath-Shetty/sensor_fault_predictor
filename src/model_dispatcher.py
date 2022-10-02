from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

models = {
 "dt": DecisionTreeClassifier(
    min_samples_split=7,min_samples_leaf=5,max_depth=None,random_state=42
    ),
 "rf": RandomForestClassifier(
    random_state=0,n_estimators=100,min_samples_split=2,min_samples_leaf=1,
    max_features='sqrt', max_depth=10,bootstrap=False
    ),
"clf_xgb": XGBClassifier(max_depth=8,
    learning_rate=0.1,
    n_estimators=1000,
    verbosity=0,
    silent=None,
    objective='binary:logistic',
    booster='gbtree',
    n_jobs=-1,
    nthread=None,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.7,
    colsample_bytree=1,
    colsample_bylevel=1,
    colsample_bynode=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    random_state=0,
    seed=None,),



}