from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np
 
 
def compute_rolling_std(X_df, feature, time_window, center=True):
    name = "_".join([feature, time_window, "std", str(center)])
    X_df[name] = X_df[feature].rolling(time_window, center=center).std()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df
 
def compute_rolling_mean(X_df, feature, time_window, center=True):
    name = "_".join([feature, time_window, "mean", str(center)])
    X_df[name] = X_df[feature].rolling(time_window, center=center).mean()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df
 
def compute_rolling_median(X_df, feature, time_window, center=True):
    name = "_".join([feature, time_window, "median", str(center)])
    X_df[name] = X_df[feature].rolling(time_window, center=center).median()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df
 
def compute_rolling_min(X_df, feature, time_window, center=True):
    name = "_".join([feature, time_window, "min", str(center)])
    X_df[name] = X_df[feature].rolling(time_window, center=center).min()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df
 
def compute_rolling_max(X_df, feature, time_window, center=True):
    name = "_".join([feature, time_window, "max", str(center)])
    X_df[name] = X_df[feature].rolling(time_window, center=center).max()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df
 
def compute_rolling_skew(X_df, feature, time_window, center=True):
    name = "_".join([feature, time_window, "skew", str(center)])
    X_df[name] = X_df[feature].rolling(time_window, center=center).skew()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df
 
def compute_rolling_diff(X, feat, periods):
    name = "_".join([feat, str(periods)])
    X[name] = X[feat].pct_change(periods=periods)
    X[name] = X[name].ffill().bfill()
    X[name] = X[name].astype(X[feat].dtype)
    return X
 
def clip_column(X_df, column, min, max):
    X_df[column] = X_df[column].clip(min, max)
    return X_df
 
 
def smoothingroll2(y, factor):
    kernel = 10*[1] + 10*[3] + 15*[5] + 10*[3] + 10*[1]
    kernel = (kernel/np.sum(kernel))
    y[:,0] = np.convolve(y[:,0], kernel, 'same')
    y[:,1] = np.convolve(y[:,1], kernel, 'same')
    return y
 
class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        return self
 
    def transform(self, X):
        X = clip_column(X, 'Beta', 0, 100)
        X = clip_column(X, "B", 0, 100)
        X = clip_column(X, "RmsBob", 0, 2)
        X = clip_column(X, "Vth", 0, 350)
        X = clip_column(X, "V", 0, 2000)
        X = clip_column(X, "Vx", -1000, 1000)
        X = clip_column(X, "Range F 13", 0, 10**9)
        X = clip_column(X, "Pdyn", 0, 5e-13)
 
        Cols = ["B", "Beta", "RmsBob", "V", "Vth", "Range F 13", "Pdyn"]
 
        X = X.drop(columns=[col for col in X if col not in Cols])
 
        X = compute_rolling_skew(X, 'Beta', '6h', True)
        X = compute_rolling_skew(X, 'B', '6h', True)
        X = compute_rolling_skew(X, 'RmsBob', '6h', True)
 
        X = compute_rolling_mean(X, "Pdyn", "2h", True)
        X = compute_rolling_mean(X, "Pdyn", "6h", True)
        X = compute_rolling_mean(X, "Pdyn", "12h", True)
        X = compute_rolling_mean(X, "Pdyn", "24h", True)
        X = compute_rolling_median(X, "Pdyn", "3h", True)
        X = compute_rolling_median(X, "Pdyn", "6h", True)
        X = compute_rolling_median(X, "Pdyn", "12h", True)
        X = compute_rolling_median(X, "Pdyn", "24h", True)
 
        X = compute_rolling_std(X, "Pdyn", "2h", True)
        X = compute_rolling_std(X, "Pdyn", "6h", True)
        X = compute_rolling_std(X, "Pdyn", "12h", True)
        X = compute_rolling_std(X, "Pdyn", "24h", False)
 
        X = compute_rolling_mean(X, "B", "24h", True)

        X = compute_rolling_std(X, "B", "24h", True)
        X = compute_rolling_std(X, 'B', "48h", True)
        X = compute_rolling_std(X, 'B', "72h", True)
        X = compute_rolling_std(X, "B", "24h", False)
        X = compute_rolling_std(X, 'B', "48h", False)
        X = compute_rolling_std(X, 'B', "72h", False)
 
        X = compute_rolling_median(X, "B", "12h", True)
        X = compute_rolling_median(X, "B", "24h", True)
        X = compute_rolling_median(X, "B", "12h", False)
        X = compute_rolling_median(X, "B", "24h", False)
 
        X = compute_rolling_mean(X, "RmsBob", "6h", True)
        X = compute_rolling_mean(X, "RmsBob", "12h", True)
        X = compute_rolling_mean(X, "RmsBob", "18h", True)
        X = compute_rolling_mean(X, "RmsBob", "24h", True)
        X = compute_rolling_mean(X, "RmsBob", "48h", True)
        X = compute_rolling_mean(X, "RmsBob", "12h", False)
        X = compute_rolling_mean(X, "RmsBob", "24h", False)
 
        X = compute_rolling_median(X, "RmsBob", "24h", True)
        X = compute_rolling_median(X, "RmsBob", "24h", False)
        X = compute_rolling_median(X, "RmsBob", "12h", True)
        X = compute_rolling_median(X, "RmsBob", "12h", False)
 
        X = compute_rolling_mean(X, "Beta", "2h", True)
        X = compute_rolling_mean(X, "Beta", "1h", True)
        X = compute_rolling_mean(X, "Beta", "12h", True)
        X = compute_rolling_mean(X, "Beta", "12h", False)
        X = compute_rolling_mean(X, "Beta", "6h", True)
        X = compute_rolling_mean(X, "Beta", "6h", False)
        X = compute_rolling_mean(X, "Beta", "24h", True)
        X = compute_rolling_mean(X, "Beta", "24h", False)
        X = compute_rolling_mean(X, "Beta", "3h", True)
        X = compute_rolling_mean(X, "Beta", "4h", True)
        X = compute_rolling_mean(X, "Beta", "5h", True)
 
        X = compute_rolling_median(X, "Beta", "24h", True)
        X = compute_rolling_median(X, "Beta", "24h", False)
        X = compute_rolling_median(X, "Beta", "12h", True)
        X = compute_rolling_median(X, "Beta", "12h", False)

        #Create a copy to avoid DataFrame fragmentation
        X = X.copy()
 
        X = compute_rolling_diff(X, 'B_12h_median_True', 48)
        X = compute_rolling_diff(X, 'Beta_12h_median_True', 48)
        X = compute_rolling_diff(X, 'RmsBob_12h_median_True', 48)
        X = compute_rolling_diff(X, 'B_12h_median_True', 96)
        X = compute_rolling_diff(X, 'Beta_12h_median_True', 96)
        X = compute_rolling_diff(X, 'RmsBob_12h_median_True', 96)
        X = compute_rolling_diff(X, 'B_12h_median_True', 192)
        X = compute_rolling_diff(X, 'Beta_12h_median_True', 192)
        X = compute_rolling_diff(X, 'RmsBob_12h_median_True', 192)
 
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', 6)
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', 12)
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', 18)
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', 24)
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', 30)
        X = compute_rolling_diff(X, 'Beta_1h_mean_True', 36)
   
 
        X = compute_rolling_mean(X, "B", "1h", True)
        X = compute_rolling_diff(X, 'B_1h_mean_True', 6)
        X = compute_rolling_diff(X, 'B_1h_mean_True', 12)
        X = compute_rolling_diff(X, 'B_1h_mean_True', 18)
        X = compute_rolling_diff(X, 'B_1h_mean_True', 24)
        X = compute_rolling_diff(X, 'B_1h_mean_True', 30)
        X = compute_rolling_diff(X, 'B_1h_mean_True', 36)
 
        X = compute_rolling_mean(X, "RmsBob", "1h", True)
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', 6)
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', 12)
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', 18)
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', 24)
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', 30)
        X = compute_rolling_diff(X, 'RmsBob_1h_mean_True', 36)
 
        X = compute_rolling_mean(X, "V", "1h", True)
        X = compute_rolling_mean(X, "V", "6h", True)
        X = compute_rolling_mean(X, "V", "12h", True)
        X = compute_rolling_mean(X, "V", "24h", True)
        X = compute_rolling_std(X, "V", "1h", True)
        X = compute_rolling_std(X, "V", "6h", True)
        X = compute_rolling_std(X, "V", "12h", True)
        X = compute_rolling_std(X, "V", "24h", True)
 
        return X
    
class CustomClf(BaseEstimator):
    def __init__(self):
        self._estimator_type = "classifier"
        self.estimator = VotingClassifier(
            [
            ('LR', LogisticRegression(max_iter=5000,
                                    class_weight={0:1, 1:1.6},
                                    n_jobs=1,
                                    tol=0.001,
                                    penalty='l2',
                                    C=1,
                                    solver='liblinear')),
            ('LGBM', LGBMClassifier(objective='binary',
                                num_leaves=20,
                                min_split_gain=0.,
                                max_depth=15,
                                learning_rate=0.02,
                                n_estimators=500,
                                class_weight={0:1, 1:2},
                                reg_lambda=0.5,
                                ))
            ],
            voting='soft',
            n_jobs=-1,
            weights=[1,3]
        )
        
 
    def fit(self, X, y):
        return self.estimator.fit(X, y=y)
    
    def predict(self, X, y):
        y = self.estimator.predict(X)
        #y = smoothingp(y, 6)
        return y
    
    def predict_proba(self, X):
        y = self.estimator.predict_proba(X)
        y = smoothingroll2(y, 30)
        return y
    
    def classes_(self):
        return self.estimator.classes_
    
    def set_params(self, **params):
        return self.estimator.set_params(**params)
    
 
def get_estimator():
 
    feature_extractor = FeatureExtractor()
    classifier = CustomClf()
    scaler = StandardScaler()
    
 
    pipe = make_pipeline(
        feature_extractor,
        scaler,
        classifier)
    return pipe