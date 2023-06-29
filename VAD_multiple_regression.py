import pandas as pd
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
from bayes_opt import BayesianOptimization


# Read EMObanck and separate VAD
df = pd.read_csv("Assinging_VAD_scores_BERT\DataSet\emobank.csv")

V, A, D = df["V"], df["A"], df["D"]


# Define the objective function for Bayesian Optimization
class XGB_evaluate: 
    # Define the parameter bounds for Bayesian Optimization
    param_bounds = {
        'max_depth': (2, 9),
        'learning_rate': (0.01, 0.2),
        'n_estimators': (100, 300),
        'subsample': (0.4, 0.9),
        'colsample_bytree': (0.4, 0.9),
    }
    
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def xgb_evaluate(self, max_depth, learning_rate, n_estimators, subsample, colsample_bytree):
        model = xgb.XGBRegressor(
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            n_estimators=int(n_estimators),
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method="gpu_hist",
            gpu_id=0
        )
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_train)
        mse = mean_squared_error(self.y_train, y_pred)
        return -mse  # Minimize the negative MSE

    # Run Bayesian Optimization
    def __call__(self):
        xgb_bo = BayesianOptimization(self.xgb_evaluate, self.param_bounds)
        xgb_bo.maximize(init_points=50, n_iter=200) #Bayesian Optimization's Hyper parameters
    
        return xgb_bo.max['params']


def Verify(y_test_D, y_predict_D):
    for i in range(495, 505):
        print("{:5.3f} | {:5.3f}".format(round(y_test_D[i], 4), round(y_predict_D[i], 4)))

    mse = mean_squared_error(y_test_D, y_predict_D)
    print("MSE: ", mse)
    print("\n===========================\n")


# ==============================================================================
# Predict D from V and A
X_Data = np.vstack((V, A))
X_Data = X_Data.transpose()
Y_Data = D

# Split data for Train and Test
X_train_VA, X_test_VA, y_train_D, y_test_D = train_test_split(X_Data, Y_Data, test_size=0.1, random_state=11)
y_train_D = y_train_D.reset_index(drop=True)
y_test_D = y_test_D.reset_index(drop=True)

# Train the XGBoost model with the best hyperparameters
# Best Hyper-parameter: target=-0.04171, colsample_bytree=0.5, learning_rate=0.1, max_depth=3, n_estimators=144.4, subsample=0.7, reg_alpha=0
XGB_model_VA_D = xgb.XGBRegressor(colsample_bytree=0.5, learning_rate=0.1, max_depth=3, n_estimators=144, subsample=0.7, reg_alpha=0)
XGB_model_VA_D.fit(X_train_VA, y_train_D)
# Predict
y_predict_D = XGB_model_VA_D.predict(X_test_VA)

# Verify
print("Predict D from V and A")
Verify(y_test_D, y_predict_D) # MSE: 0.0417

# ==============================================================================
# Predict A from V and D
X_Data = np.vstack((V, D))
X_Data = X_Data.transpose()
Y_Data = A

# Split data for Train and Test
X_train_VD, X_test_VD, y_train_A, y_test_A = train_test_split(X_Data, Y_Data, test_size=0.1, random_state=11)
y_train_A = y_train_A.reset_index(drop=True)
y_test_A = y_test_A.reset_index(drop=True)

# Train the XGBoost model with the best hyperparameters
# Best Hyper-parameter: target=-0.04449, colsample_bytree=0.9, learning_rate=0.2, max_depth=6.535, n_estimators=258.2, subsample=0.9, reg_alpha=0
XGB_model_VD_A = xgb.XGBRegressor(colsample_bytree=0.9, learning_rate=0.2, max_depth=6, n_estimators=258, subsample=0.9, reg_alpha=0)
XGB_model_VD_A.fit(X_train_VD, y_train_A)
# Predict
y_predict_A = XGB_model_VD_A.predict(X_test_VD)

# Verify
print("Predict A from V and D")
Verify(y_test_A, y_predict_A) # MSE: 0.0461

# ==============================================================================
# Predict V from D and A
X_Data = np.vstack((D, A))
X_Data = X_Data.transpose()
Y_Data = V

# Split data for Train and Test
X_train_DA, X_test_DA, y_train_V, y_test_V = train_test_split(X_Data, Y_Data, test_size=0.1, random_state=11)
y_train_V = y_train_V.reset_index(drop=True)
y_test_V = y_test_V.reset_index(drop=True)

# Train the XGBoost model with the best hyperparameters
# Best Hyper-parameter: target=-0.0924, colsample_bytree=0.4471, learning_rate=0.08324, max_depth=10.1, n_estimators=348.6, subsample=0.8342, reg_alpha=0
XGB_model_DA_V = xgb.XGBRegressor(colsample_bytree=0.4, learning_rate=0.08324, max_depth=10, n_estimators=348, subsample=0.8342, reg_alpha=0)
XGB_model_DA_V.fit(X_train_DA, y_train_V)

# Predict
y_predict_V = XGB_model_DA_V.predict(X_test_DA)

# Verify
print("Predict A from V and D")
Verify(y_test_V, y_predict_V) # MSE: 0.1060

# ==============================================================================
# Predict "the square of V's deviation" from D and A
X_Data = np.vstack((D, A))
X_Data = X_Data.transpose()

V_mean = np.mean(V)
Y_Data = (V - V_mean)**2 # the square of V's deviation

# Scale to 1.2~4.6
scaler = MinMaxScaler(feature_range=(1.2, 4.6))
Y_Data = scaler.fit_transform([[value] for value in Y_Data])
Y_Data = np.array(list(chain.from_iterable(Y_Data)))

# Split data for Train and Test
X_train_DA, X_test_DA, y_train_V_dev2, y_test_V_dev2 = train_test_split(X_Data, Y_Data, test_size=0.1, random_state=11)

# Get the best hyperparameters
best_params = XGB_evaluate(X_train_DA, y_train_V_dev2)()

# Train the final XGBoost model with the best hyperparameters
# Best Hyper-parameter: target=-0.03849, colsample_bytree=0.7876, learning_rate=0.0729, max_depth=8.951, n_estimators=275.7, subsample=0.8959, reg_alpha=0
XGB_model = xgb.XGBRegressor(
    max_depth=int(best_params['max_depth']),
    learning_rate=best_params['learning_rate'],
    n_estimators=int(best_params['n_estimators']),
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    tree_method="gpu_hist",
    gpu_id=0
)
XGB_model.fit(X_train_DA, y_train_V_dev2)

# Predict
y_predict_V_dev2 = XGB_model.predict(X_test_DA)

# Verify
print(f"Predict the square of V's deviation from D and A")
Verify(y_test_V_dev2, y_predict_V_dev2) #0.0473