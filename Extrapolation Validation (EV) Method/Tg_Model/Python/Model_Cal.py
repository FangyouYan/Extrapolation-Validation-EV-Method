import os
import numpy as np
import warnings
import sys
import pandas as pd
import joblib

'''Process of data standardization'''
def Standardization(x_train_, x_):
    x_mean = np.mean(x_train_, 0)
    x_std = np.std(x_train_, 0)
    x = (x_ - x_mean) / x_std
    return x


'''Calculation of squared correlation coefficient (R2)'''
def r2_(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    r2 = np.sum(np.multiply(x - x_mean, y - y_mean)) ** 2 / np.sum(np.power(x - x_mean, 2)) / np.sum(
        np.power(y - y_mean, 2))
    return r2


'''Calculation of root mean squared error (RMSE)'''
def _rmse(x, y):
    rmse = np.sqrt(np.mean(np.multiply(x - y, x - y)))
    return rmse


warnings.filterwarnings("ignore")
cd_project = os.path.dirname(os.path.dirname(os.getcwd()))
cd_dataset = f'{cd_project}/Tg_Model/Dataset/'
cd_model = f'{cd_project}/Tg_Model/Model/'

x_train_all = pd.read_excel(cd_dataset + r"dataset_Tg_x_train.xlsx", sheet_name='Sheet1')
y_train_all = pd.read_excel(cd_dataset + r"dataset_Tg_y_train.xlsx", sheet_name='Sheet1')
x_test_all = pd.read_excel(cd_dataset + r"dataset_Tg_x_test.xlsx", sheet_name='Sheet1')
y_test_all = pd.read_excel(cd_dataset + r"dataset_Tg_y_test.xlsx", sheet_name='Sheet1')

N_ = int(np.shape(y_train_all.values)[0])

x_train = np.array(x_train_all.values[:, :])
x_test = np.array(x_test_all.values[:, :])
y_train = np.array(y_train_all.values[:, :]).flatten()
y_test = np.array(y_test_all.values[:, :]).flatten()

x_train_sta = Standardization(x_train, x_train)
x_test_sta = Standardization(x_train, x_test)

'''MLR model'''
mlr_ = joblib.load(cd_model + "mlr_best.pkl")
pred_y_train_mlr = mlr_.predict(x_train_sta)
rmse_train_mlr = _rmse(np.array(y_train), np.array(pred_y_train_mlr))

'''Lasso model'''
lasso_ = joblib.load(cd_model + "Lasso_best.pkl")
pred_y_train_lasso = lasso_.predict(x_train_sta)
rmse_train_lasso = _rmse(np.array(y_train), np.array(pred_y_train_lasso))

'''SVM model'''
svm_ = joblib.load(cd_model + "svm_best.pkl")
pred_y_train_svm = svm_.predict(x_train_sta)
rmse_train_svm = _rmse(np.array(y_train), np.array(pred_y_train_svm))

'''Adaboost model'''
adaboost_ = joblib.load(cd_model + "adaboost_best.pkl")
pred_y_train_adaboost = adaboost_.predict(x_train_sta)
rmse_train_adaboost = _rmse(np.array(y_train), np.array(pred_y_train_adaboost))

'''Ridge model'''
ridge_ = joblib.load(cd_model + "Ridge_best.pkl")
pred_y_train_ridge = ridge_.predict(x_train_sta)
rmse_train_ridge = _rmse(np.array(y_train), np.array(pred_y_train_ridge))

'''RF model'''
rf_ = joblib.load(cd_model + "rf_best.pkl")
pred_y_train_rf = rf_.predict(x_train_sta)
rmse_train_rf = _rmse(np.array(y_train), np.array(pred_y_train_rf))

'''KNN model'''
knn_ = joblib.load(cd_model + "knn_best.pkl")
pred_y_train_knn = knn_.predict(x_train_sta)
rmse_train_knn = _rmse(np.array(y_train), np.array(pred_y_train_knn))

'''XGBoost Model'''
xgboost_ = joblib.load(cd_model + "xgboost_best.pkl")
pred_y_train_xgboost = xgboost_.predict(x_train_sta)
rmse_train_xgboost = _rmse(np.array(y_train), np.array(pred_y_train_xgboost))

'''GBDT Model'''
gbdt_ = joblib.load(cd_model + "gbdt_best.pkl")
pred_y_train_gbdt = gbdt_.predict(x_train_sta)
rmse_train_gbdt = _rmse(np.array(y_train), np.array(pred_y_train_gbdt))

'''MLP Model'''
mlp_ = joblib.load(cd_model + "mlp_best.pkl")
pred_y_train_mlp = mlp_.predict(x_train_sta)
rmse_train_mlp = _rmse(np.array(y_train), np.array(pred_y_train_mlp))

'''GPR Model'''
gpr_ = joblib.load(cd_model + "gpr_best.pkl")
pred_y_train_gpr = gpr_.predict(x_train_sta)
rmse_train_gpr = _rmse(np.array(y_train), np.array(pred_y_train_gpr))

