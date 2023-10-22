import os
import numpy as np
import warnings
import pandas as pd
import joblib
import copy

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


'''Extrapolation validation for a model'''


def ev_cal(x_train, x_test, y_train, y_test, N_fit, j, cd_model):
    '''Evaluating by the performance on the training (EV) set and test (EV) sets from the model.'''
    names[f'{j}_'] = joblib.load(f'{cd_model}{j}_best.pkl')
    names[f'{j}_n'] = copy.deepcopy(names[f'{j}_'])
    pred_y_train_ = names[f'{j}_n'].predict(x_train)
    pred_y_test_ = names[f'{j}_n'].predict(x_test)

    pred_y_train = np.array(pred_y_train_)
    pred_y_test = np.array(pred_y_test_)

    rmse_train = _rmse(np.array(y_train), pred_y_train)
    rmse_test = _rmse(np.array(y_test), pred_y_test)

    r2_train = r2_(np.array(y_train), np.array(pred_y_train))
    r2_test = r2_(np.array(y_test), np.array(pred_y_test))

    N_train = int(np.shape(y_train)[0])
    N_test = int(np.shape(y_test)[0])

    pred_y_train_e_all = np.zeros((N_train, 1))
    pred_y_test_e_all = np.zeros((N_test, 1))

    '''Re-fitting the model using the training (EV) set.'''
    '''Re-fitting 50 times to reduce machine learning (ML) model randomization.'''
    for k in range(0, N_fit):
        names[f'{j}_n1'] = names[f'{j}_'].fit(x_train, y_train)

        pred_y_train_e = names[f'{j}_n1'].predict(x_train)
        pred_y_test_e = names[f'{j}_n1'].predict(x_test)

        pred_y_train_e_all = np.c_[pred_y_train_e_all, pred_y_train_e]
        pred_y_test_e_all = np.c_[pred_y_test_e_all, pred_y_test_e]

    pred_y_MP21_train_e = np.average(np.delete(pred_y_train_e_all, 0, axis=1), axis=1)
    pred_y_MP21_test_e = np.average(np.delete(pred_y_test_e_all, 0, axis=1), axis=1)

    '''Evaluating by the performance on the training (EV) set and test (EV) sets from re-fitting the model.'''
    rmse_train_e = _rmse(np.array(y_train), np.array(pred_y_MP21_train_e))
    rmse_test_e = _rmse(np.array(y_test), np.array(pred_y_MP21_test_e))

    try:
        r2_train_e = r2_(np.array(y_train), np.array(pred_y_MP21_train_e))
        r2_test_e = r2_(np.array(y_test), np.array(pred_y_MP21_test_e))
    except:
        r2_train_e = 0
        r2_test_e = 0

    print(rmse_train, rmse_test, r2_train, r2_test)
    print(rmse_train_e, rmse_test_e, r2_train_e, r2_test_e)

    return rmse_train, rmse_test, r2_train, r2_test, rmse_train_e, rmse_test_e, r2_train_e, r2_test_e


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    cd_project = os.path.dirname(os.path.dirname(os.getcwd()))
    cd_dataset = f'{cd_project}/Tg_Model/Dataset/'
    cd_model = f'{cd_project}/Tg_Model/Model/'
    cd_figure = f'{cd_project}/Tg_Model/Figure/'

    '''Read independent and dependent variables of the training set'''
    x_train_all = pd.read_excel(cd_dataset + r"dataset_Tg_x_train.xlsx", sheet_name='Sheet1')
    y_train_all = pd.read_excel(cd_dataset + r"dataset_Tg_y_train.xlsx", sheet_name='Sheet1')

    '''Number of dependent varibles'''
    N_ = int(np.shape(y_train_all.values)[0])

    '''Number of independent varibles'''
    Feature_N = int(np.shape(x_train_all)[1])

    x_train = x_train_all.values[:, :]
    y_train = y_train_all.values[:, :]

    x_train_sta = Standardization(x_train, x_train)

    '''Calculation of leverage value (h)'''
    XTX_inverse = np.linalg.inv(np.dot(x_train_sta.T, x_train_sta))
    h_train_sta_ = np.diagonal(np.dot(np.dot(x_train_sta, XTX_inverse), x_train_sta.T))
    h_train_sta = np.expand_dims(h_train_sta_, axis=1)

    xyh_train = np.concatenate((x_train_sta, h_train_sta, y_train), axis=1)

    headers = ['x_name',
               'rmse_train_forward', 'rmse_test_forward', 'r2_train_forward', 'r2_test_forward',
               'rmse_train_forward_ev', 'rmse_test_forward_ev', 'r2_train_forward_ev', 'r2_test_forward_ev',
               'rmse_train_backward', 'rmse_test_backward', 'r2_train_backward', 'r2_test_backward',
               'rmse_train_backward_ev', 'rmse_test_backward_ev', 'r2_train_backward_ev', 'r2_test_backward_ev']

    N_fit = 50

    Models = ['knn', 'svm', 'mlp', 'rf', 'ridge', 'lasso', 'mlr', 'adaboost', 'gpr', 'gbdt', 'xgboost']

    names = locals()

    for j in Models:
        names[f'result_{j}'] = []

        '''Each independent variable is serialized, and then the training and test sets are re-divided in 8:2.'''

        for i in range(0, np.shape(xyh_train)[1] - 1):
            xyh_sort = xyh_train[np.argsort(xyh_train[:, i])]

            '''Forward sequence'''
            xyh_sort_test_forward = xyh_sort[0: int(0.2 * N_)]
            xyh_sort_train_forward = xyh_sort[int(0.2 * N_):]

            '''The training (EV) set and test (EV) set in 8:2 according to the forward sequence.'''
            x_train_forward, y_train_forward = xyh_sort_train_forward[:, 0: Feature_N], xyh_sort_train_forward[:, -1]
            x_test_forward, y_test_forward = xyh_sort_test_forward[:, 0: Feature_N], xyh_sort_test_forward[:, -1]

            print('------------------------', j, '------------------------', i, '------------------------')
            (rmse_train_forward, rmse_test_forward, r2_train_forward, r2_test_forward, rmse_train_forward_ev,
             rmse_test_forward_ev, r2_train_forward_ev, r2_test_forward_ev) = ev_cal(
                x_train_forward, x_test_forward, y_train_forward, y_test_forward, N_fit, j, cd_model)

            '''Backward sequence'''
            xyh_sort_test_backward = xyh_sort[-int(0.2 * N_):]
            xyh_sort_train_backward = xyh_sort[0: int(N_ - 0.2 * N_) + 1]

            '''The training (EV) set and test (EV) set in 8:2 according to the backward sequence.'''
            x_train_backward, y_train_backward = xyh_sort_train_backward[:, 0: Feature_N], xyh_sort_train_backward[:,
                                                                                           -1]
            x_test_backward, y_test_backward = xyh_sort_test_backward[:, 0: Feature_N], xyh_sort_test_backward[:, -1]

            (rmse_train_backward, rmse_test_backward, r2_train_backward, r2_test_backward, rmse_train_backward_ev,
             rmse_test_backward_ev, r2_train_backward_ev, r2_test_backward_ev) = ev_cal(
                x_train_backward, x_test_backward, y_train_backward, y_test_backward, N_fit, j, cd_model)

            names[f'result_{j}'].append([i,
                                         rmse_train_forward, rmse_test_forward, r2_train_forward, r2_test_forward,
                                         rmse_train_forward_ev, rmse_test_forward_ev, r2_train_forward_ev,
                                         r2_test_forward_ev, rmse_train_backward, rmse_test_backward, r2_train_backward,
                                         r2_test_backward, rmse_train_backward_ev, rmse_test_backward_ev,
                                         r2_train_backward_ev, r2_test_backward_ev])

        names[f'df_{j}'] = pd.DataFrame(names[f'result_{j}'])
        names[f'df_{j}'].to_excel(f'{cd_dataset}extra_{j}.xlsx', index=False, header=headers)
