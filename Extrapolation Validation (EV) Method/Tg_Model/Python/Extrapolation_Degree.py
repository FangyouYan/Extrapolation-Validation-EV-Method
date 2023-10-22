import os
import numpy as np
import warnings
import pandas as pd


'''Calculation of extrapolation degree'''
def extrapolation_degree(x_train, x_test):
    extra_degree_all = []
    for j in range(0, int(np.shape(x_train)[1])):
        x_train_max = np.max(x_train[:, j])
        x_train_min = np.min(x_train[:, j])
        x_train_mean = np.mean(x_train[:, j])

        x_test_min = x_train_min - x_test[:, j]
        x_test_min[np.where(x_test_min < 0)] = 0
        x_test_max = x_test[:, j] - x_train_max
        x_test_max[np.where(x_test_max < 0)] = 0
        x_test_gap = abs(x_train_mean - x_test[:, j])
        x_test_gap[np.where((x_test_min == 0) & (x_test_max == 0))] = 0

        if np.sum(x_test_gap) == 0:
            extra_degree_x = 0
        else:
            extra_degree_x = (np.sum(x_test_min) + np.sum(x_test_max)) / np.sum(x_test_gap)

        extra_degree_all.append(extra_degree_x)

    extra_degree = np.mean(np.array(extra_degree_all))

    return extra_degree


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

    '''Calculation of leverage value (h)'''
    XTX_inverse = np.linalg.inv(np.dot(x_train.T, x_train))
    h_train_ = np.diagonal(np.dot(np.dot(x_train, XTX_inverse), x_train.T))
    h_train = np.expand_dims(h_train_, axis=1)

    xyh_train = np.concatenate((x_train, h_train, y_train), axis=1)

    extra_degree = []
    for i in range(0, np.shape(xyh_train)[1] - 1):

        xyh_sort = xyh_train[np.argsort(xyh_train[:, i])]

        '''Forward sequence'''
        xyh_sort_test_forward = xyh_sort[0: int(0.2 * N_)]
        xyh_sort_train_forward = xyh_sort[int(0.2 * N_):]

        '''Backward sequence'''
        xyh_sort_test_backward = xyh_sort[-int(0.2 * N_):]
        xyh_sort_train_backward = xyh_sort[0: int(N_ - 0.2 * N_) + 1]

        '''The training (EV) set and test (EV) set in 8:2 according to the forward sequence.'''
        x_train_forward, y_train_forward = xyh_sort_train_forward[:, 0: Feature_N], xyh_sort_train_forward[:, -1]
        x_test_forward, y_test_forward = xyh_sort_test_forward[:, 0: Feature_N], xyh_sort_test_forward[:, -1]

        '''The training (EV) set and test (EV) set in 8:2 according to the backward sequence.'''
        x_train_backward, y_train_backward = xyh_sort_train_backward[:, 0: Feature_N], xyh_sort_train_backward[:, -1]
        x_test_backward, y_test_backward = xyh_sort_test_backward[:, 0: Feature_N], xyh_sort_test_backward[:, -1]

        '''Calculation of forward extrapolation degree'''
        forward_extra_degree = extrapolation_degree(x_train_forward, x_test_forward)

        '''Calculation of backward extrapolation degree'''
        backward_extra_degree = extrapolation_degree(x_train_backward, x_test_backward)

        extra_degree.append([i, forward_extra_degree, backward_extra_degree])

    df_extra_degree = pd.DataFrame(extra_degree)
    df_extra_degree.to_excel(f'{cd_dataset}extra_degree.xlsx', index=False,
                             header=['x_name', 'forward_extra_degree', 'backward_extra_degree'])
