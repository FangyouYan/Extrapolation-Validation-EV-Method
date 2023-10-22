import os
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.colors
import numpy as np
import pandas as pd
'''Obtain root mean squared error (RMSE) for the training set of the model'''
from Model_Cal import rmse_train_mlr, rmse_train_rf, rmse_train_gbdt, rmse_train_adaboost, rmse_train_xgboost
from Model_Cal import rmse_train_lasso, rmse_train_ridge, rmse_train_knn, rmse_train_mlp, rmse_train_svm, rmse_train_gpr


'''Calculation of the standard deviation of the samples within the 95% confidence level interval (σ95)'''
def std_95_(y):
    mean = np.mean(y)
    std = np.std(y)
    lower_limit = mean - 1.96 * std
    upper_limit = mean + 1.96 * std

    sample_95 = y[(y>=lower_limit) * (y<=upper_limit)]
    std_95 = np.std(sample_95)

    return std_95

def ev_plot(extra_degree1_sort, extra_degree2_sort, x_name_sort, aae_test1_sort, aae_test1_e_sort, aae_test2_sort,
            aae_test2_e_sort, extra_variance, fig_path, fig_name, aae_new):
    plt.rc('font', family='Times New Roman')
    x = np.arange(len(x_name_sort))
    bar_width = 0.45
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 15), linewidth=5)
    ax1.set_ylim(-1, 30)
    ax1.set_xlim(0, 100)
    x_t_t = np.linspace(0, 80, 8)
    x_t_l = [0] + [np.around(i, 2) for i in x_t_t[1:]]
    y_t_t = list(np.arange(len(x_name_sort)))
    y_t_l = x_name_sort.tolist()
    ax1.set_yticks(y_t_t, y_t_l)
    map_r1 = (0.8/0.2)
    extra_degree1_sort_c = extra_degree1_sort*map_r1
    colors1 = plt.cm.Blues(extra_degree1_sort_c)
    ax1.barh(x, aae_test1_sort, color=colors1, align='center', height=bar_width, alpha=1, edgecolor='black',
             linewidth=0.6, label=f'{fig_name} Model')
    c1 = ax1.scatter(aae_test1_e_sort, x, color=colors1, marker='o', alpha=1, edgecolor='black', linewidths=0.6, s=100,
                     label='Extrapolation')
    ax1.axvline(x=aae_new, c='#7f7f7f', ls='--', lw=1)  
    ax1.axvline(x=extra_variance, c='#828FDB', ls='--', lw=1)
    cmap1 = cm.get_cmap('Blues')
    norm1 = mpl.colors.Normalize(vmin=0, vmax=0.8)  
    im1 = mpl.cm.ScalarMappable(norm=norm1, cmap=cmap1)  
    cax1 = fig.add_axes([0.09, 0.14, 0.015, 0.7])
    cb1 = plt.colorbar(im1, cmap='Blues', cax=cax1)
    cb1.set_ticks([0, 0.2, 0.4, 0.6, 0.8])
    cb1.ax.tick_params(labelsize=12)
    ax1.set_xlabel('RMSE', fontdict={'weight': 'bold', 'size': 36})
    ax1.yaxis.set_ticks_position('right')
    ax1.tick_params(axis='both', labelsize=30)
    ax1.set_yticklabels([])
    ax1.invert_xaxis()
    ax1.tick_params(axis='both', labelsize=30)
    ax2.set_ylim(-1, 30)
    ax2.set_xlim(0, 100)
    ax2.set_yticks(y_t_t, y_t_l)
    map_r2 = (0.8/0.2)
    extra_degree2_sort_c = extra_degree2_sort*map_r2
    colors2 = plt.cm.Oranges(extra_degree2_sort_c)
    ax2.barh(x, aae_test2_sort, color=colors2, align='center', height=bar_width, edgecolor='black', linewidth=0.6,
             label=f'{fig_name} Model')
    ax2.scatter(aae_test2_e_sort, x, color=colors2, marker='o', alpha=1, edgecolor='black', linewidths=0.6, s=100,
                label='Extrapolation')
    ax2.axvline(x=aae_new, c='#7f7f7f', ls='--', lw=1)  
    ax2.axvline(x=extra_variance, c='#828FDB', ls='--', lw=1)
    cmap2 = cm.get_cmap('Oranges')
    norm2 = mpl.colors.Normalize(vmin=0, vmax=0.8)  
    im2 = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2)  
    cax2 = fig.add_axes([0.92, 0.14, 0.015, 0.7])
    cb2 = plt.colorbar(im2, cmap='Oranges', cax=cax2)
    cb2.set_ticks([0, 0.2, 0.4, 0.6, 0.8])
    cb2.ax.tick_params(labelsize=12)
    ax2.set_xlabel('RMSE', fontdict={'weight': 'bold', 'size': 36})
    ax2.yaxis.set_ticks_position('left')
    ax2.tick_params(axis='both', labelsize=30)
    plt.subplots_adjust(hspace=0.8)
    plt.savefig(f'{fig_path}extra_{fig_name}.png')
    plt.close()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    cd_project = os.path.dirname(os.path.dirname(os.getcwd()))

    cd_dataset = f'{cd_project}/Tg_Model/Dataset/'
    cd_model = f'{cd_project}/Tg_Model/Model/'
    cd_figure = f'{cd_project}/Tg_Model/Figure/Extra/'
    x_train_all = pd.read_excel(cd_dataset + r"dataset_Tg_x_train.xlsx", sheet_name='Sheet1')
    y_train_all = pd.read_excel(cd_dataset + r"dataset_Tg_y_train.xlsx", sheet_name='Sheet1')
    extra_degree = pd.read_excel(cd_dataset + r"extra_degree.xlsx", sheet_name='Sheet1')

    Feature_N = int(np.shape(x_train_all)[1])

    x_train = x_train_all.values[:, :]
    y_train = y_train_all.values[:, :]

    '''Calculation of the standard deviation of the samples within the 95% confidence level interval (σ95)'''
    std_95 = std_95_(y_train)

    names = locals()
    model = ['knn', 'svm', 'xgboost', 'mlp', 'adaboost', 'lasso', 'gbdt', 'mlr', 'ridge', 'rf', 'gpr']

    for i in model:
        '''Obtaining extrapolation validation results and extrapolation degree'''
        model_result = pd.read_excel(f'{cd_dataset}extra_{i}.xlsx')
        extra_degree_forward, extra_degree_backward = extra_degree['forward_extra_degree'], extra_degree['backward_extra_degree']
        x_name = model_result['x_name']
        rmse_test_forward, rmse_test_forward_ev = model_result['rmse_test_forward'], model_result['rmse_test_forward_ev']
        rmse_test_backward, rmse_test_backward_ev = model_result['rmse_test_backward'], model_result['rmse_test_backward_ev']

        error_test = abs(rmse_test_forward_ev - rmse_test_forward) + abs(rmse_test_backward_ev - rmse_test_backward)
        error_sort = np.append(np.argsort(error_test.values[:Feature_N]), Feature_N)

        x_name_sort = x_name[error_sort]
        extra_degree_forward_sort, extra_degree_backward_sort = extra_degree_forward[error_sort], extra_degree_backward[error_sort]
        rmse_test_forward_sort, rmse_test_forward_ev_sort = rmse_test_forward[error_sort], rmse_test_forward_ev[error_sort]
        rmse_test_backward_sort, rmse_test_backward_ev_sort = rmse_test_backward[error_sort], rmse_test_backward_ev[error_sort]

        '''Generate the extrapolation validation plot'''
        ev_plot(extra_degree_forward_sort, extra_degree_backward_sort, x_name_sort, rmse_test_forward_sort, rmse_test_forward_ev_sort,
                rmse_test_backward_sort, rmse_test_backward_ev_sort, std_95, cd_figure, f'{i}', names[f'rmse_train_{i}'])

