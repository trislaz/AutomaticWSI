

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np 
from glob import glob


def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description='dispatches a set of files into folds')
 
    parser.add_argument('--main_name', required=True,
                        metavar="str", type=str,
                        help='main name')  
    parser.add_argument('--y_variable', required=False,
                        metavar="str", type=str,
                        help='name of the variable to predict')
    args = parser.parse_args()
    return args

def write_score(value, name):
    file = open(name, "w") 
    print("Writting {} to {}".format(value, name))
    file.write(str(value)) 
    file.close() 

try:
    import os
    os.mkdir("plots")
except:
    pass

def plot(table_old, num, acc_name, val_acc_name, test_acc_name):
    table = table_old.copy()
    variables = ['hidden_btleneck', 'hidden_fcn', 'drop_out', 
                 'learning_rate', 'weight_decay', 'gaussian_noise']

    custom_bucket_array = np.linspace(0, 1, 11)
    var_list = ['drop_out']
    if len(np.unique(table['gaussian_noise'])) != 1:
        var_list.append("gaussian_noise")
    for var in var_list:
        table[var] = pd.cut(table[var], custom_bucket_array)
    for var in ['learning_rate', 'weight_decay']:
        mini = int(np.floor(np.log(table[var]).min()))
        maxi = int(np.ceil(np.log(table[var]).max()))
        custom_bucket_array = np.linspace(mini, maxi, 11)
        table[var] = pd.cut(np.log(table[var]), custom_bucket_array)

    for z_name in [acc_name, val_acc_name, test_acc_name]:
        mat = np.zeros(shape=(len(variables), len(variables)))
        np.fill_diagonal(mat, 1)
        for i, x_name in enumerate(variables):
            for j, y_name in enumerate(variables):
                if mat[i, j] == 0:
                    if len(np.unique(table[x_name])) != 1 and len(np.unique(table[y_name])) != 1:
                        print(x_name, y_name)
                        pivotted = table.pivot_table(values=z_name, index=x_name, columns=y_name, aggfunc='mean')
                        column_order = np.sort(pivotted.columns)
                        index_order = np.sort(pivotted.index)
                        pivotted = pivotted.reindex_axis(index_order, axis=0)
                        pivotted = pivotted.reindex_axis(column_order, axis=1)
                        try:
                            sns.heatmap(pivotted, cmap='RdBu')
                        except:
                            import pdb; pdb.set_trace()
                        plt.tight_layout()
                        plt.savefig("plots/{}__{}__{}__{}.png".format(x_name, y_name, z_name, num))
                        plt.close()

                    mat[i, j] = 1
                    mat[j, i] = 1

options = get_options()

table = []
for f in glob("neural_networks_*.csv"):
    table.append(pd.read_csv(f, index_col=0))
table = pd.concat(table, axis=0).dropna()
table = table.reset_index()

def idx_get_max_on_validation(table, name, op=max):
    idx = table.groupby(["validation_fold"], sort=False)[name].transform(op) == table[name]
    return table.ix[idx]


acc_train = []
acc_train_chosen_f1 = []
acc_val = []
acc_val_chosen_f1 = []
acc_test = []
acc_test_chosen_f1 = []

if options.y_variable in ["RCB_class"]:

    acc_name = "acc" if "acc" in table.columns else "RCB_class_acc"
    val_f1_name = "val_f1" if "val_f1" in table.columns else "val_RCB_class_f1"
    val_acc_name = "val_acc" if "val_acc" in table.columns else "val_RCB_class_acc"
    test_acc_name = "test_acc" if "test_acc" in table.columns else "test_RCB_class_acc"
    op = max
    typep = "accuracy"
else:
    acc_name = "mean_squared_error"
    val_acc_name = "val_mean_squared_error"
    test_acc_name = "test_mean_squared_error"
    op = min
    typep = "mse"

best_run_test = []
for fold_test, table_fold in table.groupby("fold_test"):

    test_fold_table = table_fold.copy().reset_index()
    best_on_fold_val = idx_get_max_on_validation(test_fold_table, val_acc_name, op)

    if op == max:
        ind_val = best_on_fold_val[val_acc_name].argmax()
        train_acc_score = best_on_fold_val.loc[ind_val, acc_name]
        val_acc_score = best_on_fold_val.loc[ind_val, val_acc_name]
        test_acc_score = best_on_fold_val.loc[ind_val, test_acc_name]
    elif op == min:
        ind_val = best_on_fold_val[val_acc_name].argmin()
        train_acc_score = best_on_fold_val.loc[ind_val, acc_name].min()
        val_acc_score = best_on_fold_val.loc[ind_val, val_acc_name].min()
        test_acc_score = best_on_fold_val.loc[ind_val, test_acc_name].min()
    else:
        train_acc_score = best_on_fold_val[acc_name].mean()
        val_acc_score = best_on_fold_val[val_acc_name].mean()
        test_acc_score = best_on_fold_val[test_acc_name].mean()

    acc_train.append(train_acc_score)
    acc_val.append(val_acc_score)
    acc_test.append(test_acc_score)
    best_run_test.append((fold_test, ind_val))

    if options.y_variable in ["RCB_class"]:
        best_on_f1 = idx_get_max_on_validation(test_fold_table, val_f1_name, op)
        if op == max:
            train_f1_score = best_on_f1[acc_name].max()
            val_f1_score = best_on_f1[val_acc_name].max()
            test_f1_score = best_on_f1[test_acc_name].max()
        elif op == min:
            train_f1_score = best_on_f1[acc_name].min()
            val_f1_score = best_on_f1[val_acc_name].min()
            test_f1_score = best_on_f1[test_acc_name].min()
        else:
            train_f1_score = best_on_f1[acc_name].mean()
            val_f1_score = best_on_f1[val_acc_name].mean()
            test_f1_score = best_on_f1[test_acc_name].mean()



        acc_train_chosen_f1.append(train_f1_score)
        acc_val_chosen_f1.append(val_f1_score)
        acc_test_chosen_f1.append(test_f1_score)


prediction_of_best_val = []
for test, run in best_run_test:
    name = "predictions_run_{}_test_fold_{}.csv".format(run+1, int(test))
    tatt = pd.read_csv(name, index_col=0)
    tatt["fold"] = int(test)
    prediction_of_best_val.append(tatt)

pred_table = pd.concat(prediction_of_best_val)
pred_table = pred_table.dropna(axis=0)


ax = sns.scatterplot(x="y_true", y="y_test", hue="fold", palette="Set1",
                     data=pred_table, legend="full")
ax.plot(np.arange(0,1,0.01), np.arange(0,1,0.01), label="bissectrice")
plt.savefig("true_vs_pred.png")

plot(table, "none", acc_name, val_acc_name, test_acc_name)
write_score(np.mean(acc_train), "{}_train.txt".format(typep))
write_score(np.mean(acc_val), "{}_val.txt".format(typep))
write_score(np.mean(acc_test), "{}_test.txt".format(typep))
if options.y_variable in ["RCB_class"]:
    write_score(np.mean(acc_train_chosen_f1), "accuracy_f1_train.txt")
    write_score(np.mean(acc_val_chosen_f1), "accuracy_f1_val.txt")
    write_score(np.mean(acc_test_chosen_f1), "accuracy_f1_test.txt")


table.to_csv("regrouped.csv", index=False)
