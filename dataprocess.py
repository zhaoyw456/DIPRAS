import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
import os
from sklearn.preprocessing import StandardScaler
from statsmodels.robust.scale import mad
from sklearn.utils import shuffle



def get_disease(strat):
    strat_index = strat.index
    strat = np.array(strat).squeeze()
    _ = []
    for i in strat:
        _.append(i[:-1])
    strat = np.array(_).reshape(-1, 1)
    strat = pd.DataFrame(strat, index=strat_index)
    return strat
def split_data(name,x,y,strat,disease_all):

    strat_new = y.iloc[:,:-1].sum(axis = 1)
    stratall = np.array(strat_new).reshape(-1)


    x_train_all, x_test_all, y_train_all, y_test_all = train_test_split(x, y, test_size=0.1,
                                                                        random_state=1,
                                                                        stratify=stratall)

    train_index = x_train_all.index
    test_index = x_test_all.index

    return x_train_all, x_test_all, y_train_all, y_test_all,\
           disease_all.loc[train_index],disease_all.loc[test_index]

def load_data(path):
    print('load_data...')
    h5 = pd.HDFStore(path + '_data.h5'
                     , mode='r')
    x = h5.get('x')
    h5.close()

    h5 = pd.HDFStore(path + '_label.h5'
                     , mode='r')
    y = h5.get('y')
    h5.close()

    h5 = pd.HDFStore(path + '_strat.h5'
                     , mode='r')
    strat = h5.get('strat')
    h5.close()

    med_dev = pd.DataFrame(mad(x), index=x.columns)
    mad_genes = med_dev.sort_values(by=0, ascending=False).iloc[0:2000].index.tolist()
    x = x.loc[:, mad_genes]
    fitted_scaler = StandardScaler().fit(x)
    x_df_update = pd.DataFrame(fitted_scaler.transform(x),columns=x.columns)
    x_df_update.index = x.index

    return x_df_update,y,strat

def main():
    fk = 0
    name = "balance"
    path = name + '/' + name
    datapath = name+'/data_multilablespilt/'
    if not os.path.exists(datapath):
        os.makedirs(datapath)

    x, y, strat = load_data(path)
    disease_all = get_disease(strat)

    x_train_all, x_test_all, y_train_all, y_test_all,disease_train_all,disease_test_all = split_data(name,x, y, strat,disease_all)

    k_fold = StratifiedKFold(5, shuffle=True)
    index = k_fold.split(X=x_train_all, y=y_train_all['total_status'])
    for train_index, test_index in index:
        print('#########')
        fk+=1
        x_train = np.array(x_train_all.iloc[train_index, :])
        x_val = np.array(x_train_all.iloc[test_index, :])
        y_train = np.array(y_train_all.iloc[train_index, :])
        y_val = np.array(y_train_all.iloc[test_index, :])
        train_disease = np.array(disease_train_all.iloc[train_index])
        val_disease = np.array(disease_train_all.iloc[test_index])
        np.save(datapath + f"x_train{fk}.npy", x_train)
        np.save(datapath + f"x_val{fk}.npy", x_val)
        np.save(datapath + f"y_train{fk}.npy", y_train)
        np.save(datapath + f"y_val{fk}.npy", y_val)
        np.save(datapath + f"train_disease{fk}.npy", train_disease)
        np.save(datapath + f"val_disease{fk}.npy", val_disease)

    np.save(datapath + "x_train_all.npy", np.array(x_train_all))
    np.save(datapath + "y_train_all.npy", np.array(y_train_all))
    np.save(datapath + "x_test_all.npy", np.array(x_test_all))
    np.save(datapath + "y_test_all.npy", np.array(y_test_all))
    np.save(datapath + "train_disease_all.npy", np.array(disease_train_all))
    np.save(datapath + "test_disease_all.npy", np.array(disease_test_all))

if __name__ == "__main__":
    main()