import pandas as pd
import numpy as np
import os
# LSTM model
from keras.models import save, load_model
from model_build import model_build

data_df_list = []
csv_path = "./src/csv/"
n = 0
df_name = []
for file in os.listdir(csv_path):
    df_name.append(file[:-4])
    data_df_list.append(pd.read_csv(csv_path + file))
    print(data_df_list[n].columns)
    data_df_list[n] = data_df_list[n].drop(["Unnamed: 0"], axis=1)
    n += 1

max_timestep = 300
model = []
for i in range(5):
    # 特徵數量=dataframe columns - 2(video_id and frame_id)
    model.append(model_build(max_timestep=max_timestep, feature=data_df_list[i].shape[1] - 2))
    # 開始訓練
    groups = data_df_list[i].groupby('video_id')
    padding_value = 0.00
    for video_id, group in groups:
        # if video_id != 0:   # 只測試一部影片
        #     break
        X_train = group.iloc[:-1, 2:].values
        Y_train = group.iloc[1:, 2:].values
        temp_X = X_train
        temp_Y = Y_train
        if len(temp_X) < max_timestep:
            concat = np.full((max_timestep - len(temp_X), temp_X.shape[1]), padding_value)
            X_train = np.concatenate([X_train, concat])
        if len(temp_Y) < max_timestep:
            concat = np.full((max_timestep - len(temp_Y), temp_Y.shape[1]), padding_value)
            Y_train = np.concatenate([Y_train, concat])
        X_train = X_train.reshape(1, 300, data_df_list[i].shape[1] - 2)
        Y_train = Y_train.reshape(1, 300 * (data_df_list[i].shape[1] - 2))
        model[i].fit(X_train, Y_train, epochs=10, verbose=0)

    model[i].save("./" + df_name[i] + ".h5")
