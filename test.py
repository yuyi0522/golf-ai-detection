import sys
import os
from sys import platform

import numpy

file_path = "./venv/Lib/site-packages/oplib"

if platform == "win32":
    lib_dir = "Release"
    bin_dir = "bin"
    x64_dir = "x64"
    lib_path = os.path.join(file_path, lib_dir)
    bin_path = os.path.join(file_path, bin_dir)
    x64_path = os.path.join(file_path, x64_dir)

    sys.path.append(lib_path)

    os.environ["PATH"] += ";" + bin_path + ";" + x64_path + "\\Release;"
    print(os.environ["PATH"])

    try:
        import pyopenpose as op

        print("YOU SUCCESS IMPORT PYOPENPOSE!!!!!!!")
    except ImportError as e:
        print("fail to import pyopenpose!")
        raise e
else:
    print(f"當前電腦環境:{platform}\n")
    sys.exit(-1)

# 以上為import openpose
import pandas as pd
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
import seaborn as sns

def calculate_angle(x1, y1, x2, y2, x3, y3):
    b = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    c = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5
    a = ((x1 - x3) ** 2 + (y1 - y3) ** 2) ** 0.5
    if b == 0 or c == 0:
        return np.nan
    cos_theta = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
    theta = np.arccos(cos_theta) / math.pi * 180
    return theta


params = dict()
params["model_folder"] = "./venv/Lib/site-packages/oplib/models"
opWrapper, datum = op.WrapperPython(), op.Datum()
opWrapper.configure(params)
opWrapper.start()

data = []  # 以串列紀錄資料,全部append完後再轉乘dataframe

capture = cv2.VideoCapture("./src/videos_160/test2.mp4")
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 取得影像寬度
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 取得影像高度
fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")  # 設定影片的格式為 MP4
# out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (width, height))  # 產生空的影片

frame_count = -1    # 後續因為會計算座標差而捨棄每個影片的第0幀, 故此設為-1後續移除才不會影響到整體編號
while 1:
    ref, frame = capture.read()

    if not ref:
        break

    # 載入當前偵到Openpose
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # 模型輸出圖像
    process = datum.cvOutputData
    # out.write(process)  # 將取得的每一幀圖像寫入空的影片
    # print("Pose keypoints:\n", str(datum.poseKeypoints))
    if datum.poseKeypoints is None:
        continue
    keypoints = datum.poseKeypoints[0]  # poseKeypoint有可能有兩個元素(正常只會有一個包含25個串列的元素),多人的原因,故取第一個即可
    keypoints_df = pd.DataFrame(np.reshape(keypoints, (25, 3)))
    keypoints_df.columns = ["x", "y", "c"]
    keypoints_df.index = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip",
                          "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar",
                          "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]
    # print(keypoints_df)  # 輸出所有關鍵點位置及分數
    frame_data = []     # 單一幀的資料
    frame_data.extend([0, frame_count])
    # 計算右手角度,對應BODY25點位為4, 3, 2
    frame_data.append(calculate_angle(keypoints_df.at["RWrist", "x"], keypoints_df.at["RWrist", "y"],
                                      keypoints_df.at["RElbow", "x"], keypoints_df.at["RElbow", "y"],
                                      keypoints_df.at["RShoulder", "x"], keypoints_df.at["RShoulder", "y"]) / 360)
    # 計算左手角度,對應BODY25點位為7, 6, 5
    frame_data.append(calculate_angle(keypoints_df.at["LWrist", "x"], keypoints_df.at["LWrist", "y"],
                                      keypoints_df.at["LElbow", "x"], keypoints_df.at["LElbow", "y"],
                                      keypoints_df.at["LShoulder", "x"], keypoints_df.at["LShoulder", "y"]) / 360)
    # 計算右腳角度,對應BODY25點位為9, 10, 11
    frame_data.append(calculate_angle(keypoints_df.at["RHip", "x"], keypoints_df.at["RHip", "y"],
                                      keypoints_df.at["RKnee", "x"], keypoints_df.at["RKnee", "y"],
                                      keypoints_df.at["RAnkle", "x"], keypoints_df.at["RAnkle", "y"]) / 360)
    # 計算右腳角度,對應BODY25點位為12, 13, 14
    frame_data.append(calculate_angle(keypoints_df.at["LHip", "x"], keypoints_df.at["LHip", "y"],
                                      keypoints_df.at["LKnee", "x"], keypoints_df.at["LKnee", "y"],
                                      keypoints_df.at["LAnkle", "x"], keypoints_df.at["LAnkle", "y"]) / 360)
    # 計算腰角度,對應BODY25點位為1, 8, 9
    frame_data.append(calculate_angle(keypoints_df.at["Neck", "x"], keypoints_df.at["Neck", "y"],
                                      keypoints_df.at["MidHip", "x"], keypoints_df.at["MidHip", "y"],
                                      keypoints_df.at["RHip", "x"], keypoints_df.at["RHip", "y"]) / 360)
    frame_data.extend(keypoints_df.loc[["Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
                                        "MidHip", "RHip", "RKnee", "LHip", "LKnee"]]["x"].tolist())
    frame_data.extend(keypoints_df.loc[["Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
                                        "MidHip", "RHip", "RKnee", "LHip", "LKnee"]]["y"].tolist())
    data.append(frame_data)
    frame_count += 1

    # 顯示
    cv2.imshow("video", process)
    if not ref:
        break
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        print("Escape hit, closing...")
        break

# poseModel = op.PoseModel.BODY_25
# print(op.getPoseBodyPartMapping(poseModel))
# print(op.getPoseNumberBodyParts(poseModel))

capture.release()
# out.release()  # 釋放資源
cv2.destroyAllWindows()
# 處理缺失值
for i in range(len(data)):
    for j in range(2, len(data[0])):
        if (data[i][j] == 0 or np.isnan(data[i][j])) and 0 < i < (len(data) - 1):
            if data[i + 1][j] != 0 and not np.isnan(data[i + 1][j]):
                data[i][j] = (data[i - 1][j] + data[i + 1][j]) / 2
            else:
                data[i][j] = data[i - 1][j]
        elif (data[i][j] == 0 or np.isnan(data[i][j])) and i == (len(data) - 1):
            data[i][j] = data[i - 1][j]

video_num = -1
for i in range(len(data)):
    if data[i][0] == video_num:
        temp = data[i].copy()
        for j in range(7, 31):
            data[i][j] = math.fabs(data[i][j] - last_frame_data[j])
        last_frame_data = temp.copy()
    else:
        video_num += 1
        last_frame_data = data[i].copy()
for _ in data:
    if _[1] == -1:
        data.remove(_)

# 原data行名稱

col = ["video_id", "frame_id", "right_arm_angle", "left_arm_angle", "right_leg_angle", "left_leg_angle", "waist_angle",
       "Neck_delta_x", "RShoulder_delta_x", "RElbow_delta_x", "RWrist_delta_x", "LShoulder_delta_x", "LElbow_delta_x",
       "LWrist_delta_x", "MidHip_delta_x", "RHip_delta_x", "RKnee_delta_x", "LHip_delta_x", "LKnee_delta_x",
       "Neck_delta_y", "RShoulder_delta_y", "RElbow_delta_y", "RWrist_delta_y", "LShoulder_delta_y", "LElbow_delta_y",
       "LWrist_delta_y", "MidHip_delta_y", "RHip_delta_y", "RKnee_delta_y", "LHip_delta_y", "LKnee_delta_y"]
arm_angle_col = ["video_id", "frame_id", "right_arm_angle", "left_arm_angle"]
leg_angle_col = ["video_id", "frame_id", "right_leg_angle", "left_leg_angle"]
shoulder_col = ["video_id", "frame_id", "RShoulder_delta_x", "LShoulder_delta_x", "RShoulder_delta_y",
                "LShoulder_delta_y"]
elbow_wrist_col = ["video_id", "frame_id", "RElbow_delta_x", "RWrist_delta_x", "RElbow_delta_y", "RWrist_delta_y",
                   "LElbow_delta_x", "LWrist_delta_x", "LElbow_delta_y", "LWrist_delta_y"]
hip_knee_col = ["video_id", "frame_id", "MidHip_delta_x", "RHip_delta_x", "LHip_delta_x", "MidHip_delta_y",
                "RHip_delta_y", "LHip_delta_y", "RKnee_delta_x", "LKnee_delta_x", "RKnee_delta_y", "LKnee_delta_y"]

data_df = pd.DataFrame(data, columns=col)
print(data_df)

arm_angle_df = data_df[arm_angle_col]
leg_angle_df = data_df[leg_angle_col]
shoulder_df = data_df[shoulder_col]
elbow_wrist_df = data_df[elbow_wrist_col]
hip_knee_df = data_df[hip_knee_col]

# =============================

from keras.models import load_model
import keras.models


arm_angle_model = load_model("arm_angle.h5")
leg_angle_model = load_model("leg_angle.h5")
shoulder_model = load_model("shoulder.h5")
elbow_wrist_model = load_model("elbow_wrist.h5")
hip_knee_model = load_model("hip_knee.h5")

max_timestep = 300
model = [arm_angle_model, leg_angle_model, shoulder_model, elbow_wrist_model, hip_knee_model]
df = [arm_angle_df, leg_angle_df, shoulder_df, elbow_wrist_df, hip_knee_df]
name = ["arm_angle", "leg_angle", "shoulder", "elbow_wrist", "hip_knee"]
for i in range(5):
    padding_value = 0.00
    X_test = df[i].iloc[:-1, 2:].values
    Y_test = df[i].iloc[1:, 2:].values
    temp_X = X_test
    temp_Y = Y_test
    if len(temp_X) < max_timestep:
        concat = np.full((max_timestep - len(temp_X), temp_X.shape[1]), padding_value)
        X_test = np.concatenate([X_test, concat])
    if len(temp_Y) < max_timestep:
        concat = np.full((max_timestep - len(temp_Y), temp_Y.shape[1]), padding_value)
        Y_test = np.concatenate([Y_test, concat])
    X_test = X_test.reshape(1, 300, df[i].shape[1] - 2)
    Y_test = Y_test.reshape(1, 300 * (df[i].shape[1] - 2))
    score = model[i].evaluate(X_test, Y_test, verbose=0)
    print(name[i] + " mse:" + str(model[i].evaluate(X_test, Y_test, verbose=0)[1]))
    print(name[i] + " rmse:" + str(model[i].evaluate(X_test, Y_test, verbose=0)[2]))
    print(name[i] + " mae:" + str(model[i].evaluate(X_test, Y_test, verbose=0)[3]))
    print(name[i] + " mape:" + str(model[i].evaluate(X_test, Y_test, verbose=0)[4]))
