import pandas as pd
import numpy as np

# This downsamples the data in 4Hz, and adds an id to the data
def process_file(file, num_id):
    unpickled_df = pd.read_pickle(file)
    label = unpickled_df['label'][0::(700 // 4)].flatten()
    chest_ACC = unpickled_df['signal']['chest']['ACC'][0::(700 // 4)]
    chest_ACC_x = chest_ACC[:, 0].flatten()
    chest_ACC_y = chest_ACC[:, 1].flatten()
    chest_ACC_z = chest_ACC[:, 2].flatten()
    chest_ECG = unpickled_df['signal']['chest']['ECG'][0::(700 // 4)].flatten()
    chest_EMG = unpickled_df['signal']['chest']['EMG'][0::(700 // 4)].flatten()
    chest_EDA = unpickled_df['signal']['chest']['EDA'][0::(700 // 4)].flatten()
    chest_Temp = unpickled_df['signal']['chest']['Temp'][0::(700 // 4)].flatten()
    chest_Resp = unpickled_df['signal']['chest']['Resp'][0::(700 // 4)].flatten()
    wrist_ACC = unpickled_df['signal']['wrist']['ACC'][0::(32 // 4)]
    wrist_ACC_x = wrist_ACC[:, 0].flatten()
    wrist_ACC_y = wrist_ACC[:, 1].flatten()
    wrist_ACC_z = wrist_ACC[:, 2].flatten()
    wrist_BVP = unpickled_df['signal']['wrist']['BVP'][0::(64 // 4)].flatten()
    wrist_EDA = unpickled_df['signal']['wrist']['EDA'].flatten()
    wrist_TEMP = unpickled_df['signal']['wrist']['TEMP'].flatten()

    rows = len(wrist_TEMP)
    ids = np.zeros((rows,))
    ids.fill(num_id)

    print(ids.shape)
    print(label.shape)
    print(chest_ACC_x.shape)
    print(chest_ACC_y.shape)
    print(chest_ACC_z.shape)
    print(chest_ECG.shape)
    print(chest_EMG.shape)
    print(chest_EDA.shape)
    print(chest_Temp.shape)
    print(chest_Resp.shape)
    print(wrist_ACC_x.shape)
    print(wrist_ACC_y.shape)
    print(wrist_ACC_z.shape)
    print(wrist_BVP.shape)
    print(wrist_EDA.shape)
    print(wrist_TEMP.shape)

    df = pd.DataFrame({"id": ids,
                       "label": label,
                       "chest_ACC_x": chest_ACC_x,
                       "chest_ACC_y": chest_ACC_y,
                       "chest_ACC_z": chest_ACC_z,
                       "chest_ECG": chest_ECG,
                       "chest_EMG": chest_EMG,
                       "chest_EDA": chest_EDA,
                       "chest_Temp": chest_Temp,
                       "chest_Resp": chest_Resp,
                       "wrist_ACC_x": wrist_ACC_x,
                       "wrist_ACC_y": wrist_ACC_y,
                       "wrist_ACC_z": wrist_ACC_z,
                       "wrist_BVP": wrist_BVP,
                       "wrist_EDA": wrist_EDA,
                       "wrist_TEMP": wrist_TEMP})
    return df

# Process all synchronized data
# Change the file paths accordingly (1st parameter)
df01 = process_file("../WESAD/S2/S2.pkl",1)
df02 = process_file("../WESAD/S3/S3.pkl",2)
df03 = process_file("../WESAD/S4/S4.pkl",3)
df04 = process_file("../WESAD/S5/S5.pkl",4)
df05 = process_file("../WESAD/S6/S6.pkl",5)
df06 = process_file("../WESAD/S7/S7.pkl",6)
df07 = process_file("../WESAD/S8/S8.pkl",7)
df08 = process_file("../WESAD/S9/S9.pkl",8)
df09 = process_file("../WESAD/S10/S10.pkl",9)
df10 = process_file("../WESAD/S11/S11.pkl",10)
df11 = process_file("../WESAD/S13/S13.pkl",11)
df12 = process_file("../WESAD/S14/S14.pkl",12)
df13 = process_file("../WESAD/S15/S15.pkl",13)
df14 = process_file("../WESAD/S16/S16.pkl",14)
df15 = process_file("../WESAD/S17/S17.pkl",15)

# Combine files
df = pd.concat([df01, df02, df03, df04, df05,
                df06, df07, df08, df09, df10,
               df11, df12, df13, df14, df15])

# Write file to csv
df.to_csv("data.csv", index=False)

# Downsample ECG data to 100Hz
def process_file_ECG(file, num_id):
    unpickled_df = pd.read_pickle(file)
    chest_ECG = unpickled_df['signal']['chest']['ECG'][0::(700 // 100)].flatten()

    rows = len(chest_ECG)
    ids = np.zeros((rows,))
    ids.fill(num_id)

    df = pd.DataFrame({"id": ids,
                       "chest_ECG": chest_ECG})
    return df

df01_ECG = process_file_ECG("../WESAD/S2/S2.pkl",1)
df02_ECG = process_file_ECG("../WESAD/S3/S3.pkl",2)
df03_ECG = process_file_ECG("../WESAD/S4/S4.pkl",3)
df04_ECG = process_file_ECG("../WESAD/S5/S5.pkl",4)
df05_ECG = process_file_ECG("../WESAD/S6/S6.pkl",5)
df06_ECG = process_file_ECG("../WESAD/S7/S7.pkl",6)
df07_ECG = process_file_ECG("../WESAD/S8/S8.pkl",7)
df08_ECG = process_file_ECG("../WESAD/S9/S9.pkl",8)
df09_ECG = process_file_ECG("../WESAD/S10/S10.pkl",9)
df10_ECG = process_file_ECG("../WESAD/S11/S11.pkl",10)
df11_ECG = process_file_ECG("../WESAD/S13/S13.pkl",11)
df12_ECG = process_file_ECG("../WESAD/S14/S14.pkl",12)
df13_ECG = process_file_ECG("../WESAD/S15/S15.pkl",13)
df14_ECG = process_file_ECG("../WESAD/S16/S16.pkl",14)
df15_ECG = process_file_ECG("../WESAD/S17/S17.pkl",15)

df_ECG = pd.concat([df01_ECG, df02_ECG, df03_ECG, df04_ECG, df05_ECG,
                df06_ECG, df07_ECG, df08_ECG, df09_ECG, df10_ECG,
               df11_ECG, df12_ECG, df13_ECG, df14_ECG, df15_ECG])

df_ECG.to_csv("data_ECG.csv", index=False)

# Downsample ECG data to 16Hz
def process_file_BVP(file, num_id):
    unpickled_df = pd.read_pickle(file)
    wrist_BVP = unpickled_df['signal']['wrist']['BVP'][0::(64 // 16)].flatten()

    rows = len(wrist_BVP)
    ids = np.zeros((rows,))
    ids.fill(num_id)

    df = pd.DataFrame({"id": ids,
                       "wrist_BVP": wrist_BVP})
    return df

df01_BVP = process_file_BVP("../WESAD/S2/S2.pkl",1)
df02_BVP = process_file_BVP("../WESAD/S3/S3.pkl",2)
df03_BVP = process_file_BVP("../WESAD/S4/S4.pkl",3)
df04_BVP = process_file_BVP("../WESAD/S5/S5.pkl",4)
df05_BVP = process_file_BVP("../WESAD/S6/S6.pkl",5)
df06_BVP = process_file_BVP("../WESAD/S7/S7.pkl",6)
df07_BVP = process_file_BVP("../WESAD/S8/S8.pkl",7)
df08_BVP = process_file_BVP("../WESAD/S9/S9.pkl",8)
df09_BVP = process_file_BVP("../WESAD/S10/S10.pkl",9)
df10_BVP = process_file_BVP("../WESAD/S11/S11.pkl",10)
df11_BVP = process_file_BVP("../WESAD/S13/S13.pkl",11)
df12_BVP = process_file_BVP("../WESAD/S14/S14.pkl",12)
df13_BVP = process_file_BVP("../WESAD/S15/S15.pkl",13)
df14_BVP = process_file_BVP("../WESAD/S16/S16.pkl",14)
df15_BVP = process_file_BVP("../WESAD/S17/S17.pkl",15)

df_BVP = pd.concat([df01_BVP, df02_BVP, df03_BVP, df04_BVP, df05_BVP,
                df06_BVP, df07_BVP, df08_BVP, df09_BVP, df10_BVP,
               df11_BVP, df12_BVP, df13_BVP, df14_BVP, df15_BVP])

df_BVP.to_csv("data_BVP.csv", index=False)