import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import GroupKFold


path = 'data/train.csv'
df = pd.read_csv(path)
data = []
bowel_weight = np.where(df['bowel_injury'] == 1, 2, 1)
extravasation_weight = np.where(df['extravasation_injury'] == 1, 6, 1)
kidney_weight = np.where(df['kidney_low'] == 1, 2, np.where(df['kidney_high'] == 1, 4, 1))
liver_weight = np.where(df['liver_low'] == 1, 2, np.where(df['liver_high'] == 1, 4, 1))
spleen_weight = np.where(df['spleen_low'] == 1, 2, np.where(df['spleen_high'] == 1, 4, 1))
any_injury_weight = np.where(df['any_injury'] == 1, 6, 1)

for idx in tqdm(range(len(df))):
    patient_id = df.loc[idx, 'patient_id']
    dicom_folder = df.loc[idx, 'dicom_folder']
    image_path = dicom_folder.replace('/kaggle/input/rsna-2023-abdominal-trauma-detection/', 'data/') + '.png'
    # bowel = 0 if df.loc[idx, 'bowel_healthy'] == 1 else 1
    # extravasation = 0 if df.loc[idx, 'extravasation_healthy'] == 1 else 1
    # kidney = 1 if df.loc[idx, 'kidney_low'] == 1 else (2 if df.loc[idx, 'kidney_high'] == 1 else 0)
    # liver = 1 if df.loc[idx, 'liver_low'] == 1 else (2 if df.loc[idx, 'liver_high'] == 1 else 0)
    # spleen = 1 if df.loc[idx, 'spleen_low'] == 1 else (2 if df.loc[idx, 'spleen_low'] == 1 else 0)
    # data.append([image_path, patient_id, bowel, extravasation, kidney, liver, spleen])
    data.append(image_path)
column_drop = ['series_id', 'aortic_hu', 'incomplete_organ', 'dicom_folder', 'dicom_paths']
train_df = df.drop(columns=column_drop, axis=1)
train_df.insert(0, 'image_path', data)
train_df['bowel_weight'] = bowel_weight
train_df['extravasation_weight'] = extravasation_weight
train_df['kidney_weight'] = kidney_weight
train_df['liver_weight'] = liver_weight
train_df['spleen_weight'] = spleen_weight
train_df['any_injury_weight'] = any_injury_weight 
# train_df = pd.DataFrame(data, columns=['image_path', 'patient_id', 'bowel', 'extravasation', 'kidney', 'liver', 'spleen'])
# train_df.to_csv('data/train_image_ver2.csv', index=False)

Path('data/fold').mkdir(parents=True, exist_ok=True)
group_kfold =  GroupKFold(n_splits=5)
X = train_df.drop(columns=['image_path'])
groups = train_df['patient_id']
for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, groups=groups)):
    train_data = train_df.iloc[train_idx]
    val_data = train_df.iloc[val_idx]
    train_data.to_csv('data/fold/train_fold_{}.csv'.format(fold), index=False)
    val_data.to_csv('data/fold/val_fold_{}.csv'.format(fold), index=False)
print('sucessful..')
