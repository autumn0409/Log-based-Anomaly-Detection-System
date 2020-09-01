import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import re


structured_file_name = 'HDFS.log_structured.csv'

# load data
df_structured = pd.read_csv('parse_result/' + structured_file_name)
df_label = pd.read_csv('log_data/anomaly_label.csv')
vectors = list(np.load('preprocessed_data/vectors.npy'))

# remove unused column
df_structured.drop(columns=['Date', 'Time', 'Pid', 'Level', 'Component',
                            'Content', 'EventId', 'EventTemplate'], axis=1, inplace=True)

# append vectors
df_structured['Vector'] = vectors
del vectors


# extract BlockId
r1 = re.compile('^blk_-?[0-9]')
r2 = re.compile('.*blk_-?[0-9]')

paramlists = df_structured['ParameterList'].tolist()
blk_id_list = []
for paramlist in tqdm(paramlists, desc='extract BlockId'):
    paramlist = ast.literal_eval(paramlist)
    blk_id = list(filter(r1.match, paramlist))

    if len(blk_id) == 0:
        filter_str_list = list(filter(r2.match, paramlist))
        # ex: '/mnt/hadoop/mapred/system/job_200811092030_0001/job.jar. blk_-1608999687919862906'
        blk_id = filter_str_list[0].split(' ')[-1]
    else:
        # ex: ['blk_-1608999687919862906'], ['blk_-1608999687919862906', 'blk_-1608999687919862906'],
        # ['blk_-1608999687919862906 terminating']
        blk_id = blk_id[0].split(' ')[0]

    blk_id_list.append(blk_id)

df_structured['BlockId'] = blk_id_list
df_structured.drop(columns=['ParameterList'], axis=1, inplace=True)


# split training and testing data labels
df_label['Usage'] = 'testing'

n_index = df_label.Label[df_label.Label.eq('Normal')].sample(6000).index
a_index = df_label.Label[df_label.Label.eq('Anomaly')].sample(6000).index
train_index = n_index.union(a_index)
df_label.iloc[train_index, df_label.columns.get_loc('Usage')] = 'training'

df_structured = pd.merge(df_structured, df_label, on='BlockId')
del df_label


# group data by BlockId
df_structured.sort_values(by=['BlockId', 'LineId'], inplace=True)
df_structured.drop(columns=['LineId'], axis=1, inplace=True)


# split training and testing dataframe
df_test = df_structured[df_structured['Usage'] == 'testing']
df_train = df_structured[df_structured['Usage'] == 'training']
del df_structured


# training data preprocessing
x_train, y_train = [], []
max_timesteps = 0
pbar = tqdm(total=df_train['BlockId'].nunique(),
            desc='training data collection')

while len(df_train) > 0:
    blk_id = df_train.iloc[0]['BlockId']
    last_index = 0
    for i in range(len(df_train)):
        if df_train.iloc[i]['BlockId'] != blk_id:
            break
        last_index += 1

    df_blk = df_train[:last_index]
    x_train.append(np.array(df_blk['Vector'].tolist()))
    if max_timesteps < x_train[-1].shape[0]:
        max_timesteps = x_train[-1].shape[0]

    y_index = int(df_blk.iloc[0]['Label'] == 'Anomaly')
    y = [0, 0]
    y[y_index] = 1
    y_train.append(y)

    df_train = df_train.iloc[last_index:]
    pbar.update()
pbar.close()


np.savez('preprocessed_data/training_data.npz',
         x_train=x_train, y_train=y_train)
del x_train, y_train


# testing data preprocessing
x_test, y_test = [], []
max_timesteps = 0
pbar = tqdm(total=df_test['BlockId'].nunique(), desc='testing data collection')

while len(df_test) > 0:
    blk_id = df_test.iloc[0]['BlockId']
    last_index = 0
    for i in range(len(df_test)):
        if df_test.iloc[i]['BlockId'] != blk_id:
            break
        last_index += 1

    df_blk = df_test[:last_index]
    x_test.append(np.array(df_blk['Vector'].tolist()))
    if max_timesteps < x_test[-1].shape[0]:
        max_timesteps = x_test[-1].shape[0]

    y_index = int(df_blk.iloc[0]['Label'] == 'Anomaly')
    y = [0, 0]
    y[y_index] = 1
    y_test.append(y)

    df_test = df_test.iloc[last_index:]
    pbar.update()
pbar.close()


np.savez('preprocessed_data/testing_data.npz',
         x_test=x_test, y_test=y_test)
