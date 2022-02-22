import pandas as pds
import os
import psycopg2
from dotenv import load_dotenv
from scipy.io.matlab import savemat
from scipy.signal import iirnotch, filtfilt
from sqlalchemy import create_engine
import numpy as np

# Create an engine instance
load_dotenv('config.env')
url = os.environ.get('DATABASE_URL')
alchemyEngine = create_engine(url)
# Connect to PostgreSQL server

dbConnection = alchemyEngine.connect()

# Read data from PostgreSQL database table and load into a DataFrame instance

df = pds.read_sql("select * from bci_collection", dbConnection)

subject_id = 'S02'
collection_ids = df.loc[df['collector_name'] == subject_id, 'id']

subject_data = []

for collection_id in collection_ids:
    df = pds.read_sql(f"SELECT * FROM collected_data WHERE collection_id = {collection_id}", dbConnection)
    df.drop(['id', 'collection_time', 'collection_id', 'character', 'phase'], axis=1, inplace=True)
    subject_data.append(df)

# print(len(subject_data))
all_data = []
b, a = iirnotch(60.0, 30.0, fs=250.0)
for i in range(len(subject_data)):
    block_data = []
    freq_list = []
    for k, data in subject_data[i].groupby('frequency'):
        freq_list.append(k)
        d = data.sort_values('order').iloc[:1250, :-2].drop_duplicates()
        data = filtfilt(b, a, d.to_numpy().T)
        block_data.append(data.T)
        # print(np.shape(block_data))
    block_data = np.array(block_data, dtype=object)
    all_data.append(block_data)
all_data = np.array(all_data)
all_data = np.transpose(all_data, axes=(3, 2, 1, 0))
savemat('Sub02.mat', {'data': all_data})

# Close the database connection

dbConnection.close()
