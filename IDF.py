import pandas as pd
import pickle

from utils import calculate_freq, template2tokens


file_name = 'HDFS.log_templates.csv'

df_template = pd.read_csv('parse_result/' + file_name)
counter_idf = calculate_freq(template2tokens(df_template['EventTemplate']), mode_idf=True)
with open('preprocessed_data/counter_idf.pkl', 'wb') as outputfile:
    pickle.dump(counter_idf, outputfile)
