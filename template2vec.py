import pandas as pd
import pickle
from tqdm import tqdm

from utils import template2vec


template_file_name = 'HDFS.log_templates.csv'
structured_file_name = 'HDFS.log_structured.csv'

# load data
df_template = pd.read_csv('parse_result/' + template_file_name)
df_structured = pd.read_csv('parse_result/' + structured_file_name, dtype=str)
with open('preprocessed_data/counter_idf.pkl', 'rb') as inputfile:
    counter_idf = pickle.load(inputfile)
with open('preprocessed_data/embedding_table.pkl', 'rb') as inputfile:
    embedding_table = pickle.load(inputfile)


# calculate vectors for all known templates
df_template['Vector'] = template2vec(df_template['EventTemplate'].tolist(), embedding_table, counter_idf)


# convert templates to vectors for all logs
vector_structured = []
for template in tqdm(df_structured['EventTemplate']):
    try:
        vector_structured.append(
            df_template.loc[df_template['EventTemplate'] == template, 'Vector'][0])
    except Exception:
        # new template
        vector_structured.append(template2vec([template], embedding_table, counter_idf)[0])
df_structured['Vector'] = vector_structured
df_structured.to_csv('preprocessed_data/' + structured_file_name, index=False)
