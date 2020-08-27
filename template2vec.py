import pandas as pd
import pickle

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
eventid_vectors = []
for event_id, template, occurrences in df_template.iloc:
    eventid_vectors.append(
        ' '.join(map(str, template2vec([template], embedding_table, counter_idf)[0])))
df_template['Vector'] = eventid_vectors


# convert templates to vectors for all logs
vector_structured = []
for template in df_structured['EventTemplate']:
    try:
        vector_structured.append(
            df_template.loc[df_template['EventTemplate'] == template, 'Vector'][0])
    except Exception:
        # new template
        vector_structured.append(
            ' '.join(map(str, template2vec([template], embedding_table, counter_idf)[0])))
df_structured['Vector'] = vector_structured
df_structured.to_csv('preprocessed_data/' + structured_file_name, index=False)
