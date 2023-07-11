import json
from nltk.tokenize import sent_tokenize
from typing import List
'''
Function with opens json file to data without any entity and relation labels
1. For each sentence in json file
2. Create append an empty list to our entity and relation lists
'''


def format_dataset_to_predict(json_file_path: str):
    new_json_data = []
    with open(json_file_path, 'r') as json_file:
        for i, document in enumerate(json_file):
            print(i)
            json_data = json.loads(document.strip())
            json_data['ner'] = []
            json_data['relations'] = []
            for _ in json_data['sentences']:
                json_data['ner'].append([])
                json_data['relations'].append([])
            new_json_data.append(json_data)
    return new_json_data


def write_new_output(json_data: List[dict], output_file_path: str):
    with open(output_file_path, 'w') as output_file:
        for entry in json_data:
            json.dump(entry, output_file, ensure_ascii=False)
            output_file.write('\n')


def verify_dataset(path_to_json_file: str):
    with open(path_to_json_file, 'r') as json_file:
        for i, document in enumerate(json_file):
            json_data = json.loads(document.strip())

            print(len(json_data['sentences']))
            print(len(json_data['ner']))
            print(len(json_data['relations']))
            print("\n\n\n")


'''
Function that takes in a long input of string, tokenizes string by sentence and formats into our dataset
'''


def convert_unstructured_text_to_dataset_input(text: str) -> dict:
    sentences = sent_tokenize(text)
    json_data = {}
    json_data['sentences'] = sentences
    json_data['ner'] = []
    json_data['relations'] = []
    for sent_idx, sentence in enumerate(json_data['sentences']):
        json_data['ner'].append([])
        json_data['relations'].append([])
    return json_data


'''
Convert json file to csv file

csv file in format of 
subj,rel,obj
bk.%AmziClarenceDixon_Person,rel:authorOf,bk.%TheFundamentals_Writing
bk.%AmziClarenceDixon_Person,rel:denominationIs,bk.%Baptists
bk.%AmziClarenceDixon_Person,rel:keyPersonOf,bk.tle:AmericanEvangelicalism
bk.%AmziClarenceDixon_Person,rel:leaderOf,bk.%MoodyChurch_Place
bk.%AmziClarenceDixon_Person,rel:proponentOf,bk.%Fundamentalism_Belief
bk.%AmziClarenceDixon_Person,rel:roleIs,bk.%editor

where subj and obj are the entity labels and rel is the relation label
'''

# def convert_json_file_to_csv_file(json_file_path: str, output_file_path: str):
#     with open(json_file_path, 'r') as json_file:
#         with open(output_file_path, 'w') as output_file:
#             for line in json_file:
#                 json_data = json.loads(line.strip())
#                 for sent_idx, sentence in enumerate(json_data['sentences']):
#                     for relation in json_data['relations'][sent_idx]:
#                         subj_index = relation[0]
#                         rel = relation[1]
#                         obj_index = relation[2]

#                         #get entity name from sentence index

#                         #look up entity label from faithlife map using entity name

#                         output_file.write(f'{subj},{rel},{obj}\n')

new_data = format_dataset_to_predict('../NER-RL_models/data/no_label.json')
write_new_output(new_data, 'no_label_formatted.json')
verify_dataset('no_label_formatted.json')