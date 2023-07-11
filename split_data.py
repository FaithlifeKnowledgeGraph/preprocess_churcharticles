import json
import pathlib
import random
import os

# input_path = 'faithlife_model_data/faithlife_data.jsonl'
input_path = '../faithlife_model_data'
output_train_path = '../NER-RL_models/data/train.json'
output_dev_path = '../NER-RL_models/data/dev.json'
output_no_label_path = '../NER-RL_models/data/no_label.json'
output_test_path = '../NER-RL_models/data/test.json'

no_label_doc_keys = [
    'ChristianApologists', 'OrganizationUnderBishops',
    'RiseNaziGermanyDividesChurch', 'Pelagian', 'ChurchAndState',
    'SporadicPersecution', 'ChineseChurchGrows', 'Puritanism',
    'WorshipEarlyChurch'
]


def new_split(input_path: str):
    all_json_files = os.listdir(input_path)

    no_label_jsonl = []
    label_jsonl = []
    for json_file_path in all_json_files:
        file_path = os.path.join(input_path, json_file_path)
        with open(file_path, 'r') as f:
            for line in f:
                json_obj = json.loads(line.strip())
                doc_key = json_obj['doc_key']
                if doc_key in no_label_doc_keys:
                    no_label_jsonl.append(json_obj)
                else:
                    label_jsonl.append(json_obj)
    return no_label_jsonl, label_jsonl


def remove_no_label_data(input_path: str):
    no_label_jsonl = []
    label_jsonl = []
    with open(input_path, 'r') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            doc_key = json_obj['doc_key']
            if doc_key in no_label_doc_keys:
                no_label_jsonl.append(json_obj)
            else:
                label_jsonl.append(json_obj)
    return no_label_jsonl, label_jsonl


def split_and_shuffle_data(train_size: float, dev_size: float,
                           test_size: float):
    random.shuffle(label_jsonl)
    dev_size = train_size + dev_size
    train_data = label_jsonl[:int(len(label_jsonl) * train_size)]
    dev_data = label_jsonl[int(len(label_jsonl) *
                               train_size):int(len(label_jsonl) * dev_size)]
    test_data = label_jsonl[int(len(label_jsonl) * dev_size):]

    return train_data, dev_data, test_data


def write_jsonl_file(data: list, output_path: str):
    with open(output_path, 'w') as outfile:
        for entry in data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')


# no_label_jsonl, label_jsonl = remove_no_label_data(input_path)
no_label_jsonl, label_jsonl = new_split(input_path)

train_data, dev_data, test_data = split_and_shuffle_data(0.8, 0.1, 0.1)
write_jsonl_file(train_data, output_train_path)
write_jsonl_file(dev_data, output_dev_path)
write_jsonl_file(no_label_jsonl, output_no_label_path)
write_jsonl_file(test_data, output_test_path)