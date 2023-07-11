import json
import csv

schema_for_results = [
    'doc_key', 'sent_index', 'sentence', 'ner_label', 'predicted_ner',
    'relation_label'
]
schema_for_train = [
    'doc_key', 'sent_index', 'sentence', 'ner_label', 'relation_label'
]

input_path_to_results = 'faithlife_data/processed_data/test_results.json'
input_path_to_train = 'faithlife_data/processed_data/faithlife_data.jsonl'

output_path = 'faithlife_processed_data.csv'
row_data = []

with open(input_path_to_train, 'r') as f:
    for line in f:
        json_obj = json.loads(line.strip())
        doc_key = json_obj['doc_key']

        # yield json_obj
        all_sentences = json_obj['sentences']
        for i, sentence in enumerate(all_sentences):
            line_data = []
            line_data.append(doc_key)
            line_data.append(i)
            line_data.extend(sentence)
            for ner_in_sentence in json_obj['ner'][i]:
                print(ner_in_sentence)
                copy_line = line_data.copy()
                line_data.append(ner_in_sentence)
                row_data.append(line_data)
                line_data = copy_line
    # for relations_in_sentence in json_obj['relations'][i]:
    #     print(relations_in_sentence)

with open(output_path, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(schema_for_train)
    writer.writerows(row_data)