import json
import pathlib
import random
from typing import List
from collections import Counter

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


def read_article(input_path: str):
    entity_example_map = Counter()
    concept_count = 0
    writing_count = 0
    person_count = 0
    place_count = 0
    span_count_longer_than_spec = 0
    total_num_of_relations = 0
    with open(input_path, 'r') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            doc_key = json_obj['doc_key']
            print("article name:", doc_key)
            num_of_sentences = extract_number_of_sentences(json_obj)
            print('num_of_sentences:', num_of_sentences)
            character_count_of_each_sentence = extract_character_count_for_sentences(
                json_obj)
            total_character_count_for_doc = get_total_character_count_for_doc(
                character_count_of_each_sentence)
            print('total_character_count_for_doc:',
                  total_character_count_for_doc)
            average_character_count = get_average_character_count_for_doc(
                character_count_of_each_sentence)
            print('average_character_count:', average_character_count)

            span_lenght_of_all_ner = extract_span_length_for_ner(json_obj)
            print('span_length of all ner:', span_lenght_of_all_ner)

            span_count_longer_than_spec += calculate_num_of_span_longer_than_spec(
                json_obj, 8)
            entity_types = ['concept', 'writing', 'person', 'place']
            entity_type_dict = {}
            for entity_type in entity_types:
                count_of_concept_entity_type = extract_entity_type_count(
                    json_obj, entity_type)
                entity_type_dict[entity_type] = count_of_concept_entity_type
            print(entity_type_dict, "\n")
            extract_entity_example_count(json_obj, entity_example_map)

            concept_count += entity_type_dict['concept']
            writing_count += entity_type_dict['writing']
            person_count += entity_type_dict['person']
            place_count += entity_type_dict['place']

            num_of_relation_in_doc = extract_number_of_relations(json_obj)
            print("num_of_relation_in_doc:", num_of_relation_in_doc, "\n")
            total_num_of_relations += num_of_relation_in_doc
    print('person_count:', person_count)
    print('writing_count:', writing_count)
    print('place_count:', place_count)
    print('concept_count:', concept_count)
    print("relation_count:", total_num_of_relations)
    print("entit_example_map:", entity_example_map.most_common(100))
    print("span_counter_longer than spec:", span_count_longer_than_spec)


def extract_number_of_sentences(json_obj: dict) -> int:
    return len(json_obj['sentences'])


def extract_character_count_for_sentences(json_obj: dict) -> List[int]:
    sentences_character_count = []
    for sentence in json_obj['sentences']:
        sentences_character_count.append(len(sentence))
    return sentences_character_count


def get_total_character_count_for_doc(
        sentences_character_count: List[int]) -> int:
    return sum(sentences_character_count)


def get_average_character_count_for_doc(
        sentences_character_count: List[int]) -> float:
    return sum(sentences_character_count) / len(sentences_character_count)


def extract_span_length_for_ner(json_obj: dict) -> List[int]:
    ner_span_length_count = []
    for ner in json_obj['ner']:
        if ner:
            for ner_tuple in ner:
                span_length = ner_tuple[1] - ner_tuple[0]
                ner_span_length_count.append(span_length)
    return ner_span_length_count


def extract_entity_type_count(json_obj: dict, entity_type: str) -> int:
    count = 0
    for ner in json_obj['ner']:
        if ner:
            for ner_tuple in ner:
                if ner_tuple[2] == entity_type:
                    count += 1
    return count


def extract_entity_example_count(json_obj: dict, entity_map: dict) -> None:
    for ner in json_obj['ner']:
        if ner:
            for ner_tuple in ner:
                if ner_tuple[4] not in entity_map:
                    entity_map[ner_tuple[4]] = 0
                entity_map[ner_tuple[4]] += 1


def extract_number_of_relations(json_obj: dict) -> int:
    return len(json_obj['relations'])


def calculate_num_of_span_longer_than_spec(json_obj: dict, spec: int) -> int:
    count = 0
    for ner in json_obj['ner']:
        if ner:
            for ner_tuple in ner:
                span_length = ner_tuple[1] - ner_tuple[0]
                if span_length > spec:
                    count += 1
    return count


read_article(output_train_path)
read_article(output_dev_path)
read_article(output_test_path)