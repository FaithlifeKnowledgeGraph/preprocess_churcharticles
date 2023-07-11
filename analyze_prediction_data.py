import json
import pathlib
import random
from typing import List
from collections import Counter
from preprocessing.faithlife_utils import load_faithlife_database_to_single_df, make_faithlife_map_2

prediction_path = 'prediction_data/'

# [[0, 0, 'concept', 'bk.%pagans', 'Paganism']]

entities_missed = Counter()
entities_mislabeled_index = Counter()
entities_mislabeled_type = Counter()


class Predictions:

    def __init__(self, path_to_predictions: str) -> None:
        self.path_to_predictions = path_to_predictions
        self.total_entities = 0
        self.fp = 0
        self.tp = 0
        self.fn = 0

        self.entity_not_identified = 0
        self.entity_identified_correctly = 0
        self.entity_identified = 0

        self.baseline_true = 0
        self.baseline_false = 0
        self.baseline_total = 0
        self.load_faithlife_concept_map()
        self.read_file()

    def load_faithlife_concept_map(self):
        path_to_faithlife_db_folder = "preprocessing/faithlife_data/entities"
        faithlife_df = load_faithlife_database_to_single_df(
            path_to_faithlife_db_folder)
        self.faithlife_map = make_faithlife_map_2(faithlife_df)

    def calculate_unique_entities(self):
        unique_entities_count = 0
        with open(self.path_to_predictions, 'r') as f:
            for line in f:
                json_obj = json.loads(line.strip())
                sentence_index = 0
                for sent_idx, sentence in enumerate(json_obj['sentences']):

                    for predicted_entity_tuples in json_obj['predicted_ner'][
                            sent_idx]:
                        entity_name_tokenized = sentence[
                            predicted_entity_tuples[0] -
                            sentence_index:predicted_entity_tuples[1] -
                            sentence_index + 1]

                        entity_name = " ".join(entity_name_tokenized).rstrip()
                        if entity_name in self.faithlife_map and self.faithlife_map[
                                entity_name][1] != predicted_entity_tuples[2]:
                            unique_entities_count += 1
                            # print("unique entity:", entity_name, "\n"
                            #       "type:", predicted_entity_tuples[2], "\n\n")

                    sentence_index += len(sentence)
        return unique_entities_count

    def read_file(self):
        with open(self.path_to_predictions, 'r') as f:
            for line in f:
                json_obj = json.loads(line.strip())
                doc_key = json_obj['doc_key']
                print("\n\n\n", "article name:", doc_key)
                sentence_index = 0

                self.total_entities += sum([len(x) for x in json_obj['ner']])

                for sent_idx, sentence in enumerate(json_obj['sentences']):
                    # print("sentence_index:", sentence_index)
                    # print("sentence:", sentence)
                    for predicted_entity_tuples in json_obj['predicted_ner'][
                            sent_idx]:
                        predicted_entity_tuples.append(sentence[
                            predicted_entity_tuples[0] -
                            sentence_index:predicted_entity_tuples[1] -
                            sentence_index + 1])
                        # print("predicted entities:", predicted_entity_tuples)
                    for entity_tuples in json_obj['ner'][sent_idx]:
                        pass

                        # print("real entities:", entity_tuples)
                    sentence_index += len(sentence)
                    # print("\n\n")
                    self.check_positives(json_obj['predicted_ner'][sent_idx],
                                         json_obj['ner'][sent_idx])
                    self.check_negatives(json_obj['predicted_ner'][sent_idx],
                                         json_obj['ner'][sent_idx])

                    self.calculate_entity_identification(
                        json_obj['predicted_ner'][sent_idx],
                        json_obj['ner'][sent_idx])

                    self.check_baseline(json_obj['ner'][sent_idx])

    def print_results(self):
        print("entities missed:", entities_missed, "\n\n")
        print("entities_mislabeled_index:", entities_mislabeled_index, "\n\n")
        print("entities_mislabeled_type:", entities_mislabeled_type, "\n\n")
        print("total_entities", self.total_entities)
        print("true_positives", self.tp)
        print("false_positives", self.fp)
        print("false_negatives", self.fn)
        print("precision", self.tp / (self.tp + self.fp))
        print("recall", self.tp / (self.tp + self.fn))
        print("f1", 2 * self.tp / (2 * self.tp + self.fp + self.fn))
        print("entity_identified_correctly", self.entity_identified_correctly)
        print("entity_identified", self.entity_identified)
        print("entity_not_identified", self.entity_not_identified)
        print("entit_identification_total:",
              self.entity_identified_correctly / self.entity_identified)
        print("baseline_true", self.baseline_true)
        print("baseline_false", self.baseline_false)
        print("baseline_total", self.baseline_true / self.baseline_total)

    def check_baseline(
        self,
        real_entities: List[List[str]],
    ):

        for real_entity in real_entities:
            real_start_index, real_end_index, real_entity_type, label, real_entity_name = real_entity
            if real_entity_name in self.faithlife_map:
                self.baseline_true += 1
            else:
                self.baseline_false += 1
            self.baseline_total += 1

    def calculate_entity_identification(self,
                                        predicted_entities: List[List[str]],
                                        real_entities: List[List[str]]):
        for real_entity in real_entities:
            real_start_index, real_end_index, real_entity_type, label, real_entity_name = real_entity
            if not self.correct_entity_real(predicted_entities,
                                            real_entity_name, real_start_index,
                                            real_end_index, real_entity_type):
                # print("false negative", real_entity_name, real_entity[-1])
                self.entity_not_identified += 1

            else:
                self.entity_identified_correctly += 1
            self.entity_identified += 1

    def check_negatives(self, predicted_entities: List[List[str]],
                        real_entities: List[List[str]]):
        for real_entity in real_entities:
            real_start_index, real_end_index, real_entity_type, label, real_entity_name = real_entity
            if not self.correct_entity_real(predicted_entities,
                                            real_entity_name, real_start_index,
                                            real_end_index, real_entity_type):
                # print("false negative", real_entity_name, real_entity[-1])
                entities_missed[real_entity_name] += 1
                self.fn += 1

    def check_positives(self, predicted_entities: List[List[str]],
                        real_entities: List[List[str]]):
        '''Check positive entitie examples

        1. Iterate through all predicted entities
        2. For each predicted entity, compare against all real entities
        3. 
        
        
        '''

        for predicted_entity in predicted_entities:
            pred_start_index, pred_end_index, pred_entity_type, pred_entity_name_list = predicted_entity
            # if pred_entity_type not in real_entities[]
            pred_entity_name = " ".join(pred_entity_name_list).rstrip()

            if self.correct_entity(real_entities, pred_entity_name,
                                   pred_start_index, pred_end_index,
                                   pred_entity_type):
                self.tp += 1

            # If the entity name was in the real entities
            # Now check if the reason was 1) mislabeled index or 2) mislabeled type
            elif self.check_if_entity_name_in_real_entities(
                    real_entities, pred_entity_name):

                if self.check_if_there_is_a_mislabeled_index(
                        real_entities, pred_entity_name, pred_start_index,
                        pred_end_index):
                    entities_mislabeled_index[pred_entity_name] += 1
                    print("mislabeled_index",
                          (pred_entity_name, pred_start_index, pred_end_index),
                          real_entities, "\n")
                    self.fp += 1
                elif self.check_if_there_is_a_mislabeled_type(
                        real_entities, pred_entity_name, pred_start_index,
                        pred_end_index, pred_entity_type):
                    print("mislabeled_type",
                          (pred_entity_name, pred_entity_type), real_entities,
                          "\n")
                    entities_mislabeled_type[pred_entity_name] += 1
                    self.fp += 1
            else:
                print("false positive", "pred_entity_name:", pred_entity_name,
                      real_entities, "\n")
                self.fp += 1

    def correct_entity_real(
        self,
        pred_entities,
        real_entity_name,
        real_start_index,
        real_end_index,
        real_entity_type,
    ):
        for pred_entity in pred_entities:
            # Missed these entities completelty
            if real_entity_name == " ".join(
                    pred_entity[3]) and real_start_index == pred_entity[
                        0] and real_end_index == pred_entity[
                            1] and real_entity_type == pred_entity[2]:
                return True
        return False

    def correct_entity(
        self,
        real_entities,
        pred_entity_name,
        pred_start_index,
        pred_end_index,
        pred_entity_type,
    ):
        for real_entity in real_entities:
            # Missed these entities completelty
            if pred_entity_name == real_entity[
                    4] and pred_start_index == real_entity[
                        0] and pred_end_index == real_entity[
                            1] and pred_entity_type == real_entity[2]:
                # print("correct entity", pred_entity_name, real_entity[-1])
                return True
        return False

    def check_if_entity_name_in_real_entities(self, real_entities,
                                              pred_entity_name):
        for real_entity in real_entities:
            # Missed these entities completelty
            if pred_entity_name == real_entity[4]:
                return True
        return False

    def check_if_there_is_a_mislabeled_index(self, real_entities,
                                             pred_entity_name,
                                             pred_start_index, pred_end_index):
        for real_entity in real_entities:
            # Missed these entities completelty
            if pred_entity_name == real_entity[
                    4] and pred_start_index == real_entity[
                        0] and pred_end_index == real_entity[1]:
                return False
        return True

    def check_if_there_is_a_mislabeled_type(
        self,
        real_entities,
        pred_entity_name,
        pred_start_index,
        pred_end_index,
        pred_entity_type,
    ):
        for real_entity in real_entities:
            # Missed these entities completelty
            if pred_entity_name == real_entity[
                    4] and pred_start_index == real_entity[
                        0] and pred_end_index == real_entity[
                            1] and pred_entity_type == real_entity[2]:
                return False
        return True


pred = Predictions(prediction_path + 'run_16.json')
pred.load_faithlife_concept_map()
unique_entity = pred.calculate_unique_entities()
print("unique_entity:", unique_entity)
pred.print_results()

print("number of entities missed:", sum(entities_missed.values()))
print("number of entities missed index:",
      sum(entities_mislabeled_index.values()))
print("number of entity type mislabeled:",
      sum(entities_mislabeled_type.values()))
