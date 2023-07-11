import unittest
import csv
import ast

from dygiee_preprocessing.entity_labeler import EntityLabeler
from dygiee_preprocessing.faithlife_tokenizer import FaithlifeTokenizer
from faithlife_utils import load_faithlife_database_to_single_df, make_faithlife_map

path_to_faithlife_db = 'faithlife_data/entities'
path_to_tokenized_sentence = './dygiee_preprocessing/test_data/OrganizationUnderBishops_tokenized_sentences.csv'
path_to_ner_tuples = './dygiee_preprocessing/test_data/OrganizationUnderBishops_entities.csv.csv'


class TestEntityLabeler(unittest.TestCase):

    def setUp(self) -> None:
        self.tokenizer = FaithlifeTokenizer()
        self.tokenized_sentences = self.read_test_tokenized_sentence_data()
        self.ner_tuples = self.read_test_entity_data()
        self.database_df = load_faithlife_database_to_single_df(
            path_to_faithlife_db)

        self.entity_label_to_entity_type = make_faithlife_map(self.database_df)
        self.entity_labeler = EntityLabeler(self.tokenizer,
                                            self.entity_label_to_entity_type,
                                            self.database_df)

    def read_test_tokenized_sentence_data(self) -> None:
        tokenized_sentences = []
        with open(
                './dygiee_preprocessing/test_data/OrganizationUnderBishops_tokenized_sentences.csv',
                newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                tokenized_sentences.append(ast.literal_eval(row[0]))
        return tokenized_sentences

    def read_test_entity_data(self):
        ner_tuples = []
        with open(
                './dygiee_preprocessing/test_data/OrganizationUnderBishops_ner_tuples.csv',
                newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                ner_tuples.append([tuple(ast.literal_eval(row[0]))])
        return ner_tuples

    def test_label_ner_for_sentences(self):

        ner_tuples = self.entity_labeler.label_ner_for_sentences(
            self.ner_tuples, self.tokenized_sentences,
            'OrganizationUnderBishops')
        for ner_tuple in ner_tuples:
            print(ner_tuple, "\n")


if __name__ == '__main__':
    unittest.main()