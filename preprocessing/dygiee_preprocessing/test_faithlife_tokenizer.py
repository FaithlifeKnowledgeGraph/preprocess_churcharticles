import unittest
import csv
import os

from dygiee_preprocessing.faithlife_tokenizer import FaithlifeTokenizer


class TestFaithlifeTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = FaithlifeTokenizer()
        self.sentences = self.read_test_data()

    def read_test_data(self):
        sentences = []
        with open(
                './dygiee_preprocessing/test_data/OrganizationUnderBishops_sentences.csv',
                newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                sentences.append(row)
        return sentences

    # def test_correct_number_of_sentences(self):
    #     self.assertEqual(len(self.sentences), 3)

    # def test_tokenizer(self):
    #     tokenized_sentences = self.tokenizer.tokenize_article_sentences(
    #         self.sentences)


if __name__ == '__main__':
    unittest.main()