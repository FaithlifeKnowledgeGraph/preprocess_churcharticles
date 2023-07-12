import os
import pytest
import sys

sys.path.append("../..")

from unittest.mock import patch
from preprocessing.loadFaithlifeData import LoadFaithlifeData

TEST_PATH_TO_DATA = 'fathlife_data'
TEST_CHURCH_HISTORY_FOLDER = 'church-history-Articles-en'

from preprocessing.loadFaithlifeData import LoadFaithlifeData


@patch('glob.glob')
def test_load_files(mock_glob):
    test_files = [
        f'{TEST_PATH_TO_DATA}/{TEST_CHURCH_HISTORY_FOLDER}/file1.csv',
        f'{TEST_PATH_TO_DATA}/{TEST_CHURCH_HISTORY_FOLDER}/file2.csv',
        f'{TEST_PATH_TO_DATA}/{TEST_CHURCH_HISTORY_FOLDER}/Relations.csv'
    ]

    mock_glob.return_value = test_files

    loader = LoadFaithlifeData(TEST_PATH_TO_DATA)

    result_files = loader._load_files(TEST_CHURCH_HISTORY_FOLDER, '*.csv',
                                      ['Relations.csv'])

    assert f'{TEST_PATH_TO_DATA}/{TEST_CHURCH_HISTORY_FOLDER}/file1.csv' in result_files
    assert f'{TEST_PATH_TO_DATA}/{TEST_CHURCH_HISTORY_FOLDER}/file2.csv' in result_files
    assert f'{TEST_PATH_TO_DATA}/{TEST_CHURCH_HISTORY_FOLDER}/Relations.csv' not in result_files