# Contains Classes associated with loading faithlife data.

from typing import List
import glob

CHURCH_HISTORY_FOLDER = 'church-history-Articles-en'
BOOKS_NER_ANNOTATION_FOLDER = 'books-with-NER-annotations'
ENTITY_FOLDER = 'entities'

EXCLUDED_CSV_FILES = ['Relations.csv', 'Roles.csv']
EXCLUDED_MD_FILES = ['ReadMe.md']


class LoadFaithlifeData:

    def __init__(self, path_to_church_articles: str):
        """Class that manages loading church history article data, annotations, 
        and faithlife database.

        Attributes:
            path_to_church_articles: Path to church history article data.
            entity_folder: Path to faithlife database.
            relation_data_files: All csv files containing relation data from church history articles.
            markdown_data_files: All markdown files for articles from church history articles.
            books_ner_annotation: All markdown files for books with NER annotations.
        """
        self.path_to_church_articles = path_to_church_articles
        self.relation_data_files = self._load_files(CHURCH_HISTORY_FOLDER,
                                                    '*.csv',
                                                    EXCLUDED_CSV_FILES)
        self.markdown_data_files = self._load_files(CHURCH_HISTORY_FOLDER,
                                                    '*.md', EXCLUDED_MD_FILES)
        self.books_ner_annotation = self._load_files(
            BOOKS_NER_ANNOTATION_FOLDER, '*.md', None)
        self.entity_data = self._load_files(ENTITY_FOLDER, '*.csv', None)

    def _load_files(
        self,
        folder: str,
        pattern: str,
        excluded_files: List[str],
    ) -> List[str]:
        """Load specified files from a directory, excluding some files if specified.
        
        Args:
            folder: The name of the folder to load files from.
            pattern: The pattern to match files against.
            excluded_files: The files to exclude from the list of files to load.

        Returns:
            A list of file paths that match the pattern and are not excluded.
        """
        file_list = glob.glob(
            f'{self.path_to_church_articles}/{folder}/{pattern}')

        if excluded_files is not None:
            for filename in excluded_files:
                try:
                    file_list.remove(
                        f'{self.path_to_church_articles}/{folder}/{filename}')
                except ValueError:
                    print(
                        f'File {filename} not found in {folder} folder. Continuing without it'
                    )
        return file_list
