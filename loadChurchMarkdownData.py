import glob


class LoadChurchData:

    def __init__(self, path_to_church_articles: str, entity_folder: str):
        self.path_to_church_articles = path_to_church_articles
        self.entity_folder = entity_folder
        self.relations_metadata_filename = f'{self.path_to_church_articles}/church-history-Articles-en/Relations.csv'
        self.roles_metadata_filename = f'{self.path_to_church_articles}/church-history-Articles-en/Roles.csv'
        self.relation_data_files = self._load_relation_data()
        self.markdown_data_files = self._load_markdown_data()
        self.books_ner_annotation = self._load_books_with_ner_annotations()
        self.entity_data = self._load_entity_data()

    def _load_relation_data(self):
        annotated_data = glob.glob(
            f'{self.path_to_church_articles}/church-history-Articles-en/*.csv')
        annotated_data.remove(
            f'{self.path_to_church_articles}/church-history-Articles-en/Relations.csv'
        )
        annotated_data.remove(
            f'{self.path_to_church_articles}/church-history-Articles-en/Roles.csv'
        )
        return annotated_data

    def _load_entity_data(self):
        entity_data = glob.glob(
            f'{self.path_to_church_articles}/{self.entity_folder}/*.csv')
        return entity_data

    def _load_markdown_data(self):
        markdown_data = glob.glob(
            f'{self.path_to_church_articles}/church-history-Articles-en/*.md')
        markdown_data.remove(
            f'{self.path_to_church_articles}/church-history-Articles-en/ReadMe.md'
        )
        return markdown_data

    def _load_books_with_ner_annotations(self):
        return glob.glob(
            f'{self.path_to_church_articles}/books-with-NER-annotations/*.md')
