# Main class to run classes and prototype

from loadChurchMarkdownData import LoadChurchData
from markdown_preprocessing.processMarkdownData import MarkdownDataProcessor
from markdown_preprocessing.processBooksWithNERAnnotations import BooksWithNERAnnotationProcesser
from markdown_preprocessing.article_structs import ArticleReader
from dygiee_preprocessing.preprocess_faithlife_articles import PreprocessFaithlifeArticles
from faithlife_utils import load_faithlife_database_to_single_df


def pipeline_1():
    faithlifeData = LoadChurchData(path_to_church_articles='faithlife_data',
                                   entity_folder='entities')
    processed_articles = MarkdownDataProcessor(faithlifeData,
                                               True).process_html()

    path_to_faithlife_db = 'faithlife_data/entities'
    database_df = load_faithlife_database_to_single_df(path_to_faithlife_db)
    articleReader = ArticleReader(processed_articles, database_df)
    output_path = "../faithlife_model_data/faithlife_data"

    preprocesser = PreprocessFaithlifeArticles(database_df, articleReader,
                                               output_path)
    preprocesser.process_articles()


def pipeline_2():
    faithlifeData = LoadChurchData(path_to_church_articles='faithlife_data',
                                   entity_folder='entities')
    test = BooksWithNERAnnotationProcesser(faithlifeData).read_data()


# pipeline_2()
pipeline_1()