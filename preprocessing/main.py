# Main class to run classes and prototype

from loadFaithlifeData import LoadFaithlifeData
from markdown_preprocessing.processMarkdownData import MarkdownDataProcessor
from markdown_preprocessing.processBooksWithNERAnnotations import BooksWithNERAnnotationProcesser
from markdown_preprocessing.article_structs import ArticleReader
from dygiee_preprocessing.preprocess_faithlife_articles import PreprocessFaithlifeArticles, FaithlifeTokenizer
from faithlife_utils import load_faithlife_database_to_single_df

from typing import List
import os
import pathlib
import json


def pipeline_1():
    faithlifeData = LoadFaithlifeData(path_to_church_articles='faithlife_data')
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
    PATH_TO_BOOKS = "faithlife_data/books-with-NER-annotations-modified/*.md"

    faithlifeData = LoadFaithlifeData(path_to_church_articles='faithlife_data')
    books = BooksWithNERAnnotationProcesser(PATH_TO_BOOKS).process_html()
    create_dataset_from_books(books)


# TODO: This should be integrated in the DYGIEWriter class instead
# TODO: Apply FaithlifeTokenizer to every sentence and add to tokenized_sentence
def create_dataset_from_books(books: List[object]):
    output_dir = "faithlife_ner_model_data/"
    all_docs = []
    tokenizer = FaithlifeTokenizer()
    for book in books:
        f'Writing {book.book_name} to jsonl'

        book.tokenized_sentence = [[] for _ in range(len(book.sentences))]

        for i, sentence in enumerate(book.sentences):
            print(sentence[0])
            book.tokenized_sentence[i] = tokenizer.tokenize_sentence(
                sentence[0])

        output_json_dict = {}
        output_json_dict["doc_key"] = book.book_name
        output_json_dict["sentences"] = book.tokenized_sentence
        ner_tuples = book.ner_tuples
        unpacked_ner_tuples = [
            [tuple(nt) for nt in inner_list] for inner_list in ner_tuples
        ]

        output_json_dict["ner"] = unpacked_ner_tuples
        output_json_dict["relations"] = [[] for _ in range(len(book.sentences))
                                        ]

        all_docs.append(output_json_dict)

        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_output = os.path.join(output_dir, book.book_name)
        with open(f'{file_output}.jsonl', 'w') as outfile:
            for entry in all_docs:
                json.dump(entry, outfile, ensure_ascii=False)
                outfile.write('\n')


# pipeline_1()
pipeline_2()
