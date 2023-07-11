import pandas as pd

from markdown_preprocessing.article_structs import ArticleReader
from faithlife_utils import load_faithlife_database_to_single_df, make_faithlife_map, make_faithlife_map_2
from dygiee_preprocessing.faithlife_tokenizer import FaithlifeTokenizer
from dygiee_preprocessing.entity_labeler import EntityLabeler
from dygiee_preprocessing.relation_labeler import RelationLabeler
from dygiee_preprocessing.dygiee_writer import DYGIEEWriter
from dygiee_preprocessing.sentence_aligner import SentenceAligner
from dygiee_preprocessing.data_filter import DataFilter


class PreprocessFaithlifeArticles:

    def __init__(
        self,
        database_df: pd.DataFrame,
        articleReader: ArticleReader,
        output_path: str = 'faithlife_data',
    ):
        self.articleReader = articleReader
        self.entity_label_to_entity_type = make_faithlife_map(database_df)
        self.faithlife_map_2 = make_faithlife_map_2(database_df)
        self.output_path = output_path
        self.tokenizer = FaithlifeTokenizer()
        self.entity_labeler = EntityLabeler(self.tokenizer,
                                            self.entity_label_to_entity_type,
                                            database_df)
        self.sentence_aligner = SentenceAligner(self.tokenizer)
        self.relation_labeler = RelationLabeler()
        self.data_filter = DataFilter(entity_labels_to_filter=[
            'NONE', 'event', 'denom', 'denomgroup', 'thing', 'artifact'
        ],
                                      max_span_length=8)
        self.dygiee_writer = DYGIEEWriter("../faithlife_model_data")
        self.entity_name_to_entity_label_and_entity_type = self.get_all_maps()

    def get_all_maps(self):
        final_map = {}
        for article in self.articleReader.name_to_articles.values():
            article_map = self.articleReader.create_entity_name_to_entity_label_and_entity_type(
                article)
            final_map.update(article_map)
            print("len of final_map", len(final_map))
        return final_map

    def process_articles(self):
        for article_name, article in self.articleReader.name_to_articles.items(
        ):
            print("article_name:", article.metadata.identifier)
            # Temp until chicago manual of style is fixed
            self.articleReader.filter_article_section(article,
                                                      'Recommended Reading')
            self.articleReader.filter_article_section(article, 'See Also')

            # entity_name_to_entity_label_and_entity_type = self.articleReader.create_entity_name_to_entity_label_and_entity_type(
            #     article)
            all_sentences, ner_tuples = self.articleReader.merge_all_article_info(
                article, self.entity_name_to_entity_label_and_entity_type)

            tokenized_sentences = self.tokenizer.tokenize_article_sentences(
                all_sentences)

            ner_tuples = self.entity_labeler.label_ner_for_sentences(
                ner_tuples, tokenized_sentences,
                self.entity_name_to_entity_label_and_entity_type,
                self.faithlife_map_2, article.metadata.identifier)

            ner_tuples = self.sentence_aligner.convert_character_indices_to_sentence_indices(
                tokenized_sentences, ner_tuples)

            # ner_tuples = self.entity_labeler.label_events(
            #     ner_tuples, article.metadata.identifier)

            ner_tuples = self.data_filter.remove_labeled_sentence_tuple(
                ner_tuples)

            relation_map = self.relation_labeler.search_csv_for_relation(
                article_name, tokenized_sentences, ner_tuples)
            relation_tuples = self.relation_labeler.populate_relation_field_for_article(
                relation_map)

            #TODO - Maybe delete later?
            article.ner_tuples = ner_tuples
            article.tokenized_sentences = tokenized_sentences
            article.relations = relation_tuples

            self.articleReader.save_article_to_csv(article)
            self.articleReader.transform_csv_to_json_data(article)
            output_docs = self.dygiee_writer.write_to_jsonl(article)

            self.dygiee_writer.write_labeled_data_to_json(
                output_docs, article.metadata.identifier)
