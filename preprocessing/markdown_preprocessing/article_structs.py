import csv
import json
import ast
import re
import pandas as pd

from dataclasses import dataclass, field
from typing import List, Sequence, Mapping, Tuple
from nltk.tokenize import sent_tokenize
from faithlife_utils import extract_name
from dygiee_preprocessing.faithlife_tokenizer import FaithlifeTokenizer

CACHE_FOLDER = 'cache_processed_data'
CACHE_MODEL_FOLDER = 'cache_model_data'

tokenizer = FaithlifeTokenizer()


@dataclass
class NERTuple:
    '''Dataclass class for 
    named entity recognition (NER) tuple'''

    start_index: int
    end_index: int
    entity_type: str
    entity_label: str
    entity_name: str = ""

    def __iter__(self):
        yield self.start_index
        yield self.end_index
        yield self.entity_type
        yield self.entity_label
        yield self.entity_name

    def __repr__(self) -> str:
        return f'NERTuple{tuple(self)}'


@dataclass
class RelationTuple:
    '''Dataclass class for relation tuple'''
    subject_start_index: int
    subject_end_index: int
    object_start_index: int
    object_end_index: int
    relation_type: str

    def __iter__(self):
        yield self.subject_start_index
        yield self.subject_end_index
        yield self.object_start_index
        yield self.object_end_index
        yield self.relation_type

    def __repr__(self) -> str:
        return f'RelationTuple{tuple(self)}'


@dataclass
class ArticleSubsection:
    '''Dataclass class for article sub section (h3 tag)
    
    Example:
    ### [Explorers](bk.%explorer)
    * [Vasco Nuñez de Balboa]...
    * [Juan Ponce de León]...
    
    ### [Missionaries](https://ref.ly/logos4/Factbook?ref=bk.%25evangelist)
    * [Juan De Zumárraga]...
    '''

    section_title: Tuple[str] = ("", "")
    bullet_points: List[str] = field(default_factory=list)
    ner_tuples: Mapping[str, List[Tuple[str]]] = field(default_factory=dict)
    tokenized_sentences: Sequence[str] = field(default_factory=list)


@dataclass
class ArticleSection:
    '''Dataclass class for article main section (h2 tag)
    
    Example:
    ## Key Developments
    * [European countries]...
    * Invading armies perpetuated...
    * The [Atlantic slave trade]...
    * [Reform]...
    ## Key People
    '''

    section_title: str = ""
    bullet_points: List[str] = field(default_factory=list)
    article_subsections: Sequence[ArticleSubsection] = field(
        default_factory=list)
    ner_tuples: Mapping[str, List[Tuple[str]]] = field(default_factory=dict)
    tokenized_sentences: Sequence[str] = field(default_factory=list)


@dataclass
class ArticleSummary:
    '''Dataclass class for article summary section (h1 tag)
    
    Example:

    * **Period:** AD [1480–1776](date.1480-1776)
    * **Description:** The Roman Catholic Church expanded...

    ## Summary
    While the Protestant Reformation was taking shape in...
    '''

    period: str = ""
    description: str = ""
    text: str = ""


@dataclass
class ArticleMetadata:
    '''Dataclass class for an Article's metadata
    Example: 

    Level: 1
    Status: Released
    Identifier: CatholicBeyondEurope
    Parents: Root
    Eras: bk.era:medievalchurchera; bk.era:reformationera; 
    bk.era:modernera
    '''
    level: int = 0
    status: str = ""
    identifier: str = ""
    parent: str = ""
    eras: List[str] = field(default_factory=list)


@dataclass()
class Article:
    '''Dataclass for Church History Article'''
    metadata: ArticleMetadata = ArticleMetadata()
    title: str = ""
    summary: ArticleSummary = ArticleSummary()
    article_sections: Sequence[ArticleSection] = field(default_factory=list)
    entity_name_to_entity_label_and_entity_type: dict[str, str] = field(
        default_factory=dict)

    sentences: List[List[str]] = field(default_factory=list)
    ner_tuples: List[List[NERTuple]] = field(default_factory=list)
    relations: List[List[RelationTuple]] = field(default_factory=list)
    tokenized_sentences: List[List[str]] = field(default_factory=list)


@dataclass
class ArticlesDYGIEE:
    '''Dataclass to load DYGIEE data'''
    identifier: str = ""
    tokenized_sentences: List[Tuple[str]] = field(default_factory=list)
    ner_tuples: List[Tuple[str]] = field(default_factory=list)
    relation_tuples: List[Tuple[str]] = field(default_factory=list)


class ArticleReader:
    """Class that manages reading and processing of all Bible Article.

    Attributes:
        articles: List of Article objects.
    """

    def __init__(self, articles: List[Article],
                 database_df: pd.DataFrame) -> None:
        self.name_to_articles = self._map_articles_by_name(articles)
        self.articles = articles
        self.database_df = database_df
        self.no_relations = [
            'OrganizationUnderBishops', 'RiseNaziGermanyDividesChurch',
            'WorshipEarlyChurch', 'Pelagian', "ChristianApologists",
            "ChurchAndState", 'SporadicPersecution', 'ChineseChurchGrows',
            'Puritanism'
        ]

    def _map_articles_by_name(
            self, articles: Sequence[Article]) -> Mapping[str, Article]:
        """Creates a map of article names to Article
        
        Args:
            Article: List of all Bible Article.

        Returns:
            A map of article name to article object.
        """
        name_to_articles = {}
        for article in articles:
            article_name = article.metadata.identifier
            name_to_articles[article_name] = article
        return name_to_articles

    def filter_article_section(self, article: Article, section_name: str):
        for i, article_section in enumerate(article.article_sections):
            if article_section.section_title == section_name:
                del article.article_sections[i]

    def create_entity_name_to_entity_label_and_entity_type(
            self, article: Article):
        entity_name_to_entity_label_and_entity_type = {}
        if article.metadata.identifier not in self.no_relations:
            relations_df = pd.read_csv(
                f'faithlife_data/church-history-Articles-en/{article.metadata.identifier}.csv'
            )
            subj_list = set(relations_df['subj'].tolist())
            obj_list = set(relations_df['obj'].tolist())
            all_entities_in_history_article_csv = subj_list.union(obj_list)

            for entity_label in all_entities_in_history_article_csv:
                entity_name = self.database_df.loc[
                    self.database_df['primary'] ==
                    entity_label]['label_en'].to_string(index=False)
                if '(' in entity_name:
                    entity_name = extract_name(entity_name)

                entity_type = self.database_df.loc[
                    self.database_df['primary'] ==
                    entity_label]['kind'].to_string(index=False)
                entity_name_to_entity_label_and_entity_type[entity_name] = (
                    entity_label, entity_type)
        return entity_name_to_entity_label_and_entity_type

    def merge_all_article_info(
        self, article: Article, entity_name_to_entity_label_and_entity_type
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Merges all sentences in article into one list

        Args:
            article: An article object.
        """
        # if article.metadata.identifier != "OrganizationUnderBishops":
        #     return [], []
        all_sentences = [[article.metadata.identifier], [article.title]]
        ner_tuples = [[(
            0,
            len(article.metadata.identifier) - 1,
            article.metadata.identifier,
            f'bk.tle:{article.metadata.identifier}',
        )], []]

        #calculate length of all_sentences where all_sentences = [[sentence1], [sentence2], ...] using list comprehension

        # period
        all_sentences.append(["Period: " + article.summary.period])
        ner_tuples.append([])

        # description
        for i, sentence_in_description in enumerate(
                sent_tokenize(article.summary.description)):
            ner_tuples_in_sentence = []
            if i == 0:
                all_sentences.append(
                    ["Description: " + sentence_in_description])
            else:
                all_sentences.append([sentence_in_description])

            for entity_name, (
                    entity_label, entity_type
            ) in entity_name_to_entity_label_and_entity_type.items():
                if entity_name in sentence_in_description:
                    if is_subsequence([
                            entity_name
                    ], tokenizer.tokenize_sentence(sentence_in_description)):

                        entity_label, entity_type = entity_name_to_entity_label_and_entity_type[
                            entity_name]

                        start_index = sentence_in_description.find(entity_name)
                        end_index = start_index + len(entity_name) - 1

                        ner_tuples_in_sentence.append(
                            (start_index, end_index, entity_name,
                             entity_label))

            ner_tuples.append(ner_tuples_in_sentence)

        # Summary
        for i, sentence_in_text in enumerate(
                sent_tokenize(article.summary.text)):
            ner_tuples_in_sentence = []
            if i == 0:
                all_sentences.append(["Summary: " + sentence_in_text])
            else:
                all_sentences.append([sentence_in_text])

            for entity_name, (
                    entity_label, entity_type
            ) in entity_name_to_entity_label_and_entity_type.items():
                if entity_name in sentence_in_text:
                    if is_subsequence(
                        [entity_name],
                            tokenizer.tokenize_sentence(sentence_in_text)):

                        entity_label, entity_type = entity_name_to_entity_label_and_entity_type[
                            entity_name]

                        start_index = sentence_in_text.find(entity_name)
                        end_index = start_index + len(entity_name) - 1

                        ner_tuples_in_sentence.append(
                            (start_index, end_index, entity_name,
                             entity_label))

            ner_tuples.append(ner_tuples_in_sentence)

        for article_section in article.article_sections:
            # TODO - Parse chicago citation for recommended reading
            # if article_section.section_title == 'Recommended Reading':
            #     print(article_section.bullet_points)
            #     for sentence in article_section.bullet_points:
            #         test = parse_chicago_citation(sentence)
            #         print("chicago:", test)
            #         # all_sentences.append(sentence)
            #         # ner_tuples.append([])

            if type(article_section.section_title) == tuple:
                all_sentences.append([article_section.section_title[0]])
                ner_tuples.append([article_section.section_title[1]])
            else:
                all_sentences.append([article_section.section_title])
                ner_tuples.append([])
            for article_subsection in article_section.article_subsections:
                if type(article_subsection.section_title) == tuple:

                    all_sentences.append([article_subsection.section_title[0]])

                    ner_tuples.append([article_subsection.section_title[1]])
                else:
                    all_sentences.append([article_subsection.section_title])
                    ner_tuples.append([])
                for i, sentence in enumerate(article_subsection.bullet_points):
                    all_sentences.append([sentence])
                    ner_tuples.append(article_subsection.ner_tuples[i])
            for i, article_section_sentence in enumerate(
                    article_section.bullet_points):
                all_sentences.append([article_section_sentence])
                ner_tuples.append(article_section.ner_tuples[i])

        # print(len(all_sentences), len(ner_tuples))
        # for i, sentence in enumerate(all_sentences):
        # print("setence: ", sentence, "ner_tuple:", ner_tuples[i], "\n")
        return all_sentences, ner_tuples

    def write_all_articles_to_sentence_csv(self) -> None:
        for article in self.articles:
            all_sentences, ner_tuples = self.merge_all_article_info(article)
            self.write_article_sentence_to_csv(article.metadata.identifier,
                                               all_sentences)
            self.write_ner_tuples_to_csv(article.metadata.identifier,
                                         ner_tuples)

    def write_article_sentence_to_csv(self, article_name: str,
                                      sentences: List[str]) -> None:
        with open(
                f'markdown_preprocessing/cache_article_data/{article_name}_sentences.csv',
                'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["sentence"])
            for sent_idx, sentence_list in enumerate(sentences):
                for sentence in sentence_list:
                    # print("test:", sentence)
                    writer.writerow([sentence])

    def write_ner_tuples_to_csv(self, article_name: str,
                                ner_tuples: List[List[str]]):
        print("writing ner_tuples:")
        print(ner_tuples)
        with open(
                f'markdown_preprocessing/cache_article_data/{article_name}_ner_tuples.csv',
                'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["ner_tuple"])
            for sent_idx, ner_tuple_list in enumerate(ner_tuples):
                print(ner_tuple_list)
                for ner_tuple in ner_tuple_list:
                    # print("test:", sentence)
                    writer.writerow([ner_tuple])

    def save_article_to_csv(self, article: Article):
        self.write_article_tokenize_sentences_to_csv(article)
        self.write_article_entities_to_csv(article)
        self.write_article_relations_to_csv(article)

    def write_article_tokenize_sentences_to_csv(self,
                                                article: Article) -> None:
        """Saves article to csv file

        Args:
            article: An article object.
        """

        with open(
                f'{CACHE_FOLDER}/{article.metadata.identifier}_tokenized_sentences.csv',
                'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["tokenized_sentence"])
            for sent_idx, sentence_list in enumerate(
                    article.tokenized_sentences):
                writer.writerow([sentence_list])

    def write_article_entities_to_csv(self, article: Article) -> None:
        with open(f'{CACHE_FOLDER}/{article.metadata.identifier}_entities.csv',
                  'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "sentence_indice", "start_index", "end_index", "entity_type",
                "entity_label", "entitty_name"
            ])
            for sentence_indice, entity_tuples in enumerate(
                    article.ner_tuples):
                for entity_tuple in entity_tuples:
                    writer.writerow([
                        sentence_indice + 1, entity_tuple.start_index,
                        entity_tuple.end_index, entity_tuple.entity_type,
                        entity_tuple.entity_label, entity_tuple.entity_name
                    ])

    def write_article_relations_to_csv(self, article: Article) -> None:

        with open(
                f'{CACHE_FOLDER}/{article.metadata.identifier}_relations.csv',
                'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "sentence_indice", "subject_start_index", "subject_end_index",
                "object_start_index", "object_end_index", "relation_type"
            ])
            for sentence_indice, relation_tuples in enumerate(
                    article.relations):
                for relation_tuple in relation_tuples:
                    subject_start_index, subject_end_index, object_start_index, object_end_index, relation_type = relation_tuple
                    writer.writerow([
                        sentence_indice + 1, subject_start_index,
                        subject_end_index, object_start_index,
                        object_end_index, relation_type
                    ])

    def transform_csv_to_json_data(self, article: Article) -> None:
        """Transforms csv file to json data

        Args:
            article_identifer: Name of article.
        """
        with open(
                f'{CACHE_FOLDER}/{article.metadata.identifier}_sentences.csv',
                'r') as csv_file:
            reader = csv.reader(csv_file)
            sentences = []
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                sentences.append(ast.literal_eval(row[0]))

        with open(f'{CACHE_FOLDER}/{article.metadata.identifier}_entities.csv',
                  'r') as csv_file:
            reader = csv.reader(csv_file)
            entities = []
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                entities.append(
                    [int(row[1]),
                     int(row[2]), row[3], row[4], row[5]])

        with open(
                f'{CACHE_FOLDER}/{article.metadata.identifier}_relations.csv',
                'r') as csv_file:
            reader = csv.reader(csv_file)
            relations = []
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                relations.append([
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                    int(row[4]), row[5]
                ])

        with open(f'{CACHE_MODEL_FOLDER}/{article.metadata.identifier}.json',
                  'w') as json_file:
            json.dump(
                {
                    "doc_key": article.metadata.identifier,
                    "sentences": sentences,
                    "ner": entities,
                    "relations": relations
                },
                json_file,
                ensure_ascii=False)

    def load_article_from_csv_to_Articles_DYGIEE(
            self, article_identifier: str) -> ArticlesDYGIEE:
        """Loads csv file to ArticlesDYGIEE object

        Args:
            article_identifier: Name of article.

        Returns:
            ArticlesDYGIEE object.
        """
        with open(f'{CACHE_FOLDER}/{article_identifier}_sentences.csv',
                  'r') as csv_file:
            reader = csv.reader(csv_file)
            sentences = []
            for row in reader:
                sentences.append(row[0])

        with open(f'{CACHE_FOLDER}/{article_identifier}_entities.csv',
                  'r') as csv_file:
            reader = csv.reader(csv_file)
            entities = []
            for row in reader:
                entities.append({
                    "sentence_indice": row[0],
                    "start_index": row[1],
                    "end_index": row[2],
                    "entity_type": row[3],
                    "entity_label": row[4],
                    "entity_name": row[5]
                })

        with open(f'{CACHE_FOLDER}/{article_identifier}_relations.csv',
                  'r') as csv_file:
            reader = csv.reader(csv_file)
            relations = []
            for row in reader:
                relations.append({
                    "sentence_indice": row[0],
                    "subject_start_index": row[1],
                    "subject_end_index": row[2],
                    "object_start_index": row[3],
                    "object_end_index": row[4],
                    "relation_type": row[5]
                })

        return ArticlesDYGIEE(sentences=sentences,
                              entities=entities,
                              relations=relations)


def parse_chicago_citation(citation):
    """
    Parses a Chicago style citation in the format "Article Title" in Book Title (Book Author).

    Parameters:
    citation (str): The Chicago style citation string to be parsed.

    Returns:
    dict: A dictionary containing the extracted information, with keys for the article title, book title,
    and book author.
    """
    # print(citation)
    pattern = r'"(.+)" in (.+?)( \((?:ed\. )?([^)]+)\))?'

    match = re.match(pattern, citation)
    if match:
        title, publication, _, editor = match.groups(default="")
        return {"title": title, "publication": publication, "editor": editor}
    else:
        return None


def is_subsequence(
    subseq,
    sentence,
):
    subseq_pos = 0  # Position of next character to match in subsequence
    for char in sentence:
        if char == subseq[subseq_pos]:
            subseq_pos += 1
            if subseq_pos == len(subseq):  # Reached end of subsequence
                return True
    return False
