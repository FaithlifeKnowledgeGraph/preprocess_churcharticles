import pandas as pd
from markdown_preprocessing.article_structs import Article, NERTuple, RelationTuple
from typing import List


class RelationLabeler:

    def __init__(self) -> None:
        pass

    def search_for_corresponding_obj_label(
        self,
        obj_relation: str,
        ner_tuples: List[NERTuple],
    ) -> NERTuple:
        '''Search for a corresponding obj label in ner_tuples to complete relation extraction.

        This presumes that a subj label exists in our ner_tuples so we attempt to complete the relation
        by finding a corresponding obj_label in ner_tuples.

        Args:
            obj_relation: Entity label for obj relation from csv file.
            ner_tuples: List of tuples where indices are [start, end, entity_type, entity_label, sentence]

        Returns:
            If it exists, the ner_tuple which has a corresponding obj_label that matches subj_label
            to complete the relation.
        
        
        '''
        for ner_tuple in ner_tuples:
            if ner_tuple.entity_label == obj_relation:
                return ner_tuple
        return ()

    def search_csv_for_relation(
        self,
        article_name: str,
        tokenized_sentences: List[str],
        ner_tuples: List[List[NERTuple]],
    ) -> dict[int, List[str]]:
        '''Use article's annotation to extract relations.
        
        Given an article's annotation, we iterate through the ner_tuples and find rows which
        align with the article's annotation. Then, we attempt to find a relation between 
        subject and object. So, we assert that for a subject that there exists an entity label
        in our ner_tuples with a corresponding object type that matches a given relation
        

        ner_tuple = [start_index, end_index, entity_type, entity_label, sentence]
        Args:
            article: Data structure for faithlife article data.
            article_name: Identifier of article.

        Returns:
            Article with relations field filled out.
        '''

        no_relations = [
            'OrganizationUnderBishops', 'RiseNaziGermanyDividesChurch',
            'WorshipEarlyChurch', 'Pelagian', "ChristianApologists",
            "ChurchAndState", 'SporadicPersecution', 'ChineseChurchGrows',
            'Puritanism'
        ]
        binary_flag = True  #training
        relation_map = {}
        if article_name not in no_relations:
            relations_df = pd.read_csv(
                f'faithlife_data/church-history-Articles-en/{article_name}.csv')

            relation_map = {
                i: [] for i in range(len(tokenized_sentences))
            }  #cache sentence length
            for sentence_index, ner_tuples_in_sentence in enumerate(ner_tuples):
                for ner_tuple in ner_tuples_in_sentence:
                    cells_where_entity_label_exists_as_subj = relations_df.loc[
                        relations_df['subj'] == ner_tuple.entity_label]
                    for _, t in cells_where_entity_label_exists_as_subj.iterrows(
                    ):
                        obj_tuple = self.search_for_corresponding_obj_label(
                            t.obj, ner_tuples_in_sentence)

                        if obj_tuple and binary_flag:
                            relation_map[sentence_index].append([
                                ner_tuple.start_index, ner_tuple.end_index,
                                obj_tuple.start_index, obj_tuple.end_index,
                                "rel:TRUE"
                            ])
                        elif obj_tuple:
                            relation_map[sentence_index].append([
                                ner_tuple.start_index, ner_tuple.end_index,
                                obj_tuple.start_index, obj_tuple.end_index,
                                t.rel
                            ])
                        else:
                            pass

        return relation_map

    def populate_relation_field_for_article(
        self,
        relation_map: dict[str, str],
    ) -> List[List[RelationTuple]]:
        relation_tuples = []
        for sentence_index, relations in relation_map.items():
            relation_tuples.append([])
            for relation in relations:
                relation_tuples[sentence_index].append(relation)
        return relation_tuples