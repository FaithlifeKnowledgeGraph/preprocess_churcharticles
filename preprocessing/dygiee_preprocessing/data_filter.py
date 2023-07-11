from markdown_preprocessing.article_structs import NERTuple
from typing import List


class DataFilter:

    def __init__(
        self,
        entity_labels_to_filter: List[str] = [],
        max_span_length: int = 8,
    ):
        self.entity_labels_to_filter = entity_labels_to_filter  #condition 1
        self.max_span_length = max_span_length  #condition 3

    def remove_labeled_sentence_tuple(
        self,
        ner_tuples: List[List[NERTuple]],
    ) -> List[List[NERTuple]]:
        '''Removes labeled ner's (labeled_sentence tuple) with following conditions
        
        Conditions to remove:
        1) Event, denom, denomgroup entity types 
        2) The sentence indices are -1, which means we could not properly
        map character indices to sentence indices
        3) The entity span is longer than 8 words

        article.labeled_sentence = [sent_start_index, sent_end_index, entity_type, entity_label, sentence]
        
        Args:
            article: Data structure for faithlife article data.

        Returns:
            article data structure where labeled sentence indices removed that were
            filtered out.
        '''
        # print(
        #     f'Filtering out NER tuples with following entity types: {self.entity_labels_to_filter}\n'
        # )
        # print(
        #     f'Filtering out NER tuples with span length > {self.max_span_length}\n'
        # )

        for i, ner_tuples_in_sentence in enumerate(ner_tuples):
            index_to_delete = []
            for index, ner_tuple in enumerate(ner_tuples_in_sentence):

                # condition 1
                if ner_tuple.entity_type in self.entity_labels_to_filter:
                    index_to_delete.append(index)

                # condition 2
                if ner_tuple.start_index == -1 or ner_tuple.end_index == -1:
                    index_to_delete.append(index)

                # condition 3
                if ner_tuple.end_index - ner_tuple.start_index > self.max_span_length:
                    index_to_delete.append(index)

            new_ner_tuples = [
                i for j, i in enumerate(ner_tuples_in_sentence)
                if j not in index_to_delete
            ]
            ner_tuples[i] = new_ner_tuples
        return ner_tuples