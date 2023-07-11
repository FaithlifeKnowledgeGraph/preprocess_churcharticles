from markdown_preprocessing.article_structs import NERTuple
from .preprocess_faithlife_articles import FaithlifeTokenizer

from typing import List


class SentenceAligner:

    def __init__(self, tokenizer: FaithlifeTokenizer):
        self.tokenizer = tokenizer

    def convert_character_indices_to_sentence_indices(
        self,
        tokenized_sentences: List[List[str]],
        ner_tuples: List[List[NERTuple]],
    ) -> List[List[NERTuple]]:
        '''Converts character indices of entity to sentence indices.

            Args:
                article: Data structure for faithlife article data.

            Returns:
                Article data structure with labeled sentence tuple overidden with sentence indices


            Example Output:
            sentence: "The Protestant Reformation challenged traditional sources of authority
            and religious dogma (1517–1648)."

            previous_labeled_sentence: [[62, 70, "concept", "bk.%authority", "authority"], 
                                        [77, 12, "concept", "bk.%dogma", "dogma"]]
            
            labeled_sentence:  [[7, 7, "concept", "bk.%authority", "authority"], 
                                [10, 10, "concept", "bk.%dogma", "dogma"]]
            '''

        # labeled sentences is a bad named, because it implies multiple sentences
        # but its actually just 1 sentence with multiple ner labels

        ner_tuples_processed = []
        last_sent_index = 0
        for i, ner_tuples in enumerate(ner_tuples):
            ner_tuple_in_sentence = []

            for j, ner_tuple in enumerate(ner_tuples):

                ner_tuples_with_sentence_indices = self.get_sentence_indices(
                    last_sent_index, tokenized_sentences[i], ner_tuple)

                if ner_tuples_with_sentence_indices != None:
                    ner_tuple_in_sentence.append(
                        ner_tuples_with_sentence_indices)
                    #fix later, should map to a new section and not override previous one
            last_sent_index += len(tokenized_sentences[i])
            ner_tuples_processed.append(ner_tuple_in_sentence)

        return ner_tuples_processed

    def get_sentence_indices(
        self,
        prev_sent_index: int,
        tokenized_sentence: List[str],
        ner_tuple: NERTuple,
    ) -> NERTuple:
        ''' Create new tuple with sentence indices swapped out for character indices. 

            TODO - Investiage robustness, sometimes tokenization is off and this relies on 1:1 match
            for example, if it needs to find '1475', but word is tokenized as 'd. 1475'. Right now, we
            can just throw away labels

            Args:
                tokenized_sentence: List of tokens corresponding to a sentence from the article.
                labeled_sentence: Tuple containing ner information in the following format

                Format - start_index, end_index, entity_type, entity_label, sentence
                Ex - [0, 24, 'NONE', 'The Renaissance in Europe', 'bk.tle%3arenaissanceeurope']

            Returns:
                Tuple where char indices are replaced with sentence indices


            Ex:
            tokenized_sentence - ['Schism', 'separated', 'the', 'Russian', 'Orthodox', 'Church', 'and', 
            'the', 'Old', 'Believers', 'over', 'church', 'reforms', '(', '1652–1725', ')', '.']

            labeled_sentence: [53, 65, 'concept', 'bk.%Raskolniki', 'Old Believers']
            sentence_indices: [8, 9, 'concept', 'bk.%Raskolniki', 'Old Believers']

            **lowercasing is applied to line up with database
            '''
        # print("tokenized_sentnece:", tokenized_sentence)
        lower_case_tokenized_sentence = [
            word.lower() for word in tokenized_sentence
        ]
        # print("tokenized_sentence:", lower_case_tokenized_sentence)
        # print('ner_tuple:', ner_tuple)
        entity = ner_tuple.entity_name
        # print("entity:", entity)

        # print('entity:', entity)
        tokenized_entity = self.tokenizer.tokenize_entity(entity)
        lower_case_tokenized_entity = [
            word.lower() for word in tokenized_entity
        ]
        # entity = entity.split(" ")
        # print('new_entity:', lower_case_tokenized_entity)
        entity_index = 0
        total_entity_words = len(lower_case_tokenized_entity)
        # print('total_entity_words', total_entity_words)
        start_entity_sent = 0
        end_entity_sent = 0

        if self.is_subsequence(lower_case_tokenized_entity,
                               lower_case_tokenized_sentence):
            start_entity_sent, end_entity_sent = 0, len(
                lower_case_tokenized_entity) - 1
        else:
            return NERTuple(start_entity_sent + prev_sent_index,
                            end_entity_sent + prev_sent_index,
                            ner_tuple.entity_type, ner_tuple.entity_label,
                            ner_tuple.entity_name)

            # return ([
            #     start_entity_sent + prev_sent_index,
            #     end_entity_sent + prev_sent_index, ner_tuple.entity_type,
            #     ner_tuple.entity_label, ner_tuple.entity_name
            # ])

        for i in range(
                len(lower_case_tokenized_sentence) -
                len(lower_case_tokenized_entity) + 1):
            # print(
            #     'searching for:',
            #     lower_case_tokenized_sentence[i:i +
            #                                   len(lower_case_tokenized_entity)])

            if lower_case_tokenized_sentence[
                    i:i + len(lower_case_tokenized_entity
                             )] == lower_case_tokenized_entity:
                start_entity_sent, end_entity_sent = i, i + len(
                    lower_case_tokenized_entity) - 1
                # print('start:', start_entity_sent + prev_sent_index, 'end:',
                #       end_entity_sent + prev_sent_index)

                return NERTuple(start_entity_sent + prev_sent_index,
                                end_entity_sent + prev_sent_index,
                                ner_tuple.entity_type, ner_tuple.entity_label,
                                ner_tuple.entity_name)
                # return ([
                #     start_entity_sent + prev_sent_index,
                #     end_entity_sent + prev_sent_index, ner_tuple.entity_type,
                #     ner_tuple.entity_label, ner_tuple.entity_name
                # ])

        # tokenized_entity = tokenize_entity(labeled_sentence[4])
        # print("tokenized_entity:", tokenized_entity)
        # for i in range(len(tokenized_sentence) - len(tokenized_entity) + 1):
        #     if tokenized_sentence[i:i +
        #                           len(tokenized_entity)] == tokenized_entity:
        #         start_entity_sent, end_entity_sent = i, i + len(
        #             tokenized_entity) - 1
        #         return ([
        #             start_entity_sent + prev_sent_index,
        #             end_entity_sent + prev_sent_index, labeled_sentence[2],
        #             labeled_sentence[3], labeled_sentence[4]
        #         ])

        print(f'did not find: {tokenized_entity} in {entity}')
        return None

    def is_subsequence(
        self,
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
