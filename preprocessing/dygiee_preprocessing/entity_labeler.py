import pandas as pd
from typing import List, Tuple, Mapping, Set

from markdown_preprocessing.article_structs import NERTuple
from faithlife_utils import search_entity_type_in_faithlife_map_with_entity_label
from dygiee_preprocessing.faithlife_tokenizer import FaithlifeTokenizer
from preprocessing_utils import replace_special_chars_in_entity_annotations


class EntityLabeler:

    def __init__(
        self,
        tokenizer: FaithlifeTokenizer,
        entity_label_to_entity_type: Mapping[str, str],
        database_df: pd.DataFrame,
    ):
        self.tokenizer = tokenizer
        self.entity_label_to_entity_type = entity_label_to_entity_type
        self.database_df = database_df
        self.no_relations = [
            'OrganizationUnderBishops', 'RiseNaziGermanyDividesChurch',
            'WorshipEarlyChurch', 'Pelagian', "ChristianApologists",
            "ChurchAndState", 'SporadicPersecution', 'ChineseChurchGrows',
            'Puritanism'
        ]

    def label_ner_for_sentences(
        self,
        ner_tuples: List[List[NERTuple]],
        sentences: List[List[str]],
        entity_name_to_entity_label_and_entity_type: Mapping[str, Tuple],
        faithlife_map_2: dict[str, Tuple[str, str]],
        article_identifer: str,
    ) -> List[List[NERTuple]]:
        '''Labels NER for sentences in Article using 3 methods.

            1) href hyperlinks in the html document - These are already in article.ner_tuples
            this method searches the faithlife database to find the corresponding entity type.

            2) Search for occurrence of NER in sentence using labeled csv file for article (unused)

            3) Iterate through every phrase in the faithlife database and search the given sentence
            to find a match (computationally expensive).


            Args:
                article: Data structure for faithlife article data.
                article_name: Identifier of the article.

            Returns:
                Article data structure with labeled sentence where

            Example Output:
            sentence: "The Protestant Reformation challenged traditional sources of authority
            and religious dogma (1517–1648)."

            labeled_sentence:  [[62, 70, "concept", "bk.%authority", "authority"], 
                                [77, 12, "concept", "bk.%dogma", "dogma"]]

            '''

        ner_tuples = self.find_entity_type_for_href_ner_tuples_in_article_struct(
            ner_tuples, article_identifer)

        ner_tuples = self.use_history_article_labels(
            ner_tuples, sentences, entity_name_to_entity_label_and_entity_type)

        ner_tuples = self.label_sentence_using_existing_ner_tuples(
            ner_tuples, sentences)

        ner_tuples = self.label_sentence_using_faithlife_concept_database(
            sentences, ner_tuples, faithlife_map_2)
        return ner_tuples

    def use_history_article_labels(
        self,
        ner_tuples: List[List[NERTuple]],
        sentences: List[str],
        entity_name_to_entity_label_and_entity_type: Mapping[str, Tuple],
    ):
        # print(len(sentences), len(ner_tuples))
        # print(sentences, "\n\n\n")
        # print(ner_tuples)
        if entity_name_to_entity_label_and_entity_type:
            for i, sentence in enumerate(sentences):
                already_labeled_entity_labels = set()
                for ner_tuple in ner_tuples[i]:
                    already_labeled_entity_labels.add(ner_tuple.entity_name)

                # sentence_as_string = " ".join(sentence).replace(' , ', ', ')
                for entity_name in entity_name_to_entity_label_and_entity_type.keys(
                ):
                    if entity_name in already_labeled_entity_labels:
                        continue

                    if self.is_subsequence([entity_name], sentence):
                        sentence_as_string = " ".join(sentence).replace(
                            ' , ', ', ').replace(' ( ', ' (').replace(
                                ' ; ', '; ').replace(' ) ',
                                                     ') ').replace(' : ', ': ')

                        entity_label, entity_type = entity_name_to_entity_label_and_entity_type[
                            entity_name]

                        start_index = sentence_as_string.find(entity_name)
                        if start_index != -1:
                            end_index = start_index + len(entity_name) - 1

                            is_overlap = False
                            for ner_tuples_in_sentence in ner_tuples[i]:
                                # If we extract a new entity that is a subset of an existing one

                                if (start_index
                                        >= ner_tuples_in_sentence.start_index
                                        and end_index
                                        <= ner_tuples_in_sentence.end_index
                                   ) or (start_index
                                         < ner_tuples_in_sentence.start_index
                                         and end_index
                                         > ner_tuples_in_sentence.end_index):
                                    is_overlap = True
                                    # print("detected_overlap", entity_name,
                                    #       ner_tuples_in_sentence.entity_name)

                            if is_overlap != True:
                                ner_tuples[i].append(
                                    NERTuple(start_index, end_index,
                                             entity_type, entity_label,
                                             entity_name))

            ner_tuples[i].sort(key=lambda ner_tuple:
                               (ner_tuple.start_index, ner_tuple.end_index))
            #             print("entity_labels_to_entity_name_and_entity_type:",
            #                   entity_labels_to_entity_name_and_entity_type)

            # print("entity_labels_to_entity_name_and_entity_type:",
            #       entity_labels_to_entity_name_and_entity_type)

            # ner_tuples = self.find_and_add_entities_from_history_article_labels_to_article_struct(
            #     ner_tuples, sentences, all_entities_in_history_article_csv)

        return ner_tuples

    def label_sentence_using_existing_ner_tuples(
        self,
        labeled_ner_tuples: List[NERTuple],
        sentences: List[List[str]],
    ):
        already_labeled_entity_labels = set()
        for ner_tuple in labeled_ner_tuples:
            if ner_tuple:
                ner_tuple = ner_tuple[0]
                already_labeled_entity_labels.add(
                    (ner_tuple.entity_name, ner_tuple.entity_label,
                     ner_tuple.entity_type, ner_tuple.start_index,
                     ner_tuple.end_index))

        for i, sentence in enumerate(sentences):
            for ner_tuple in already_labeled_entity_labels:
                entity_name, entity_label, entity_type, ner_start_index, ner_end_index = ner_tuple
                entity_name_as_sentence = entity_name.split(" ")

                if entity_name != 'Reason' and self.is_subsequence(
                        entity_name_as_sentence, sentence):
                    sentence_as_string = " ".join(sentence).replace(
                        ' , ',
                        ', ').replace(' ( ', ' (').replace(' ; ', '; ').replace(
                            ' ) ', ') ').replace(' : ', ': ')

                    start_index = sentence_as_string.find(entity_name)
                    if start_index != -1:
                        end_index = start_index + len(entity_name) - 1

                        # Check existing tuples if there is a tuple with the same name, start and end index
                        if self.check_to_see_if_ner_tuples_is_unique(
                                labeled_ner_tuples[i], start_index, end_index):

                            print("adding existing new tuple", ner_tuple)
                            print("entity name", entity_name)
                            print("sentence", sentence_as_string)
                            print("tuples before adding:",
                                  labeled_ner_tuples[i])
                            labeled_ner_tuples[i].append(
                                NERTuple(start_index, end_index, entity_type,
                                         entity_label, entity_name))
                            # Add now to our set
                            # already_labeled_entity_labels = already_labeled_entity_labels.copy(
                            # )
                            # already_labeled_entity_labels.add(
                            #     (start_index, end_index, entity_type, entity_label,
                            #      entity_name))
                            print("tuples after adding:", labeled_ner_tuples[i],
                                  "\n\n")

            if labeled_ner_tuples[i]:
                labeled_ner_tuples[i].sort(key=lambda ner_tuple: (
                    ner_tuple.start_index, ner_tuple.end_index))
        return labeled_ner_tuples

    def check_to_see_if_ner_tuples_is_unique(self, labeled_ner_tuples,
                                             start_index, end_index):
        # Check existing tuples if there is a tuple with the same name, start and end index
        for ner_tuple in labeled_ner_tuples:
            if ner_tuple.start_index == start_index and ner_tuple.end_index == end_index:
                return False
        return True

    def label_sentence_using_faithlife_concept_database(
        self,
        sentences: List[List[str]],
        labeled_ner_tuples: List[NERTuple],
        faithlife_map_2: dict[str, Tuple[str, str]],
    ) -> List[NERTuple]:

        set_of_bad_labels = {
            'All', 'all', 'See', 'see', 'A', 'a', 'Pure', 'Reason', 'New',
            "news", 'Man', 'Understanding', 'Long', 'By', 'Also', 'His',
            'Answer', 'General', 'Good'
            'Action', 'Middle', 'On', 'Our', 'Ann', '1876', 'Use', 'Six', 'Ten',
            'Description'
        }  #Maybe First and Third works here and also Life

        # This is a computationally expensive method
        # Filter database to only include our 4 entity types
        # Then, we will call is_subsequence on every phrase in the database to find matches in the sentence
        # We add the entity to the sentence if it is a match
        # And finally sort tuples at the end

        # create set of already labeled ner_tuples
        for i, sentence in enumerate(sentences):
            already_labeled_entity_labels = set()
            for ner_tuple in labeled_ner_tuples[i]:

                already_labeled_entity_labels.add(
                    (ner_tuple.entity_name, ner_tuple.start_index,
                     ner_tuple.end_index))

            for entity_name in faithlife_map_2.keys():
                entity_name_as_sentence = entity_name.split(
                    " "
                )  #TODO - Double check if this work or tokenization needed
                if entity_name not in set_of_bad_labels and self.is_subsequence(
                        entity_name_as_sentence, sentence):
                    entity_label, entity_type = faithlife_map_2[entity_name]
                    sentence_as_string = " ".join(sentence).replace(
                        ' , ',
                        ', ').replace(' ( ', ' (').replace(' ; ', '; ').replace(
                            ' ) ', ') ').replace(' : ', ': ')
                    start_index = sentence_as_string.find(entity_name)
                    if start_index != -1:
                        end_index = start_index + len(entity_name) - 1
                        if (entity_name, start_index,
                                end_index) not in already_labeled_entity_labels:
                            print("ner_tuple added from faithlife database",
                                  entity_name, entity_label, entity_type,
                                  start_index, end_index)
                            labeled_ner_tuples[i].append(
                                NERTuple(start_index, end_index, entity_type,
                                         entity_label, entity_name))

            if labeled_ner_tuples[i]:
                labeled_ner_tuples[i].sort(key=lambda ner_tuple: (
                    ner_tuple.start_index, ner_tuple.end_index))

        return labeled_ner_tuples

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

    def find_entity_type_for_href_ner_tuples_in_article_struct(
        self,
        ner_tuples: List[List[Tuple]],
        article_identifer: str,
    ) -> List[List[NERTuple]]:
        ner_tuples_with_entity_type = []
        for i, ner_labels_from_href_in_html in enumerate(ner_tuples):

            ner_tuples_in_sentence = []
            for sent_idx, ner_tuple_from_html in enumerate(
                    ner_labels_from_href_in_html):
                entity_label = replace_special_chars_in_entity_annotations(
                    ner_tuple_from_html[3])

                if ner_tuple_from_html[2] == article_identifer:
                    entity_type = "writing"

                # try to label "bk.bio" and "_Person" as 'person' entity type
                elif "bio." in entity_label or ("_" in entity_label and "Person"
                                                == entity_label.split("_")[1]):
                    entity_type = "person"

                elif "_" in entity_label and "Writing" == entity_label.split(
                        "_")[1]:
                    entity_type = "writing"

                elif "_" in entity_label and "Place" == entity_label.split(
                        "_")[1]:
                    entity_type = "place"

                elif "bk.tle%" in entity_label:
                    entity_type = "event"

                # try to label as "_Place" as 'place' entity type

                # second test if first did not find anything
                else:
                    entity_type = search_entity_type_in_faithlife_map_with_entity_label(
                        entity_label, self.entity_label_to_entity_type)

                # ner_label = [
                #     ner_label[0], ner_label[1], entity_type, ner_label[3],
                #     ner_label[2]
                # ]  #remove later
                created_ner_tuple = NERTuple(ner_tuple_from_html[0],
                                             ner_tuple_from_html[1],
                                             entity_type, entity_label,
                                             ner_tuple_from_html[2].rstrip())
                ner_tuples_in_sentence.append(created_ner_tuple)
                '''
                1) start_index
                2) end_index
                3) entity_type
                4) entity label
                5) sentence
                '''
            ner_tuples_with_entity_type.append(ner_tuples_in_sentence)

        return ner_tuples_with_entity_type

    def label_events(
        self,
        ner_tuples: List[List[NERTuple]],
        article_identifer: str,
    ) -> List[List[NERTuple]]:
        # print("label_events", ner_tuples)
        for i, ner_tuple_in_sentence in enumerate(ner_tuples):
            for j, ner_tuple in enumerate(ner_tuple_in_sentence):
                if i == 0 and j == 0 and ner_tuple.entity_name == article_identifer:  # Overwrites event, bc in this context its refering to a document
                    ner_tuples[i] = []  # only one tuple for the title
                    ner_tuple.entity_type = 'writing'
                    ner_tuples[i].append(ner_tuple)
                    # print("check this tuple:", ner_tuples[i][j])
                elif ner_tuple.entity_type == 'NONE':
                    if 'bk.tle%' in ner_tuple.entity_label:  #this is wrong -- should be event not book
                        ner_tuple.entity_type = 'event'
        return ner_tuples
