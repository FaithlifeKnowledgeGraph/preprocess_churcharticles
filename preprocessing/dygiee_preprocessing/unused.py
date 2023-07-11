def find_and_add_entities_from_history_article_labels_to_article_struct(
    self,
    ner_tuples: List[List[NERTuple]],
    sentences: List[str],
    all_entities_in_history_article_csv: set[str],
) -> List[List[Tuple]]:

    for i in range(len(sentences)):
        # search database for more labels
        # print(
        #     f'\n\nsearching entities for sentence: {article.sentences[i]}')
        # print(f'Existing entities: {article.ner_tuples}')
        db_entities = self.search_entity_in_faithlife_map(
            sentences[i], ner_tuples[i], self.database_df,
            all_entities_in_history_article_csv)

        ner_tuples[i].extend(db_entities)

        if ner_tuples[i]:
            print("curr_tuple:", ner_tuples[i])

            #previous statement appends new elements to end of list, must sort
            ner_tuples[i].sort(key=lambda ner_tuple:
                               (ner_tuple.start_index, ner_tuple.end_index))
    return ner_tuples


def search_entity_in_faithlife_map(
    self,
    sentence: List[str],
    labeled_ner_tuples: List[NERTuple],
    database_df: pd.DataFrame,
    all_entities_in_csv: Set[str],
) -> List[NERTuple]:
    '''
    TODO: Convert lookup into a map later

    Need to clean up labeled_ner_tuples for comparison to work properly
    
    '''
    print("sentence:", sentence)
    print("labeled_ner_tuples:", labeled_ner_tuples)
    sentence = sentence[0].lower()  #fix
    print('sentence: ', sentence)
    print('labeled_ner_tuples: ', labeled_ner_tuples)
    # print('all_entities_in_csv: ', all_entities_in_csv)

    already_labeled_entity_labels = set()
    for ner_tuple in labeled_ner_tuples:
        cleaned_label = replace_special_chars_in_entity_annotations(
            ner_tuple.entity_label)
        already_labeled_entity_labels.add(cleaned_label)

    # print('already_labeled_entity_labels:', already_labeled_entity_labels)
    entities = []
    # Can't locate person because it add labels like "Denis Diderot (French Encyclopaedist)"
    # for bk.%DenisDiderot_Person
    for entity_label in all_entities_in_csv:
        if entity_label not in already_labeled_entity_labels:
            entity_name = database_df.loc[database_df['primary'] ==
                                          entity_label]['label_en'].to_string(
                                              index=False)
            entity_name_lower = entity_name.lower()
            if '(' in entity_name:
                entity_name = self.extract_name(entity_name)

            entity_type = database_df.loc[database_df['primary'] ==
                                          entity_label]['kind'].to_string(
                                              index=False)
            # print('entity_name:', entity_name)
            # print('entity_label: ', entity_label)
            # print('entity_type: ', entity_type)

            start_index = sentence.find(entity_name_lower)
            if start_index != -1:
                end_index = start_index + len(entity_name_lower)
                entities.append(
                    NERTuple(start_index, end_index, entity_type, entity_label,
                             entity_name))
                # entities.append([
                #     start_index, end_index, entity_type, entity_label,
                #     entity_name
                # ])

            #     print(
            #         f"did find: entity_name: {entity_name} \n sentence:{sentence}"
            #     )
            # else:
            #     print(
            #         f'did not find: entity_name: {entity_name} \n sentence:{sentence}'
            #     )
    print('found_entities:', entities)
    return entities
