import pandas as pd
import re

from typing import Tuple, Mapping


def load_faithlife_database_to_single_df(
        path_to_faithlife_db_folder: str) -> pd.DataFrame:
    '''Loads faithlife concept database to a single panda dataframe.

    Args:
        path_to_faithlife_db_folder: Directory of faithlife database csv files.

    Returns:
        Pandas dataframe that contains faithlife concept database.
    '''

    file_names = [
        'agent', 'bio', 'era', 'event', 'lco', 'lcv', 'lsto', 'place', 'thing',
        'tle'
    ]
    column_names = ['kind', 'primary', 'label_en', 'rank']

    database_df = pd.concat([
        pd.read_csv(f'{path_to_faithlife_db_folder}/{file}.csv',
                    names=column_names,
                    header=0,
                    index_col=False) for file in file_names
    ])
    return database_df


def search_entity_type_in_faithlife_map_with_entity_label(
    entity_label: str,
    faithlife_map: dict[str, str],
) -> str:
    '''Searches for entity type in faithlife database.

    Args:
        ner_label_from_html: 
        Ex: [0, 24, 'The Renaissance in Europe', 'bk.tle%3arenaissanceeurope']
        faithlife_map: 

    Returns:
        The entity type found in faithlife map or 'NONE' if not found.
    '''

    if entity_label in faithlife_map:
        return faithlife_map[entity_label]
    return "NONE"


def extract_name(input_string):
    # Find the index of the opening parenthesis
    start_index = input_string.find('(')
    if start_index != -1:
        # If there is an opening parenthesis, remove it and everything after it
        return input_string[:start_index - 1]
    else:
        # If there are no parentheses, return the entire string
        return input_string


def make_faithlife_map(faithlife_df: pd.DataFrame) -> Mapping[str, str]:
    '''Convert faithlife dataframe into a hashmap.

    Args:
        faithlife_df: Dataframe containing faithlife concept 
        database data.
    
    Returns:
        A hashmap of entity label to the entity type
        Example:
        { 
            bk.#24Elders:  person
            lsto.AbrahamicCovenant: concept
        }
    '''
    faithlife_map = {}
    for entity_label, entity_type in zip(faithlife_df['primary'],
                                         faithlife_df['kind']):
        faithlife_map[entity_label] = entity_type
    return faithlife_map


def make_faithlife_map_2(
        faithlife_df: pd.DataFrame) -> dict[str, Tuple[str, str]]:
    faithlife_map = {}

    # Remove rows in faithlife_df where the entity_type is 'thing', 'artifact', 'denom', or 'denomgroup'
    faithlife_df = faithlife_df[faithlife_df['kind'].isin(
        {'person', 'place', 'writing', 'concept'})]

    # Remove entry where kind is a person named 'Long', ex: person,bio.long_1,Long,199945
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'Long']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'Pure']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'Reason']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'New']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'News']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'Man']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'Understanding']
    # faithlife_df = faithlife_df[faithlife_df['label_en'] != 'Grace']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'First']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'Age']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'By']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'Also']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'See']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'Answer']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'His']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'Third']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'General']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'Good']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'Action']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'Middle']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'A']
    faithlife_df = faithlife_df[faithlife_df['label_en'] != 'On']

    # faithlife_df = faithlife_df[faithlife_df['label_en'] != 'Hunger']

    for _, df in faithlife_df[['label_en', 'primary', 'kind']].iterrows():
        entity_name = df['label_en']

        if entity_name != entity_name:
            continue
        if '(' in entity_name:

            entity_name = extract_name(entity_name)
        entity_label = df['primary']
        entity_type = df['kind']
        faithlife_map[entity_name] = (entity_label, entity_type)
    return faithlife_map


def parse_chicago_citation(citation):
    """
    Parses a Chicago style citation in the format "Article Title" in Book Title (Book Author).

    Parameters:
    citation (str): The Chicago style citation string to be parsed.

    Returns:
    dict: A dictionary containing the extracted information, with keys for the article title, book title,
    and book author.
    """
    pattern = r'"(.*)" in (.*) \((.*)\)'
    match = re.match(pattern, citation)
    if not match:
        raise ValueError('Invalid citation format')
    article_title, book_title, book_author = match.groups()
    return {
        'article_title': article_title.strip(),
        'book_title': book_title.strip(),
        'book_author': book_author.strip()
    }