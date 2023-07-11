from dataclasses import dataclass, field
from typing import Sequence, List


@dataclass
class JSONLFaithlifeEntity:
    '''
    {
        # document ID (please make sure doc_key can be used to identify a certain document)
        "doc_key": "CNN_ENG_20030306_083604.6",

        # sentences in the document, each sentence is a list of tokens
        "sentences": [
            [...],
            [...],
            ["tens", "of", "thousands", "of", "college", ...],
            ...
        ],

        # entities (boundaries and entity type) in each sentence
        "ner": [
            [...],
            [...],
            [[26, 26, "LOC"], [14, 14, "PER"], ...], #the boundary positions are indexed in the document level
            ...,
        ],

        # relations (two spans and relation type) in each sentence
        "relations": [
            [...],
            [...],
            [[14, 14, 10, 10, "ORG-AFF"], [14, 14, 12, 13, "ORG-AFF"], ...],
            ...
        ]
    }
    '''
    doc_key: str
    sentences: List[Sequence[str]] = field(default_factory=list)
    ner: List[Sequence[str]] = field(default_factory=list)
    relations: List[Sequence[str]] = field(default_factory=list)


@dataclass
class JSONLFaithlifeRelation:
    '''
    {
        "doc_key": "CNN_ENG_20030306_083604.6",
        "sentences": [...],
        "ner": [...],
        "relations": [...],
        "predicted_ner": [
            [...],
            [...],
            [[26, 26, "LOC"], [14, 15, "PER"], ...],
            ...
        ]
    }
    '''
    doc_key: str
    sentences: List[Sequence[str]] = field(default_factory=list)
    ner: List[Sequence[str]] = field(default_factory=list)
    relations: List[Sequence[str]] = field(default_factory=list)
    predicted_ner: List[List[str]] = field(default_factory=list)


'''
1. Count number of sentences to decide how to do train, dev, test split
2. Flatten all sentences
3. Create a map of doc_key to sentence

'''