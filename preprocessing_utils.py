import re

from typing import List


def replace_special_chars_in_entity_annotations(entity_label: str):
    '''Replace special characters in entity annotations in Article.

    Args:
        entity_label: Entity label from html from Article
    Return
        Entity labeled with annotaitons removed
    
    ex: 
    
    bk.tle%3arenaissanceeurope --> bk.tlearenaissanceeurope
    '''
    entity_label = entity_label.replace("%25", "%")
    entity_label = entity_label.replace("%23", "%")
    entity_label = entity_label.replace("%40", "@")  #for places
    entity_label = entity_label.replace("%3", "%")  #for tle
    return entity_label
