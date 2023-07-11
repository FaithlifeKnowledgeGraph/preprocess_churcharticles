import json
import pathlib
import os

from typing import List
from markdown_preprocessing.article_structs import Article


class DYGIEEWriter:

    def __init__(
        self,
        output_dir: str = "../faithlife_model_data",
    ) -> None:
        self.output_dir = output_dir

    def write_to_jsonl(self, article: Article) -> List[dict]:
        all_docs = []
        f'Writing {article.metadata.identifier} to jsonl'
        output_json_dict = {}
        output_json_dict["doc_key"] = article.metadata.identifier
        output_json_dict["sentences"] = article.tokenized_sentences
        ner_tuples = article.ner_tuples
        unpacked_ner_tuples = [
            [tuple(nt) for nt in inner_list] for inner_list in ner_tuples
        ]

        output_json_dict["ner"] = unpacked_ner_tuples

        relation_tuples = article.relations
        unpack_relation_tuples = [
            [tuple(nt) for nt in inner_list] for inner_list in relation_tuples
        ]

        output_json_dict["relations"] = unpack_relation_tuples

        all_docs.append(output_json_dict)
        return all_docs

    def write_labeled_data_to_json(self, all_docs: List[dict],
                                   article_name: str) -> None:
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        file_output = os.path.join(self.output_dir, article_name)
        with open(f'{file_output}.jsonl', 'w') as outfile:
            for entry in all_docs:
                json.dump(entry, outfile, ensure_ascii=False)
                outfile.write('\n')
