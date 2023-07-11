from typing import List
import re


class FaithlifeTokenizer:

    def __init__(self):
        pass

    def tokenize_article_sentences(
            self, sentences: List[List[str]]) -> List[List[str]]:
        '''Tokenizes sentences in article.

        Args:
            article: Data structure for faithlife article data.

        Returns:
        '''
        tokenized_sentences = []
        for sent_idx, sentence_list in enumerate(sentences):
            for sentence in sentence_list:
                tokenized_sentence = self.tokenize_sentence(sentence)  #
                # print(f'sentence {sent_idx}: {tokenized_sentence}')
                tokenized_sentences.append(tokenized_sentence)
        return tokenized_sentences

    def tokenize_sentence(self, sentence: str) -> List[str]:
        '''Tokenize sentence from faithlife article.

        Args:
            sentence: Sentence from faithlife article.
        
        Return
            List of tokens from sentence.
        '''

        # Find all instances of text within parentheses
        parentheses_matches = re.findall(r'\([^()]+\)', sentence)
        # Replace the text within parentheses with placeholder tokens
        for i, match in enumerate(parentheses_matches):
            sentence = sentence.replace(match, f'{{{i}}}')
        # Tokenize the sentence
        tokens = sentence.split()
        # Replace the placeholder tokens with the original text within parentheses
        for i, match in enumerate(parentheses_matches):
            tokens = [token.replace(f'{{{i}}}', match) for token in tokens]

        #deal with period at the end
        if tokens and tokens[-1][-1] == '.':
            punctuation_mark = tokens[-1][-1]
            tokens[-1] = tokens[-1][0:len(tokens[-1]) - 1]
            tokens.append(punctuation_mark)

        tokens = self.seperate_punctuation_marks(tokens)
        tokens = self.separate_parentheses(tokens)

        return tokens

    def tokenize_entity(self, entity: str) -> List[str]:
        '''Tokenize sentence from faithlife article.

        Args:
            sentence: Sentence from faithlife article.
        
        Return
            List of tokens from sentence.
        '''

        # Find all instances of text within parentheses
        parentheses_matches = re.findall(r'\([^()]+\)', entity)
        # Replace the text within parentheses with placeholder tokens
        for i, match in enumerate(parentheses_matches):
            entity = entity.replace(match, f'{{{i}}}')
        # Tokenize the sentence
        tokens = entity.split()

        # Replace the placeholder tokens with the original text within parentheses
        for i, match in enumerate(parentheses_matches):
            tokens = [token.replace(f'{{{i}}}', match) for token in tokens]

        tokens = self.seperate_punctuation_marks(tokens)
        tokens = self.separate_parentheses(tokens)
        return tokens

    def separate_parentheses(self, input_list):
        output = []
        for word in input_list:

            if word and word[0] == '(' and word[-1] == ')':
                output.append('(')
                output.append(word[1:len(word) - 1])
                output.append(')')
            elif word and word[0] == '(':
                output.append('(')
                output.append(word[1:])
            elif word and word[-1] == ')':
                output.append(word[0:len(word) - 1])
                output.append(')')
            else:
                output.append(word)
        return output

    def seperate_punctuation_marks(self, input_list):
        output = []
        punctuation_marks = {'"', "'", ',', '“', ':', ';'}
        for word in input_list:
            if word and word[0] in punctuation_marks and word[
                    -1] in punctuation_marks:
                output.append(word[0])
                output.append(word[1:len(word) - 1])
                output.append(word[-1])

            elif word and word[0] in punctuation_marks:
                output.append(word[0])
                output.append(word[1:])

            elif word and word[-1] in punctuation_marks:
                output.append(word[0:len(word) - 1])
                output.append(word[-1])

            else:
                output.append(word)
        return output
