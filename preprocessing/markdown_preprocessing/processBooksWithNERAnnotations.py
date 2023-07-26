# from loadFaithlifeData import LoadFaithlifeData
'''
For each book
1. Convert markdown to html
2. Go section by section 
3. For now, just skip each section
4. Extract all entities 


'''
import bs4
import glob
import markdown

# from article_structs import NERTuple
from dataclasses import dataclass, field
from typing import List

# PATH_TO_BOOKS = "../faithlife_data/books-with-NER-annotations-modified/*.md"


@dataclass
class NERTuple:
    '''Dataclass class for 
    named entity recognition (NER) tuple'''

    start_index: int
    end_index: int
    entity_type: str
    entity_label: str
    entity_name: str = ""

    def __iter__(self):
        yield self.start_index
        yield self.end_index
        yield self.entity_type
        yield self.entity_label
        yield self.entity_name

    def __repr__(self) -> str:
        return f'NERTuple{tuple(self)}'


@dataclass
class BookSubsection:
    book_subsection_name: str = ""
    sentences: List[List[str]] = field(default_factory=list)
    ner_tuples: List[List[NERTuple]] = field(default_factory=list)
    entity_name_to_entity_label_and_entity_type: dict[str, str] = field(
        default_factory=dict)
    tokenized_sentences: List[List[str]] = field(default_factory=list)


@dataclass
class BookSection:
    '''Dataclass class for Book Sections.'''
    book_section_name: str = ""
    book_subsections: List[BookSubsection] = field(default_factory=list)
    sentences: List[List[str]] = field(default_factory=list)
    ner_tuples: List[List[NERTuple]] = field(default_factory=list)
    entity_name_to_entity_label_and_entity_type: dict[str, str] = field(
        default_factory=dict)
    tokenized_sentences: List[List[str]] = field(default_factory=list)


@dataclass
class BookChapter:
    book_chapter_name: str = ""
    book_sections: List[BookSection] = field(default_factory=list)
    sentences: List[List[str]] = field(default_factory=list)
    ner_tuples: List[List[NERTuple]] = field(default_factory=list)
    entity_name_to_entity_label_and_entity_type: dict[str, str] = field(
        default_factory=dict)
    tokenized_sentences: List[List[str]] = field(default_factory=list)


@dataclass
class Book:
    '''Dataclass class for Books.'''
    book_name: str = ""
    book_chapters: List[BookChapter] = field(default_factory=list)
    book_section_map: dict[str, BookChapter] = field(default_factory=dict)

    sentences: List[List[str]] = field(default_factory=list)
    ner_tuples: List[List[NERTuple]] = field(default_factory=list)
    entity_name_to_entity_label_and_entity_type: dict[str, str] = field(
        default_factory=dict)
    tokenized_sentences: List[List[str]] = field(default_factory=list)


class BooksWithNERAnnotationProcesser:

    def __init__(self, path_to_books: str):
        self.path_to_books = path_to_books
        self.html_data = self.markdown_to_html()

    def markdown_to_html(self):
        html_data = []
        for book in glob.glob(self.path_to_books):
            with open(book, 'r') as f:
                html_from_markdown = markdown.markdown(f.read())
                html_data.append(html_from_markdown)
        return html_data

    def process_html(self) -> List[Book]:
        books = []
        for book in self.html_data:
            soup = bs4.BeautifulSoup(book, 'html.parser')
            new_book = Book()

            book_title = soup.find_all('h1')[0].text
            setattr(new_book, 'book_name', book_title)

            book_chapters = soup.find_all('h1')[1:]
            new_book = self.get_book_chapter_data(new_book, book_chapters)

            books.append(new_book)
        return books

    def get_book_chapter_data(
        self,
        book: Book,
        book_chapter_html: bs4.element.Tag,
    ) -> Book:
        sentence_counter = 0
        for i, h1_header in enumerate(book_chapter_html):

            bookChapter = BookChapter()
            setattr(bookChapter, 'book_chapter_name', h1_header.text)

            h1_book_chapter_section = h1_header.find_next_sibling()
            sentence_segmented = h1_book_chapter_section.text.split('\n')
            ner_tuples = self.extract_ner_tuples(h1_book_chapter_section)
            book.ner_tuples.extend([] for _ in range(len(sentence_segmented)))

            bookChapter.ner_tuples.extend(
                [] for _ in range(len(sentence_segmented)))

            prev_last_index = 0
            for sentence_index, sentence in enumerate(sentence_segmented):
                book.sentences.append([sentence])
                bookChapter.sentences.append([sentence])

                if sentence_counter > 0:
                    prev_last_index += len(
                        sentence_segmented[sentence_index - 1]) + 1

                for ner_tuple in ner_tuples:
                    start_index, end_index, word, href_link = ner_tuple

                    if word in sentence and start_index == (
                            sentence.index(word) + prev_last_index
                    ) and end_index == len(word) - 1 + sentence.index(
                            word) + prev_last_index:
                        book.ner_tuples[sentence_index] = ner_tuples
                        bookChapter.ner_tuples[sentence_index] = ner_tuples

                sentence_counter += 1

            h1_book_chapter_sibling = h1_book_chapter_section.findNextSibling()

            if h1_book_chapter_sibling == None:  #there is no subsections in this h2 heading
                continue

            elif h1_book_chapter_sibling.name == 'h2':

                all_h2_subsections = []
                orphan_h2_header = h1_book_chapter_section.findPrevious('h2')
                h2_headers = h1_book_chapter_sibling.findAllNext('h2')
                if orphan_h2_header:
                    all_h2_subsections.append(orphan_h2_header)
                all_h2_subsections.extend(h2_headers)

                bookChapter, sentence_counter = self.get_book_section_data(
                    book, prev_last_index, sentence_counter, bookChapter,
                    all_h2_subsections)
            book.book_chapters.append(bookChapter)
        return book

    def get_book_section_data(
        self,
        book: Book,
        sentence_counter: int,
        prev_last_index: int,
        bookChapter: BookChapter,
        all_h2_book_sections: List[bs4.element.Tag],
    ) -> BookChapter:
        for h2_header in all_h2_book_sections:
            bookSection = BookSection()
            bookSection.book_section_name = h2_header.text

            h2_subsection = h2_header.find_next()

            sentence_segmented = h2_subsection.text.split('\n')
            ner_tuples = self.extract_ner_tuples(h2_subsection)

            book.ner_tuples.extend([] for _ in range(len(sentence_segmented)))
            bookSection.ner_tuples.extend(
                [] for _ in range(len(sentence_segmented)))

            prev_last_index = 0
            for sentence_index, sentence in enumerate(sentence_segmented):
                book.sentences.append([sentence])
                bookSection.sentences.append([sentence])

                if sentence_counter > 0:
                    prev_last_index += len(
                        sentence_segmented[sentence_index - 1]) + 1

                for ner_tuple in ner_tuples:
                    start_index, end_index, word, href_link = ner_tuple

                    # if word in sentence:
                    #     print('word:', word)
                    #     print('sentence:', sentence)
                    #     print('true_start:', start_index)
                    #     print("our_start",
                    #           sentence.index(word) + prev_last_index - 1)
                    #     print(
                    #         'our_end:',
                    #         len(word) - 1 + sentence.index(word) +
                    #         prev_last_index - 1)
                    #     print('true_end:', end_index)

                    if word in sentence and start_index == (
                            sentence.index(word) + prev_last_index -
                            1) and end_index == len(word) - 1 + sentence.index(
                                word) + prev_last_index - 1:
                        book.ner_tuples[sentence_index] = ner_tuples
                        bookSection.ner_tuples[sentence_index] = ner_tuples

                sentence_counter += 1

            h2_book_chapter_sibling = h2_header.findNextSibling()

            if h2_book_chapter_sibling == None:  #there is no subsections in this h2 heading
                continue

            elif h2_book_chapter_sibling.name == 'h3':

                all_h3_subsections = []
                orphan_h3_header = h2_book_chapter_sibling.findPrevious('h3')
                if orphan_h3_header:
                    all_h3_subsections.append(orphan_h3_header)
                h3_headers = h2_book_chapter_sibling.findAllNext('h3')
                all_h3_subsections.extend(h3_headers)

                bookSection, sentence_counter = self.get_book_subsection_data(
                    book, prev_last_index, sentence_counter, bookSection,
                    all_h3_subsections)

            bookChapter.book_sections.append(bookSection)
        return bookChapter, sentence_counter

    def get_book_subsection_data(
        self,
        book: Book,
        prev_last_index: int,
        sentence_counter: int,
        bookSection: BookSection,
        all_h3_subsection_html: List[bs4.element.Tag],
    ) -> BookSection:
        for h3_header in all_h3_subsection_html:
            print('h3_header:', h3_header)
            bookSubsection = BookSubsection()
            bookSubsection.section_title = h3_header.text

            h3_subsection = h3_header.find_next()
            # print('h3_subsection:', h3_subsection)

            sentence_segmented = h3_subsection.text.split('\n')
            print('sentence_segmented:', sentence_segmented)
            ner_tuples = self.extract_ner_tuples(h3_subsection)
            book.ner_tuples.extend([] for _ in range(len(sentence_segmented)))
            bookSubsection.ner_tuples.extend(
                [] for _ in range(len(sentence_segmented)))

            prev_last_index = 0
            for sentence_index, sentence in enumerate(sentence_segmented):
                book.sentences.append([sentence])
                bookSubsection.sentences.append([sentence])

                if sentence_counter > 0:
                    prev_last_index += len(
                        sentence_segmented[sentence_index - 1]) + 1

                for ner_tuple in ner_tuples:
                    start_index, end_index, word, href_link = ner_tuple

                    if word in sentence:
                        print('word:', word)
                        print('sentence:', sentence)
                        print('true_start:', start_index)
                        print("our_start",
                              sentence.index(word) + prev_last_index)
                        print(
                            'our_end:',
                            len(word) - 1 + sentence.index(word) +
                            prev_last_index)
                        print('true_end:', end_index)
                    if word in sentence and start_index == (
                            sentence.index(word) + prev_last_index
                    ) and end_index == len(word) - 1 + sentence.index(
                            word) + prev_last_index:
                        book.ner_tuples[sentence_index] = ner_tuples
                        bookSubsection.ner_tuples[sentence_index] = ner_tuples

                sentence_counter += 1
            bookSection.book_subsections.append(bookSubsection)
        return bookSection, sentence_counter

    def extract_ner_tuples(
        self,
        html_element: bs4.element.NavigableString,
    ) -> List[str]:
        """Extracts NER tuples from html text with href links.

        """
        ner_tuples = []

        things_with_tuples = html_element.findAll(href=True)

        if things_with_tuples:
            for href_words in things_with_tuples:
                word, href_link = href_words.text, href_words['href'].split(
                    'https://ref.ly/logos4/Factbook?ref=')[-1]

                start_index = html_element.text.find(word)
                end_index = start_index + len(word) - 1
                ner_tuples.append((start_index, end_index, word, href_link))

        return ner_tuples
