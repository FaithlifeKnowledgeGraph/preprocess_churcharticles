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

PATH_TO_BOOKS = "../faithlife_data/books-with-NER-annotations-modified/*.md"


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
class BookSection:
    '''Dataclass class for Book Sections.'''
    book_section_name: str = ""
    sentences: List[List[str]] = field(default_factory=list)
    ner_tuples: List[List[NERTuple]] = field(default_factory=list)
    entity_name_to_entity_label_and_entity_type: dict[str, str] = field(
        default_factory=dict)
    tokenized_sentences: List[List[str]] = field(default_factory=list)


@dataclass
class Book:
    '''Dataclass class for Books.'''
    book_name: str = ""
    book_sections: List[BookSection] = field(default_factory=list)
    book_section_map: dict[str, BookSection] = field(default_factory=dict)

    sentences: List[List[str]] = field(default_factory=list)
    ner_tuples: List[List[NERTuple]] = field(default_factory=list)
    entity_name_to_entity_label_and_entity_type: dict[str, str] = field(
        default_factory=dict)
    tokenized_sentences: List[List[str]] = field(default_factory=list)


class BooksWithNERAnnotationProcesser:

    def __init__(self):
        self.html_data = self.markdown_to_html()
        print(len(self.html_data))

    def markdown_to_html(self):
        html_data = []
        for book in glob.glob(PATH_TO_BOOKS):
            f = open(book, 'r')
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

            books.append(new_book)
        print(books)
        return books

    def parse_book_sections(self, soup: bs4.BeautifulSoup) -> bs4.element.Tag:
        book_sections = soup.find_all('h2')
        return book_sections

    def get_data_from_section(self, section: bs4.element.Tag) -> BookSection:
        book_section = BookSection(section.text)
        return book_section


b = BooksWithNERAnnotationProcesser()
b.process_html()