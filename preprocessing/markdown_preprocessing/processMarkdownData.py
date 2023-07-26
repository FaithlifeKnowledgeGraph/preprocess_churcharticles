import markdown
import bs4
import pathlib

from loadFaithlifeData import LoadFaithlifeData
from markdown_preprocessing.article_structs import *
from typing import Sequence


class MarkdownDataProcessor:
    '''MarkdownDataProcessors converts church history articles from 
    markdown data to article struct.

    Attributes:
        dataloader: Class that loads Markdown data.
        cache_data: Boolean to cache html data to output directory.
        html_data: List of html strings, from converted markdown files.
    '''

    def __init__(
        self,
        dataloader: LoadFaithlifeData,
        cache_data: bool = False,
    ):
        self.dataloader = dataloader
        self.cache_data = cache_data
        self.html_data = self.markdown_to_html()

    def markdown_to_html(self) -> Sequence[str]:
        '''Converts markdown files to list of html strings.

        Returns:
            List of html strings, from converted markdown files
        '''
        html_data = []
        for markdown_file in self.dataloader.markdown_data_files:
            with open(markdown_file, 'r') as f:
                html_from_markdown = markdown.markdown(f.read())
            if self.cache_data:
                self.cache_html_data_to_output(markdown_file,
                                               html_from_markdown)
            html_data.append(html_from_markdown)
        return html_data

    def cache_html_data_to_output(
        self,
        markdown_file: str,
        html_data: str,
    ) -> None:
        article_name = markdown_file.split('/')[-1].split('.')[0]
        pathlib.Path('markdown_preprocessing/cache_markdown_data').mkdir(
            parents=True, exist_ok=True)
        with open(
                f'markdown_preprocessing/cache_markdown_data/{article_name}.txt',
                'w') as f:
            f.write(html_data)

    def process_html(self) -> List[Article]:
        '''Processes list of html strings and converts to dataclasses.
        
        Returns:
            List of Article Dataclasses
        '''
        articles = []
        for html in self.html_data:
            soup = bs4.BeautifulSoup(html, "html.parser")
            new_article = Article()

            article_title = soup.find_all('h1')[0].text
            new_article.title = article_title

            metadata = self.parse_metadata_from_article(soup)
            new_article = self.get_data_from_metadata(new_article, metadata)

            h1_header = self.parse_summary_from_article(soup)
            new_article = self.get_data_from_h1_header(new_article, h1_header)

            h2_header = self.parse_article_sections(soup)
            new_article = self.get_data_from_h2_headers(new_article, h2_header)
            articles.append(new_article)
        return articles

    def parse_metadata_from_article(
        self,
        soup: bs4.BeautifulSoup,
    ) -> bs4.element.Tag:
        '''Parses metadata information from markdown article.

        Args:
            soup: Beautiful Soup Module used to parse html data.
        '''
        metadata = soup.find_all('p')[0]
        return metadata

    def parse_summary_from_article(
        self,
        soup: bs4.BeautifulSoup,
    ) -> bs4.element.Tag:
        '''Parses summary information from markdown article.
        
        Args:
            soup: Beautiful Soup Module used to parse html data.
        '''
        h1_header = soup.find_all('h1')[0]
        return h1_header

    def parse_article_sections(
        self,
        soup: bs4.BeautifulSoup,
    ) -> bs4.element.Tag:
        '''Parses article section information from markdown article.
        
        Args:
            soup: Beautiful Soup Module used to parse html data.
        '''
        h2_headers = soup.find_all('h2')
        return h2_headers

    def get_data_from_metadata(
        self,
        article: Article,
        metadata: bs4.element.Tag,
    ) -> Article:
        '''Gets data from metadata tag and stores in Article Dataclass.

        Args:
            article: Article dataclass.
            metadata: Bs4 tag containing <p> element corresponding to 
            the Article metadata.
        '''
        METADATA_ATTRIBUTES = [
            'level', 'status', 'identifier', 'parents', 'eras'
        ]
        metadata_attributes = metadata.text.split("\n")

        article.metadata = ArticleMetadata()
        for attribute in metadata_attributes:
            metadata_values = attribute.split(": ")
            metadata_key, metadata_value = metadata_values[0].lower(
            ), metadata_values[1:]

            if metadata_key in METADATA_ATTRIBUTES:
                setattr(article.metadata, metadata_key, metadata_value[0])

        return article

    def get_data_from_h1_header(
        self,
        article: Article,
        h1_header: bs4.element.Tag,
    ) -> Article:
        '''Gets data from summary tag and stores in Article Dataclass.
        
        Args:
            article: Article dataclass.
            metadata: Bs4 tag containing <h1> element corresponding to 
            the article's summary section.
        '''
        article_header = h1_header.text

        article_html_list = h1_header.find_next()
        article_header, article_html_list
        article.summary = ArticleSummary()

        all_lists = article_html_list.findAll('li')
        for in_list in all_lists:
            in_list_values = in_list.text.split(": ")
            in_list_key, in_list_value = in_list_values[0].lower(
            ), in_list_values[1]
            if in_list_key == 'period':
                #TODO - regex replace !!smallcaps|ad!! -> AD
                setattr(article.summary, in_list_key, in_list_value)
            elif in_list_key == 'description':
                setattr(article.summary, in_list_key, in_list_value)
        return article

    def get_data_from_h2_headers(
        self,
        article: Article,
        h2_headers_html: bs4.element.Tag,
    ) -> Article:
        '''Gets data from section tag and stores in Article Dataclass.

        This is specifically sections with ##headers. Each h2 header may
        contain 0 or more subsections that must be embedded within a section

        For example, a key people h2 header may contain subsections of 'fundamenalist',
        'neo-evanelicals', etc. 

        The exception is the summary section which will always be the first
        h2 header if it exists in the article.

        This function loops through all h2 headers to the end and returns the article struct
        
        Args:
            article: Article dataclass.
            metadata: Bs4 tag containing <h2> element corresponding to 
            the article's sections.
        
        Returns:
            Article dataclass with sections and subsections.
        '''
        sentence_counter = 0
        for i, h2_header in enumerate(h2_headers_html):
            if h2_header.text == 'Summary':
                article.summary.text = h2_header.findAllNext('p')[0].text
                continue

            articleSection = ArticleSection()
            articleSection.section_title = h2_header.text

            h2_article_section = h2_header.find_next('ul')

            all_lists = h2_article_section.findAll('li')
            for bullet_point_index, h2_bullet_point in enumerate(all_lists):
                ner_tuples = self.extract_ner_tuples(h2_bullet_point)
                article.ner_tuples.append(ner_tuples)
                articleSection.ner_tuples[bullet_point_index] = ner_tuples
                articleSection.bullet_points.append(h2_bullet_point.text)
                article.sentences.append([h2_bullet_point.text])

                sentence_counter += 1

            h2_sibling = h2_header.findNextSibling()
            if h2_sibling == None:  #there is no subsections in this h2 heading
                continue

            elif h2_sibling.name == 'h3':
                all_h3_subsections = []
                orphan_h3_header = h2_article_section.findPrevious('h3')
                h3_headers = h2_article_section.findAllNext('h3')
                all_h3_subsections.append(orphan_h3_header)
                all_h3_subsections.extend(h3_headers)

                articleSection, sentence_counter = self.get_data_from_h3_headers(
                    article, sentence_counter, all_h3_subsections,
                    articleSection)
            article.article_sections.append(articleSection)
        return article

    def get_data_from_h3_headers(
        self,
        article: Article,
        sentence_counter: int,
        all_h3_subsections: bs4.element.Tag,
        articleSection: ArticleSection,
    ) -> ArticleSection:
        '''Gets data from h3 headers

        This function is called when a h2 header has subsections. So the 
        ArticleSection dataclass instatinated is passed here and this method
        will get all subsections to embed within the article section dataclass.

        Args:
            Article: Article dataclass.
            sentence_counter: Counter for current sentence.
            all_h3_subsections: List of all h3 subsections.
            articleSection: ArticleSection dataclass.
        
        Returns:
            ArticleSection dataclass with subsections.
        '''

        for h3_header in all_h3_subsections:
            articleSubsection = ArticleSubsection()
            articleSubsection.section_title = h3_header.text

            subsection = h3_header.find_next('ul')

            for bullet_point_index, h3_bullet_point in enumerate(
                    subsection.findAll('li')):
                ner_tuples = self.extract_ner_tuples(h3_bullet_point)
                article.ner_tuples.append(ner_tuples)
                articleSubsection.ner_tuples[bullet_point_index] = ner_tuples
                articleSubsection.bullet_points.append(h3_bullet_point.text)
                article.sentences.append([h3_bullet_point.text])

                sentence_counter += 1
            articleSection.article_subsections.append(articleSubsection)
        return articleSection, sentence_counter

    def extract_ner_tuples(
        self,
        html_element: bs4.element.NavigableString,
    ) -> List[str]:
        """Extracts NER tuples from html text with href links.

        In the markdown files, there are labels with [entity](entity_label)
        sometimes the entity label is in the form a href link. This function
        searches for these links and extracts the entity name and the entity

        Example 1: [Protestant liberalism](bk.%ProtestantLiberalism)
        Example 2: [Germany](https://ref.ly/logos4/Factbook?ref=bk.%40Germany)

        This function extract the entity name and the entity label. It also
        extracts the start and end index of the entity name in the text.

        This is the suiteable format input for our model.

        Args:
            text: html text with potential entity_labels
        
        Returns:
            List of entity label in the form of a tuple 
        """
        ner_tuples = []
        for href_words in html_element.findAll(href=True):
            word, href_link = href_words.text, href_words['href'].split(
                'https://ref.ly/logos4/Factbook?ref=')[-1]

            start_index = html_element.text.find(word)
            end_index = start_index + len(word) - 1
            ner_tuples.append((start_index, end_index, word, href_link))
        return ner_tuples
