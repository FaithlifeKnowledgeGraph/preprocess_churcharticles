import markdown
import bs4
import pathlib

from loadFaithlifeData import LoadFaithlifeData
from markdown_preprocessing.article_structs import *
from typing import Sequence


class MarkdownDataProcessor:
    '''MarkdownDataProcessors converts markdown data to dataclasses.

    Attributes:
        dataloader: Class that loads Markdown data.
        html_data: List of html strings, from converted markdown files.
    '''

    def __init__(self,
                 dataloader: LoadFaithlifeData,
                 cache_data: bool = False):
        self.dataloader = dataloader
        self.cache_data = cache_data
        self.html_data = self.markdown_to_html()

    def markdown_to_html(self) -> Sequence[str]:
        '''Converts markdown files to html strings.

        Returns:
            List of html strings, from converted markdown files
        '''
        html_data = []
        for markdown_file in self.dataloader.markdown_data_files:
            f = open(markdown_file, 'r')
            html_from_markdown = markdown.markdown(f.read())
            if self.cache_data:
                self.cache_html_data_to_output(markdown_file,
                                               html_from_markdown)
            html_data.append(html_from_markdown)
        return html_data

    def cache_html_data_to_output(self, markdown_file: str,
                                  html_data: str) -> None:
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
            new_article = self.get_data_from_h2_header(new_article, h2_header)
            articles.append(new_article)
        return articles

    def parse_metadata_from_article(
            self, soup: bs4.BeautifulSoup) -> bs4.element.Tag:
        '''Parses metadata information from markdown article.

        Args:
            soup: Beautiful Soup Module used to parse html data.
        '''
        metadata = soup.find_all('p')[0]
        return metadata

    def parse_summary_from_article(self,
                                   soup: bs4.BeautifulSoup) -> bs4.element.Tag:
        '''Parses summary information from markdown article.
        
        Args:
            soup: Beautiful Soup Module used to parse html data.
        '''
        h1_header = soup.find_all('h1')[0]
        return h1_header

    def parse_article_sections(self,
                               soup: bs4.BeautifulSoup) -> bs4.element.Tag:
        '''Parses article section information from markdown article.
        
        Args:
            soup: Beautiful Soup Module used to parse html data.
        '''
        h2_headers = soup.find_all('h2')
        return h2_headers

    def get_data_from_metadata(self, article: Article,
                               metadata: bs4.element.Tag) -> Article:
        '''Gets data from metadata tag and stores in Article Dataclass.

        Args:
            article: Article dataclass.
            metadata: Bs4 tag containing <p> element corresponding to 
            the Article metadata.
        '''
        metadata_attributes = metadata.text.split("\n")
        article.metadata = ArticleMetadata()
        for attribute in metadata_attributes:
            metadata_values = attribute.split(": ")
            metadata_key, metadata_value = metadata_values[0].lower(
            ), metadata_values[1:]

            if metadata_key == 'level':
                article.metadata.level = metadata_value[0]
            elif metadata_key == 'status':
                article.metadata.status = metadata_value[0]
            elif metadata_key == 'identifier':
                article.metadata.identifier = metadata_value[0]
            elif metadata_key == 'parents':
                article.metadata.parent = metadata_value[0]
            elif metadata_key == 'eras':
                article.metadata.eras = metadata_value[:]

        return article

    def get_data_from_h1_header(self, article: Article,
                                h1_header: bs4.element.Tag) -> Article:
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
                #todo - regex replace !!smallcaps|ad!! -> AD
                article.summary.period = in_list_value
            elif in_list_key == 'description':
                article.summary.description = in_list_value
        return article

    def get_data_from_h2_header(self, article: Article,
                                h2_headers_html: bs4.element.Tag) -> Article:
        '''Gets data from section tag and stores in Article Dataclass.
        
        Args:
            article: Article dataclass.
            metadata: Bs4 tag containing <h2> element corresponding to 
            the article's sections.
        '''
        sentence_counter = 0
        for i, h2_header in enumerate(h2_headers_html):
            if i == 0 and h2_header.text == 'Summary':  #This is the summary section
                header_name = h2_header.text
                # articleSection = ArticleSection()
                # articleSection.section_title = header_name
                summary_text = h2_header.findAllNext('p')
                article.summary.text = summary_text[0].text
                continue

            header_name = h2_header.text

            articleSection = ArticleSection()
            articleSection.section_title = header_name

            h2_article_section = h2_header.find_next('ul')

            all_lists = h2_article_section.findAll('li')
            for bullet_point_index, h2_bullet_point in enumerate(all_lists):
                article.ner_tuples.append([])

                articleSection.ner_tuples[bullet_point_index] = []
                for href_words in h2_bullet_point.findAll(href=True):
                    word, href_link = href_words.text, href_words[
                        'href'].split(
                            'https://ref.ly/logos4/Factbook?ref=')[-1]

                    start_index = h2_bullet_point.text.find(word)
                    end_index = start_index + len(word) - 1

                    article.ner_tuples[sentence_counter].append(
                        [start_index, end_index, word, href_link])
                    articleSection.ner_tuples[bullet_point_index].append(
                        (start_index, end_index, word, href_link))
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

                for h3_header in all_h3_subsections:
                    articleSubsection = ArticleSubsection()

                    articleSubsection.section_title = h3_header.text
                    if 'href' in h3_header:
                        h3_header['href'].split(
                            'https://ref.ly/logos4/Factbook?ref=')[-1]
                    # article.ner_tuples.append([])

                    # h3_value = h3_header.findNext('a')
                    # articleSubsection.section_title = (
                    #     h3_value.text, h3_value['href'].split(
                    #         'https://ref.ly/logos4/Factbook?ref=')[-1])
                    subsection = h3_header.find_next('ul')

                    articleSubsection.ner_tuples = {}
                    for bullet_point_index, h3_bullet_point in enumerate(
                            subsection.findAll('li')):
                        article.ner_tuples.append([])

                        articleSubsection.ner_tuples[bullet_point_index] = []
                        for href_words in h3_bullet_point.findAll(href=True):

                            word, href_link = href_words.text, href_words[
                                'href'].split(
                                    'https://ref.ly/logos4/Factbook?ref=')[-1]

                            start_index = h3_bullet_point.text.find(word)
                            end_index = start_index + len(word) - 1

                            article.ner_tuples[sentence_counter].append(
                                [start_index, end_index, word, href_link])
                            articleSubsection.ner_tuples[
                                bullet_point_index].append(
                                    (start_index, end_index, word, href_link))
                        articleSubsection.bullet_points.append(
                            h3_bullet_point.text)
                        article.sentences.append([h3_bullet_point.text])

                        sentence_counter += 1
                    articleSection.article_subsections.append(
                        articleSubsection)
            article.article_sections.append(articleSection)
        return article
