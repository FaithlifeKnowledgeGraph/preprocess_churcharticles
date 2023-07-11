from loadChurchMarkdownData import LoadChurchData


class BooksWithNERAnnotationProcesser:

    def __init__(self, dataloader: LoadChurchData):
        self.data_loader = dataloader

    def read_data(self):
        for path_to_book in self.data_loader.books_ner_annotation:
            with open(path_to_book, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    print(line)
            break
