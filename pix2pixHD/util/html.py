import dominate
from dominate.tags import *
import os


class HTML:
    """
    A class for generating HTML documents with images and tables.

    Args:
        web_dir (str): The directory where the HTML files will be saved.
        title (str): The title of the HTML document.
        refresh (int, optional): The refresh rate of the HTML document in seconds. Defaults to 0.

    Attributes:
        title (str): The title of the HTML document.
        web_dir (str): The directory where the HTML files will be saved.
        img_dir (str): The directory where the images will be saved.
        doc (dominate.document): The HTML document object.
        t (dominate.tags.table): The HTML table object.

    Methods:
        get_image_dir(): Returns the image directory.
        add_header(str): Adds a header to the HTML document.
        add_table(border=1): Adds a table to the HTML document.
        add_images(ims, txts, links, width=512): Adds images with captions and links to the HTML document.
        save(): Saves the HTML document to a file.
    """

    def __init__(self, web_dir, title, refresh=0):
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv='refresh', content=str(refresh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style='table-layout: fixed;')
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=512):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style='word-wrap: break-word;', halign='center', valign='top'):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style='width:%dpx' % (width), src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims = []
    txts = []
    links = []
    for n in range(4):
        ims.append('image_%d.jpg' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.jpg' % n)
    html.add_images(ims, txts, links)
    html.save()
