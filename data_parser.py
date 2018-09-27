import xml.etree.ElementTree as ET
import re

class DataParser():

    def __init__(self):
        self.tags_split_pattern = re.compile('[><]')
        pass

    def get_tags(self, tags_file, min_ct = 0):
        tree = ET.parse(tags_file)
        root = tree.getroot()
        tags = {}
        for child in root:
            tags[child.attrib['TagName']] = int(child.attrib['Count'])
        filtered_tags = []
        for tag,ct in tags.items():
            if ct >= min_ct:
                filtered_tags.append(tag)

        del root
        del tree
        return filtered_tags

    def _split_tags(self,tags_string):
        t = tags_string[1:-1]
        return [x for x in self.tags_split_pattern.split(t) if x != '']

    def get_posts_and_tags(self,posts_file):
        tree = ET.parse(posts_file)
        root = tree.getroot()
        posts = []
        for child in root:
            try:
                body, tags = child.attrib['Body'], child.attrib['Tags']
                tags_list = self._split_tags(tags)
                posts.append((body,tags_list))
            except:
                pass

        del root
        del tree
        return posts
