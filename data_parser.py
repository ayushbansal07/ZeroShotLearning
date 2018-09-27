import xml.etree.ElementTree as ET
import re
import json

class DataParser():

    def __init__(self):
        self.tags_split_pattern = re.compile('[><]')
        self.ques_cleaning_pattern = re.compile("[\s\n\r\t.,:;\-_\'\"?!#&()]")
        pass

    def get_tags(self, tags_file, min_ct = 0, filename = None):
        tree = ET.parse(tags_file)
        root = tree.getroot()
        tags = {}
        for child in root:
            tags[child.attrib['TagName']] = int(child.attrib['Count'])
        filtered_tags = []
        for tag,ct in tags.items():
            if ct >= min_ct:
                filtered_tags.append(tag)

        if filename is not None:
            json_data = json.dumps(filtered_tags)
            with open(filename,'w') as f:
            	json.dump(json_data,f)

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

    def build_vocab(self,posts_file,min_count=0):
        counts = {}
        with open(posts_file) as f:
            ques_list = json.load(f)
        for ques in ques_list:
            words = [x for x in pattern.split(ques['Post']) if x != '']
            for word in words:
                if word not in counts:
                    counts[word] = 0
                counts[word] += 1
        return counts
