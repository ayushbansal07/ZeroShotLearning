import xml.etree.ElementTree as ET
import re
import json
from bs4 import BeautifulSoup
import numpy as np

class DataParser():

    def __init__(self):
        self.tags_split_pattern = re.compile('[><]')
        self.ques_cleaning_pattern = re.compile("[\s\n\r\t.,:;\-_\'\"?!#&()]")
        pass

    def get_tags(self, tags_file, min_ct = 0, target_filename = None):
        tree = ET.parse(tags_file)
        root = tree.getroot()
        tags = {}
        for child in root:
            tags[child.attrib['TagName']] = int(child.attrib['Count'])
        filtered_tags = []
        for tag,ct in tags.items():
            if ct >= min_ct:
                filtered_tags.append(tag)

        if target_filename is not None:
            with open(target_filename,'w') as f:
            	json.dump(filtered_tags,f)

            del filtered_tags
        del root
        del tree
        return filtered_tags

    def _split_tags(self,tags_string):
        t = tags_string[1:-1]
        return [x for x in self.tags_split_pattern.split(t) if x != '']

    def get_posts_and_tags(self,posts_file, target_filename = None):
        tree = ET.parse(posts_file)
        root = tree.getroot()
        #posts = []
        json_list = []
        for child in root:
            try:
                body, tags = child.attrib['Body'], child.attrib['Tags']
                tags_list = self._split_tags(tags)
                soup = BeautifulSoup(body,'html.parser')
                body = soup.text
                #posts.append((body,tags_list))
                temp = {}
                temp['Post'] = body
                temp['Tags'] = tags_list
                json_list.append(temp)
            except:
                pass

        if target_filename is not None:
        	with open(target_filename,'w') as f:
        		json.dump(json_list,f)

        del json_list
        del root
        del tree
        #return posts

    def build_vocab(self,posts_file,min_count=0,target_filename=None):
        counts = {}
        vocab = []
        with open(posts_file) as f:
            ques_list = json.load(f)
        for ques in ques_list:
            words = [x for x in self.ques_cleaning_pattern.split(ques['Post']) if x != '']
            for word in words:
                if word not in counts:
                    counts[word] = 0
                counts[word] += 1
        for key,value in counts.items():
            temp = {}
            if (value > min_count):
                temp['Word'] = key
                temp['Count'] = value
                vocab.append(temp)
        temp = {}
        temp['Word'] = '<unk>'
        temp['Count'] = 0
        vocab.append(temp)
        if target_filename is not None:
            with open(target_filename,'w') as f:
                json.dump(vocab,f)
            del counts
        else:
            return counts

    def bag_of_words(self,vocab_file,posts_file,target_filename=None):
        with open(vocab_file) as f:
            vocab = json.load(f)
            vocab_size = len(vocab)
            vocab_rev = {}
            i = 0
            for x in vocab:
                vocab_rev[x['Word']] = i
                i += 1
        with open(posts_file) as f:
            ques_list = json.load(f)

        bow = []
        for ques in ques_list:
            words = [x for x in self.ques_cleaning_pattern.split(ques['Post']) if x != '']
            temp = np.zeros(vocab_size)
            for word in words:
                if word in vocab_rev:
                    temp[vocab_rev[word]] += 1
                else:
                    temp[vocab_rev['<unk>']] += 1
            bow.append(temp)

        bow = np.array(bow)
        if target_filename is not None:
            np.save(target_filename, bow)
            del bow
        else:
            return bow
