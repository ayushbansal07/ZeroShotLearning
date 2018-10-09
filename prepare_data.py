from data_parser import DataParser
import os
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse

DATA_DIR = 'data/dbaData/'
OUTPUT_DIR = "output/"
if not os.path.exists(DATA_DIR+OUTPUT_DIR):
    os.mkdir(DATA_DIR+OUTPUT_DIR)
TAGS_FILE = 'Tags.xml'
POSTS_FILE = 'Posts.xml'

#HyperParameters
MIN_TAGS_COUNT = 10
MAX_QUES = 24000
MIN_COUNT = 15

dp = DataParser()

#Prepare Tags List
dp.get_tags(DATA_DIR+TAGS_FILE,min_ct=MIN_TAGS_COUNT,target_filename=DATA_DIR+OUTPUT_DIR+"tags_json.json")

#Get list of Posts along with tags
dp.get_posts_and_tags(DATA_DIR+POSTS_FILE,target_filename=DATA_DIR+OUTPUT_DIR+"posts_json_{}k.json".format(MAX_QUES/1000),max_ques=MAX_QUES)

#Build Vocabulary
dp.build_vocab(DATA_DIR+OUTPUT_DIR+"posts_json_{}k.json".format(MAX_QUES/1000),min_count=MIN_COUNT,
                            target_filename=DATA_DIR+OUTPUT_DIR+"vocab_json_{}.json".format(MIN_COUNT))

#Create Bag of words with tfidf Transform for logreg training (X_data)
dp.bag_of_words(DATA_DIR+OUTPUT_DIR+"vocab_json_{}.json".format(MIN_COUNT),DATA_DIR+OUTPUT_DIR+"posts_json_{}k.json".format(MAX_QUES/1000),
                            target_filename=DATA_DIR+OUTPUT_DIR+"bow_filter.npy")
tfidfTransformer = TfidfTransformer()
bow = np.load(DATA_DIR+OUTPUT_DIR+"bow_filter.npy")
tfidf_data = tfidfTransformer.fit_transform(bow)
sparse.save_npz(DATA_DIR+OUTPUT_DIR+"tfifdf_transformed.npz",tfidf_data)

#Save one-hot-tags for RBM training and labels for logreg (y_data)
dp.get_tags_one_hot(DATA_DIR+OUTPUT_DIR+"tags_json.json",DATA_DIR+OUTPUT_DIR+"posts_json_{}k.json".format(MAX_QUES/1000),
                            DATA_DIR+OUTPUT_DIR+"tags_one_hot_sparse.npz",to_sparse=True)
