__author__ = 'dflatow'

import logging
import codecs
import random
import os
import tool
import numpy as np


import pymongo
import time

from sklearn.feature_extraction.text import CountVectorizer

from data_piece import DataPiece



logger = logging.getLogger(__name__)

SEPERATOR = "\t"
NUM_OF_FIELDS = 7

def encode_list_ascii(data):
    return [x.encode(encoding='ascii', errors='ignore') for x in data]

def encode_list_uft8(data):
    return [x.encode(encoding='utf-8') for x in data]

def rem_tab_nl_cr(text):
    return text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

def process_data(data, data_source=None):
    # if it's a tweet, process as tweet. otherwise photo
    # write to CSV each line.

    if data_source in  ['Twitter', 'Foursquare']:
        #if 'coordinates' not in data or data['coordinates'] is None:
        #    return None
        if data['user']['location'] is None:
            location = 'None'
        else:
            location = data['user']['location'].replace('\n', ' '
            ).replace('\t', ' ').replace('\r', ' ')

        text = data['text'].strip().replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        

        if (len(text) == 0) and (location == 'None'):
            return None
        elif len(text) == 0:
            text = 'None'


        t_list = [text,
                  str(data['coordinates']['coordinates'][0]),
                  str(data['coordinates']['coordinates'][1]),
                  location,
                  str(data['user']['id']).replace('\n', ' ').replace('\t', ' ').replace('\r', ' '),
                  str(data['created_time']),
                  data_source]


    elif data_source == 'Instagram':

        if 'name' not in data['location']:
            location = 'None'
        else:
            location = data['location']['name'].replace('\n', ' ').replace('\t', ' ').replace('\r', ' '),

        text = data['text'].strip().replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')

        if (len(text) == 0) and (location == 'None'):
            return None
        elif len(text) == 0:
            text = 'None'


        t_list = [text,
                  str(data['location']['longitude']),
                  str(data['location']['latitude']),
                  location,
                  str(data['user']['id']).replace('\n', ' ').replace('\t', ' ').replace('\r', ' '),
                  str(data['created_time']),
                  data_source]

    else:
        raise Exception('data_source not known')

    return encode_list_ascii(t_list)
    
def determine_source(data_point):
    
    if 'caption' in data_point.keys():
        source = 'instagram_instagram'
        
    elif 'source' in data_point.keys():
        
        source_desc = data_point['source']
        
        if 'http://twitter.com/download/iphone' in source_desc:
            source = 'twitter_iphone'
        elif 'foursquare.com' in source_desc:
            source = 'twitter_foursquare'
        elif 'http://instagram.com' in source_desc:
            source = 'twitter_instagram'
        elif 'http://twitter.com/download/android' in source_desc:
            source = 'twitter_android'
        else:
            source = "twitter_other"
    else:
        source = 'unknown'

    return source

def download_file(collection='tweets', file_name=None, database='citybeat_production', data_type='tweet', limit=10000,
                  data_dir="./Data/"):
    mongodb_address = 'ec2-23-22-67-45.compute-1.amazonaws.com'

    mongodb_port = 27017
    mongodb_user = 'citybeat'
    mongodb_password = 'production'

    connection = pymongo.Connection(mongodb_address, mongodb_port)
    connection['admin'].authenticate(mongodb_user, mongodb_password)

    data_source = connection[database][collection]

    c = 1
    with open(data_dir + file_name, 'w+') as file:
        for n, _data_point in enumerate(data_source.find()):
            if n%1000 == 0:
                print '%d downloaded'%(n)

            data_source = determine_source(_data_point)
            t_list = process_data(_data_point, data_source=data_source)

            if t_list is None:
                continue

            if len(t_list)==NUM_OF_FIELDS:

                if (limit is None) or (c <= limit):
                    file.write(SEPERATOR.join(t_list) + '\n')
                    c += 1
                else:
                    break

def read_data(limit=1000, data_file="./Data/citybeat_data.csv"):

    data_list = codecs.open(data_file, 'r', 'ascii').readlines()

    res = []
    for n, line in enumerate(data_list):
        if (limit is not None) and (n >= limit):
            break

        dat = line.split(SEPERATOR)

        if len(dat)==NUM_OF_FIELDS:
            tweet = DataPiece(*dat)
            res.append(tweet)
        else:
            print dat
            raise Exception('missing fields')

    return res

# CHECK OVER THIS FUNCTION / UPDATE TO USE READ_DATA
def load_data(dataset="cmu", training_ratio = 0.8, do_shuffle=True, 
              limit_for_testing = -1, limit=4000000, with_foursquare=True, 
              train_data_type='tweets', test_data_type='tweets', filter_test_4sq=False):

    if dataset=='cmu':
        res = read_cmu_data_file()

    elif dataset=='nyc':
        if train_data_type==test_data_type:
            res = read_nyc_data_file(limit=limit, with_foursquare=with_foursquare, data_type=train_data_type)

        else:
            train_limit = limit * training_ratio
            test_limit = limit * (1 - training_ratio)

            train = read_nyc_data_file(limit=limit, with_foursquare=with_foursquare, data_type=train_data_type)
            train, _ = _split_into_train_and_test(train, training_ratio, filter_test_4sq, do_shuffle)

            test = read_nyc_data_file(limit=limit, with_foursquare=with_foursquare, data_type=test_data_type)
            _, test = _split_into_train_and_test(test, training_ratio, filter_test_4sq, do_shuffle)

            return train, test

        return _split_into_train_and_test(res, training_ratio, filter_test_4sq, do_shuffle)


def simple_tokenizer(text):
    words = text.split()
    return words

def read_nyc_data_file(limit=4000000, data_type='tweets', with_foursquare=True):
    data_dir = "/Users/daflatow/Dropbox/Python/CrystalBall/Data/"
    unzip_if_not_exist(data_dir + 'data.csv')


    
    assert data_type in ['tweets', 'photos', 'all'], 'wrong data type'

    if data_type == 'tweets':
        data_list = codecs.open(data_dir + 'data.csv', 'r', 'utf-8').readlines()
    elif data_type == 'photos':
        data_list = codecs.open(data_dir + 'data_photos.csv', 'r', 'utf-8').readlines()
    elif data_type == 'all':
        # read both tweets and instagram photos
        data_list = codecs.open(data_dir + 'data.csv', 'r', 'utf-8').readlines() + codecs.open('data_photos.csv', 'r', 'utf-8').readlines()


    res = []
    for n, line in enumerate(data_list):
        if n >= limit:
            break

        if n%1000000 == 0:
            logger.info("Reading %d lines" %(n))
        dat = line.strip().split(SEPERATOR)

        if len(tmp)==NUM_OF_FIELDS:
            # convert lat, lon, time_created to floats
            dat[1] = float(tmp[1])
            dat[2] = float(tmp[2])

            # check 
            if not with_foursquare and (tmp[0].startswith("I'm at") or tmp[0].find("http")!=-1 ):
                continue

            tweet = DataPiece(*dat)
            # eliminate location info
            res.append(tweet)

    logger.info("Lenght of res " + str(len(res)))

    return res

def read_cmu_data_file():
    unzip_if_not_exist('./Data/full_text.txt')
    file = codecs.open('./Data/full_text.txt', 'r', 'ISO-8859-1')
    res = []
    users = {}
    for n, line in enumerate(file.readlines()):
        if n%1000 == 0:
            print 'reading %d lines'%(n)

        tmp = line.strip().split("\t")
        if len(tmp)<6:continue

        if tmp[0] in users:
            users[tmp[0]][0] += tmp[5] # connect text
        else:
            users[tmp[0]] = [tmp[5], float(tmp[3]), float(tmp[4])]

    for n, key in enumerate(users):
        tweet = DataPiece(users[key][1], users[key][2], users[key][0], "")
        res.append(tweet)
    return res

    
    
def _split_into_train_and_test(data, training_ratio, filter_test_4sq, do_shuffle):

    if not do_shuffle:
        random.shuffle(data)

    n_train = int(training_ratio*len(data))
    if filter_test_4sq:
        return data[:n_train], [r for r in data[n_train:] if r.get_text().find("http")==-1]
    else:
        return data[:n_train], data[n_train:]


def unzip_if_not_exist(file_name):
    current_file_path = os.path.realpath(__file__)
    current_file_path = os.path.abspath(os.path.join(current_file_path, os.pardir))

    print file_name
    print current_file_path
    if not os.path.isfile(os.path.join(current_file_path, file_name)):
        print 'begin decompression...'
        import gzip
        f = gzip.open('./Data/' + file_name+'.gz', 'rb')
        file_content = f.read()
        f2 = open(file_name, 'w')
        f2.write(file_content)
        f.close()
        f2.close()

def clean_spam(tweets, min_spam_thresh=30, jaccard_tresh=0.3):
    """ simple method to clean spam. For each n-gram, we need it to exist in more than N users' tweets. This
     potentially prevents flood spaming (a spammer tweet a lot of same tweets). For each n-gram we also check that
     all tweets containing that n-gram are sufficiently different (spam tweets often contain the same text across
     many tweets)."""

    # vectorizer params
    ngram_range = (1,2)
    min_df = 0.00002
    max_df = 0.1

    print 'Cleaning spaming...'
    vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df, stop_words='english')

    print 'getting data'
    tweet_text = [d.get_text() for d in tweets]

    print 'begin vectorizing'
    X = vectorizer.fit_transform(tweet_text)
    print 'X shape ', X.shape
    names = vectorizer.get_feature_names()


    spam_rows = [] # keep track of spam rows while looping over columns of vectorized tweet text

    for i in range(X.shape[1]):

        non_zero_col_i_indexes = X[:, i].nonzero()[0] # bool
        non_zero_col_i_tweets = [tweets[r] for r in non_zero_col_i_indexes]

        if i%100 == 0:
            print names[i]

        is_spam_row = tool.identify_similar_spam_tweets(non_zero_col_i_tweets, 
                                                        jaccard_tresh=jaccard_tresh)
        new_spam_rows = non_zero_col_i_indexes[is_spam_row]
        
        # only consider spam if there are more than 30 spam messages
        if len(new_spam_rows) > min_spam_thresh: 
            print 'new spam: %s' % len(new_spam_rows)
            print tweets[new_spam_rows[0]].get_text()
            print '...'
            print tweets[new_spam_rows[-1]].get_text()
            print '---------'
            # record new spam rows
            spam_rows.extend(new_spam_rows)

    spam_tweets = move_index_elements_to_new_list(tweets, spam_rows)

    return spam_tweets

def identify_spam_rows(tweets, ngram_count_vector, jaccard_thresh, min_thresh):
    "returns list of spam indexes in tweets"
    non_zero_col_i_indexes = ngram_count_vector.nonzero()[0] # bool
    non_zero_col_i_tweets = [tweets[r] for r in non_zero_col_i_indexes]

    is_spam_row = tool.identify_similar_spam_tweets(non_zero_col_i_tweets, 
                                                    jaccard_tresh=jaccard_tresh)
    new_spam_rows = non_zero_col_i_indexes[is_spam_row]

    # only consider spam if there are more than 30 spam messages
    if len(new_spam_rows) > min_spam_thresh: 
        print 'new spam: %s' % len(new_spam_rows)
        print tweets[new_spam_rows[0]].get_text()
        print '...'
        print tweets[new_spam_rows[-1]].get_text()
        print '---------'
        # record new spam rows
        spam_rows.extend(new_spam_rows)
            

def move_index_elements_to_new_list(list_to_split, indexes):

    new_list = []
    for i in sorted(np.unique(indexes), reverse=True):
        new_list.append(list_to_split.pop(i))

    return new_list

if __name__ == '__main__':
    download_file(file_name='citybeat_data.csv', limit=None)
