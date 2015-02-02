import pymongo
import codecs
from data_piece import DataPiece
from numpy.random import choice, seed
import math
from sklearn.feature_extraction.text import CountVectorizer
import cPickle as pickle
from argparse import ArgumentParser
import os
import numpy as np

def process_data_point(data_point):
    
    source = determine_source(data_point)

    if source.startswith('twitter'):
        
        if data_point['user']['location'] is None:
            location = u''
        else:
            location = rem_special_chars(data_point['user']['location'])

        text = rem_special_chars(data_point['text'].strip())

        if (len(text) == 0) and (location == u''):
            return None
        elif len(text) == 0:
            text = u''

        t_list = [text,
                  unicode(data_point['coordinates']['coordinates'][1]), #lat
                  unicode(data_point['coordinates']['coordinates'][0]), #lon
                  location, # user specified location
                  unicode(data_point['user']['id']),
                  data_point['created_time'],
                  source]


    elif source.startswith('instagram'):
        
        try:
            try:
                if (data_point is not None) and ('name' in data_point['location']):
                    location = rem_special_chars(data_point['location']['name'])
                else:
                    location = u''
            except:
                location = u''
            try:
                if data_point['caption'] is None:
                    text = u''
                else:
                    text = rem_special_chars(data_point['caption']['text'])
            except:
                text = u''

            if (len(text) == 0) and (location == u''):
                return None
            elif len(text) == 0:
                text = u''

            t_list = [text,
                      unicode(data_point['location']['latitude']),
                      unicode(data_point['location']['longitude']),
                      location,
                      unicode(data_point['user']['id']),  
                      data_point['created_time'],
                      source]
        except:
            return None
            
    else:
        raise Exception('source not known')
    
    return t_list
        
def rem_special_chars(text):
    
    special_chars = [u'\n', u'\r', u'\t', u'"', u'\x85']
    for char in special_chars:
        text = text.replace(char, u' ')
        
    return text

def determine_source(data_point):
    
    if 'caption' in data_point.keys():
        source = u'instagram_instagram'
        
    elif 'source' in data_point.keys():
        
        source_desc = data_point['source']
        
        if 'http://twitter.com/download/iphone' in source_desc:
            source = u'twitter_iphone'
        elif 'foursquare.com' in source_desc:
            source = u'twitter_foursquare'
        elif 'http://instagram.com' in source_desc:
            source = u'twitter_instagram'
        elif 'http://twitter.com/download/android' in source_desc:
            source = u'twitter_android'
        else:
            source = u'twitter_other'
    else:
        source = u'unknown'

    return source

def get_vectorizer(data, min_user, ngram_range, min_df, max_df):
    """ simple method to clean spams. For each n-gram, we need it to exist in more than N users' tweets. This
     potentially prevents flood spaming (a spammer tweet a lot of same tweets)"""
    
    vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df, 
                                 max_df=max_df, stop_words='english')
    
    vocabulary = []
    docs = [d.get_info() for d in data]
    X = vectorizer.fit_transform(docs)
    names = vectorizer.get_feature_names()

    # build vocab using ngrams shared by more than min_users
    for i in range(X.shape[1]):
        row_indexes = X[:, i].nonzero()[0]
        tmp_set = set()
        for r in row_indexes:
            tmp_set.add(data[r].get_user_id())
        if len(tmp_set) >= min_user:
            vocabulary.append(names[i])

    return CountVectorizer(ngram_range=ngram_range, vocabulary=vocabulary)

def get_data_and_write_to_file(file_object, source, limit):
    
    for data_point in get_mongo_cursor(source, limit): 
        data = process_data_point(data_point)
        
        if data is not None:
            file_object.write(u'\t'.join(data) + u'\n')
            
def data_list_to_line(data_list):
    return u'\t'.join(data_list) + u'\n'

def data_piece_to_line(piece):
    return data_list_to_line(list(piece.get_data()))

def write_data_pieces_to_file(fname, data):
    
    with codecs.open(fname, 'w', 'utf-8') as f:
        for d in data:
            f.write(data_piece_to_line(d))

def download_citybeat_data(save_dir=None, source=None, fname=None, limit=100):
    
    if fname is None:
        fname = '%s.csv' % source
        print 'saving download as %s%s' % (save_dir, fname)

    with codecs.open('%s%s' % (save_dir, fname), 'w', 'utf-8') as f:
        
        # download and save tweets
        get_data_and_write_to_file(f, source, limit)

def read_citybeat_data(source, read_dir=None, fname=None, limit=100):

    bad_count = 0
    data = []

    with codecs.open('%s%s' % (read_dir, fname), 'r', 'utf-8') as f:
        for line in f:
            if (limit is not None) and (len(data) > limit):
                break

            line = line.replace(u'\n', u'')
            try:
                d = DataPiece(*line.split(u'\t'))
            except:
                bad_count += 1
                continue
            
            if (source is 'twitter_all') or (d.get_source() == source):
                data.append(d)
            else:
                raise Exception('source not recognized')

    # sort data inplace by created time
    data.sort(key=lambda x: int(x.get_created_time()), reverse=False)

    print "%s lines of data could not be read" % bad_count
    return data

def return_all_datasets(sources=['twitter_all'], **kwargs):
    """ kwargs are read_citybeat_data's optional args
    """
    data = read_citybeat_data('twitter_all', **kwargs)
    
    datasets = {}    
    datasets['twitter_all'] = data[:]

    non_all_sources = list(set(sources) - set(['twitter_all']))
    if len(non_all_sources) == 0:
        return datasets
    else:
        for source in non_all_sources:
            datasets[source] = []

        for d in data:
            if d.get_source() in datasets.keys():
                datasets[d.get_source()].append(d)

        return datasets


def determine_split_date(sorted_data, train_frac):

    ind = int(math.ceil(float(train_frac) * len(sorted_data)))
    return int(sorted_data[ind].get_created_time())

def split_train_and_test(data, split_date, gap=None):
    
    train = []
    test = []

    

    for d in data:
        if int(d.get_created_time()) <= split_date:
            train.append(d)
        elif int(d.get_created_time()) > split_date + gap:
            test.append(d)
    
    return train, test

def get_smallest_dataset(datasets):
    # TODO: improve klunky implementation
    smallest_dataset = None
    
    for key in datasets.keys():

        if (smallest_dataset is None) or (len(datasets[key]) < len(datasets[smallest_dataset])):
            smallest_dataset = key

    return smallest_dataset

def calc_min_train_test_sizes(split_datasets):
    # TODO: improve klunky implementation
    min_train_size = None
    min_test_size = None
    
    for key in split_datasets.keys():
        train_size = len(split_datasets[key][0]) 
        test_size = len(split_datasets[key][1])

        if (min_train_size is None) or (train_size < min_train_size):
            min_train_size = train_size

        if (min_test_size is None) or (test_size < min_test_size):
            min_test_size = test_size
        
    return min_train_size, min_test_size

def create_train_and_test_datasets(read_dir=None, fname=None, train_frac=None,
                                   split_date=None, gap=None, limit=100, 
                                   sampling=True, sources=None):
                                   
    # set seed so that if re-ran same train/test sets are produced
    seed(12234)

    # get all data
    datasets = return_all_datasets(read_dir=read_dir, fname=fname, limit=limit, sources=sources)

    if split_date is None and train_frac is None:
        raise Exception('must specify split date or train frac')
    elif split_date is not None and train_frac is not None:
        raise Exception('cannot specify split date AND train frac')

    if split_date is None:
        # find the smallest dataset to determine split date
        smallest_dataset = get_smallest_dataset(datasets)
        split_date = determine_split_date(datasets[smallest_dataset], train_frac)
    
    print 'spliting data at %s' % split_date
    for key in datasets.keys():
        datasets[key] = split_train_and_test(datasets[key], split_date, gap=gap)

    trainsets = {}
    testsets = {}

    for key in datasets.keys():

        if sampling:
            min_train_size, min_test_size = calc_min_train_test_sizes(datasets)
            trainsets[key] = choice(datasets[key][0], size=min_train_size, replace=False)
            testsets[key] = choice(datasets[key][1], size=min_test_size, replace=False)
        else:
            trainsets[key] = datasets[key][0]
            testsets[key] = datasets[key][1]
                
    return trainsets, testsets

def load_and_save_train_and_test_datasets(read_fname, read_dir, 
                                          pickle_prefix, save_dir, limit=100, 
                                          gap=None, sampling=True, ngram_range=None,
                                          min_df=None, max_df=None, min_user=None, 
                                          split_date=None, train_frac=None, train_set=None, test_set=None): 
    if train_set is None or test_set is None:
        datasets_to_get = ['twitter_all', 'twitter_instagram', 'twitter_iphone', 
                    'twitter_android', 'twitter_foursquare']
    else:
        datasets_to_get = [train_set, test_set]

    trainsets, testsets = create_train_and_test_datasets(read_dir=read_dir, 
                                                         fname=read_fname, 
                                                         limit=limit, gap=gap, 
                                                         sampling=sampling, 
                                                         split_date=split_date, 
                                                         train_frac=train_frac, 
                                                         sources=datasets_to_get)


    sources = ['twitter_iphone', 'twitter_android', 'twitter_foursquare', 
               'twitter_instagram', 'twitter_all'] 

    train_test_combos = [(x, y) for x in sources for y in sources]

    print 'vectorizing/pickling train/train combos'
    for train, test in train_test_combos:
        
        if (train_set is not None) and (train != train_set):
            continue

        if (test_set is not None) and (test != test_set):
            continue

        X_training, X_testing, ngrams = vectorize_data(trainsets[train], 
                                                       testsets[test], 
                                                       ngram_range, min_df, 
                                                       max_df, min_user)


        assert X_training.get_shape()[1] == X_testing.get_shape()[1], 'train and test have different ngrams!'
        ngram_counts = get_ngram_counts(ngrams, X_testing)

        print 'train: %s, test: %s, trainX shape: %s, testX shape: %s' \
            % (train, test, X_training.get_shape(), X_testing.get_shape())
                                                          

        fname = '%strain_%s_test_%s' % (pickle_prefix, train, test)

        pickle_data(save_dir, fname + '_ngrams.p', ngrams)
        pickle_data(save_dir, fname + '.p', trainsets[train], testsets[test], 
                    X_training, X_testing, ngrams, ngram_counts)

        pickle_data(save_dir, fname + '_train_X.p', X_training)


        training_coordinates = np.array([d.get_coordinate_pair() for d in trainsets[train]])
        np.save(save_dir + fname + '_train_coordinates.npy', training_coordinates)

    print 'data saved to: %s' % save_dir
        
def vectorize_data(training, testing, ngram_range, min_df,
                   max_df, min_user):

    vectorizer = get_vectorizer(training, min_user, ngram_range, min_df, max_df)
    X_training = vectorizer.transform([d.get_info() for d in training])
    X_testing = vectorizer.transform([d.get_text() for d in testing])
    
    X_training = X_training.tocsc()
    X_testing = X_testing.tocsr()

    return X_training, X_testing, vectorizer.get_feature_names()

def get_ngram_counts(ngrams, counts_mat):

    counts_mat = counts_mat.sum(0)
    counts = {}
    for i, ngram in enumerate(ngrams):
        counts[ngram] = counts_mat[0][0, i]
    return counts

def pickle_data(save_dir, fname, *items_to_save):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open('%s%s' % (save_dir, fname), "w") as f:
        pickle.dump(items_to_save, f) 
        
def load_data_from_pickle(train_set, test_set, read_dir, pickle_prefix):
    fname = '%strain_%s_test_%s.p' % (pickle_prefix, train_set, test_set)
    with open('%s%s' % (read_dir, fname),  "r") as f:
        out = pickle.load(f)
        if len(out) > 1:
            return out
        else:
            return out[0]

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--source", default='tweets', type=str)
    parser.add_argument("--gap", default=0, type=int)
    parser.add_argument("--sampling", action="store_true")

    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download_dir", default='./', type=str)
    parser.add_argument("--download_fname", default='demo_data.csv', type=str)

    #parser.add_argument("--split_train_test", action="store_true")
    parser.add_argument("--model_data_dir", default='./', type=str)
    parser.add_argument("--pickle_prefix", default='', type=str)


    parser.add_argument("--ngram_max_len", default=3, type=int)
    parser.add_argument("--min_df", default=16, type=int)
    parser.add_argument("--max_df", default=0.50, type=float)
    parser.add_argument("--min_user", default=10, type=int)
    
    # GMT midnight may 3rd 2014 = 1399075200
    parser.add_argument("--split_date", default=None, type=int)
    parser.add_argument("--train_frac", default=0.9, type=float) 
    parser.add_argument("--train_set", default='twitter_all', type=str)
    parser.add_argument("--test_set", default='twitter_all', type=str)
    args = parser.parse_args()



    NGRAM_RANGE = (1, args.ngram_max_len)

    if args.download:
        download_citybeat_data(save_dir=args.download_dir, source=args.source,
                               fname=args.download_fname, limit=args.limit) 

    else:
        load_and_save_train_and_test_datasets(args.download_fname, 
                                              args.download_dir,
                                              args.pickle_prefix, 
                                              args.model_data_dir, 
                                              limit=args.limit, 
                                              gap=args.gap, 
                                              sampling=args.sampling, 
                                              ngram_range=NGRAM_RANGE, 
                                              min_df=args.min_df, 
                                              max_df=args.max_df,
                                              min_user=args.min_user,
                                              split_date=args.split_date,
                                              train_frac=args.train_frac,
                                              train_set=args.train_set, 
                                              test_set=args.test_set)
