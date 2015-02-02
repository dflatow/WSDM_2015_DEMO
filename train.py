#!/usr/bin/env python -W ignore::DeprecationWarning
__author__ = 'daflatow'

import numpy as np
import cPickle as pickle
import itertools
import time
import os
import codecs
import logging
from argparse import ArgumentParser
import random
import math
import process_data as download_util
from tool import haversine
import ngram_gmm
import resource


logging.basicConfig(filename='predict.log',level=logging.DEBUG, 
                    format='%(levelname)s %(asctime)s %(message)s')

logging.debug('RUNNING PREDICT')

def list_to_tab_del_uni_line(data):
    return u'\t'.join([unicode(x) for x in data]) + u'\n'

def calc_mean_dist_to_gmm_center(gmm, coordinates):

    if len(coordinates) == 0:
        return None

    errors = []
    for c in coordinates:
        errors.append(haversine(c[0], c[1], gmm.means_[0][0], gmm.means_[0][1]))

    return np.mean(errors), len(errors)

def calc_start_end_indecies(length, chunks, current_chunk):
    obs_per_chunk = int(math.floor(length / float(chunks)))
    split_indecies = range(0, length + 1, obs_per_chunk)
    
    split_indecies[-1] = length
    return split_indecies[current_chunk], split_indecies[current_chunk + 1]

def compute_predict(ngrams, training_coordinates, X_training, 
                    maximum_area, minimum_ratio, 
                    max_iteration=None, model_dir=None, 
                    chunks=1, current_chunk=0, train_set=None, test_set=None):
    """compute the predictions for tweets in testing set under 
    specific condition (parameter pairs)"""

    model_out_dir = '%smodel_output/' % model_dir
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
        
    print 'shape of training matrix: ', X_training.shape
    start, end = calc_start_end_indecies(X_training.get_shape()[1], 
                                         chunks, current_chunk)

    out_prefix = "%strain_%s_test_%s_area_%0.2f_ratio_%0.2f" % (model_out_dir, 
                                train_set, test_set, maximum_area, minimum_ratio) 

    is_geo_filename = '%s_geograms_%s_chunk.csv' % (out_prefix, current_chunk)
    
           
    print 'iteritevely fitting gaussians'
    with codecs.open(is_geo_filename, 'w', 'utf-8') as is_geo_file:
        geo_info_list = []
        gmms = []

        for ngram_idx in range(start, end):

            ngram = ngrams[ngram_idx]            
            if ngram_idx % 100 == 0:
                logging.info('chunk %s, starting %s' % (current_chunk, ngram))
                logging.info('chunk %s, %0.2f' % (current_chunk, 100.0 - (100 * (end - ngram_idx) / float(end-start))))

            #if ngram_idx%100==0:
            #    logger.info("Working on ngram index " + str(ngram_idx))
            is_geo, gmm, coordinates, current_mark = ngram_gmm.repeat_fit(training_coordinates, 
                                X_training, ngram_idx, maximum_area, 
                                minimum_ratio, iteration_limit=max_iteration)

            if is_geo:
                geo_info_list.append([is_geo, gmm.means_[0], ngram])
                gmms.append([ngram_idx] + list(gmm.means_[0]) + list(np.reshape(gmm.covars_[0], (4, 1))))
                mean_dist, nobs = calc_mean_dist_to_gmm_center(gmm, coordinates)
                is_geo_file.write(list_to_tab_del_uni_line([ngram_idx, is_geo, ngram, 
                                                            gmm.means_[0][0], gmm.means_[0][1], 
                                                            mean_dist, nobs])) 

            else:
                geo_info_list.append([is_geo, None, ngram])
                gmms.append([ngram_idx] + [np.nan]*6)
                is_geo_file.write(list_to_tab_del_uni_line([ngram_idx, is_geo, ngram] + [np.nan]*4)) 
                
            
    
    gmms_filename = '%s_gmms_%s_chunk.npy' % (out_prefix, current_chunk)
    np.save(gmms_filename, np.array(gmms))


def main(args):
    data_dir = '%s' % args.model_dir
    

    print 'before load data mem :', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000

    fname = '%strain_%s_test_%s' % (args.pickle_prefix, args.train_set, args.test_set)

    training_coordinates = np.load(data_dir + fname + '_train_coordinates.npy') 
    X_training = download_util.load_data_from_pickle(args.train_set, 
                                                    args.test_set + '_train_X', 
                                                    data_dir, 
                                                    args.pickle_prefix)

    ngrams = download_util.load_data_from_pickle(args.train_set, 
                                                    args.test_set + '_ngrams', 
                                                    data_dir, 
                                                    args.pickle_prefix)


    compute_predict(ngrams, training_coordinates, 
                    X_training, args.max_area, args.ratio, 
                    max_iteration=args.max_iteration, 
                    model_dir=args.model_dir, train_set=args.train_set, 
                    test_set=args.test_set,
                    chunks=args.chunks, current_chunk=args.current_chunk)




if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument("max_area", nargs='?', default=4.0, type=float)
    parser.add_argument("ratio", nargs='?', default=0.8, type=float)
    parser.add_argument("--train_set", default='twitter_all', type=str)
    parser.add_argument("--test_set", default='twitter_all', type=str)
    parser.add_argument("--model_dir", default='./', type=str)
    parser.add_argument("--pickle_prefix", default='', type=str)
    parser.add_argument("--error_exp", default=16, type=int)
    parser.add_argument("--min_weight_ratio", default=0.001, type=float) 
    parser.add_argument("--n_components", default=1, type=int)
    parser.add_argument("--max_iteration", default=20, type=int)
    parser.add_argument("--chunks", default=1, type=int)
    parser.add_argument("--current_chunk", default=0, type=int)
    args = parser.parse_args()
    
    main(args)

