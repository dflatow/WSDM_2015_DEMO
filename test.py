import codecs
import os
import re
import numpy as np
from argparse import ArgumentParser
import process_data as download_util
from tool import haversine, point_included_by_eclipse
import difflib

def chunks_to_list(file_prefix, model_dir, parse_func):

    files = get_all_files(file_prefix, model_dir)
    data = []
    for f in files:
        data.extend(list(parse_func(f)))
        
    data.sort(key=lambda d: d[0]) 
    return data

def get_one_geo_info_list(geograms_file):
    
    with codecs.open(geograms_file, 'r', 'utf-8') as f:
        data = [x[:-1].split('\t') for x in f.readlines()]
        

        for i, d in enumerate(data):
            for j in [0, 6]:
                try:
                    data[i][j] = int(d[j])
                except:
                    data[i][j] = np.nan

            for j in [3, 4, 5]:
                data[i][j] = float(d[j])

            for j in [1]:
                data[i][j] = d[j] == u'True'

        return data
    

def get_all_files(file_prefix, model_dir):

    model_output_dir = os.path.join(model_dir, 'model_output/')
    files = []
    for f in os.listdir(model_output_dir):
        if re.search(file_prefix, f) is not None:
            files.append(model_output_dir + f)
    return files

def evaluate_model(max_area, min_ratio, train_set, test_set, model_dir, 
                   training, testing, X_testing, ngrams, ngram_counts, text_sim_thresh,
                   nearby_dist, write_train_data=False):

    out_dir = os.path.join(model_dir, 'model_output/')

    out_tag = "train_%s_test_%s_area_%0.2f_ratio_%0.2f" % (train_set, test_set, 
                                                              max_area, min_ratio) 
    
    if write_train_data:
        #write training data to file
        training_filename = '%s%s_training.csv' % (out_dir, out_tag) 
        with codecs.open(training_filename, 'w', 'utf-8') as training_file:
            for dat in training:
                training_line = list(dat.get_data())
                training_file.write(list_to_tab_del_uni_line(training_line))
    

    geograms_file_prefix = "%s_geograms" % out_tag
    geo_info_list = chunks_to_list(geograms_file_prefix, model_dir, get_one_geo_info_list)

    gmms_file_prefix = "%s_gmms" % out_tag
    gmms_list = chunks_to_list(gmms_file_prefix, model_dir, np.load)

    #for i in range(30):
    #    print list(gmms_list[i]) + geo_info_list[i]
    #raise

    geograms_filename = '%s%s_all_geograms.csv' % (out_dir, out_tag) 
    with codecs.open(geograms_filename, 'w', 'utf-8') as geo_file:
        for gram in geo_info_list:
            geo_file.write(list_to_tab_del_uni_line(gram))

    geo_aware_cnt = len(geo_info_list)
    non_geo_aware_cnt = 0
    for obj in geo_info_list:
        if obj[1] == True:
            geo_aware_cnt += 1
        else:
            non_geo_aware_cnt += 1

    error = []

    n_tweets_predicted = 0
    multiple_ngram_cnt = 0
    precise_count = 0

    n_effective_ngrams = []
    effective_ngrams_filename = '%s%s_testing.csv' % (out_dir, out_tag) 
    mult_ngrams = []
    max_dists = []

    print '\nevaluating on test set'
    with codecs.open(effective_ngrams_filename, 'w', 'utf-8') as effective_ngrams_file:
        for n, doc in enumerate(testing):
            # indexes is the indexes of ngrams for a specific tweets
            indexes = list(X_testing[n, :].nonzero()[1])
            # keep indexes of ngrams that are geo-aware
            indexes = [x for x in indexes if geo_info_list[x][1] == True]
            # remove those ngrams which are covered by other ngrams

            to_del = []
            for _i in range(len(indexes)):
                for _j in range(_i+1, len(indexes)):
                    if ngrams[indexes[_j]].find(ngrams[indexes[_i]]) != -1:
                        to_del.append(indexes[_i])

                    if ngrams[indexes[_i]].find(ngrams[indexes[_j]]) != -1:
                        to_del.append(indexes[_j])
                        #print '%s > %s' % (ngrams[indexes[_j]], ngrams[indexes[_i]])
                        

            # remove those ngrams that are more general 
            # ( e.g. Madison Square Garden is kept and Madison Square is removed)
            for val in set(to_del):
                indexes.remove(val)

            
            tmp_ngrams = [ngrams[x] for x in indexes]    
            tmp_locations = [[geo_info_list[x][3], geo_info_list[x][4]] for x in indexes]    
            tmp_counts = [ngram_counts[x] for x in indexes]    
            tmp_means = [gmms_list[x][1:3] for x in indexes]
            tmp_covars = [np.reshape(gmms_list[x][3:], (2, 2)) for x in indexes]                
    
            #if len(indexes) > 0:
            #    print tmp_locations[0], tmp_means[0]
            #    print point_included_by_eclipse(tmp_means[0], tmp_covars[0], tmp_locations[0], 2.0)

            
            has_prediction, curr_ngram, ngram_mean, ngram_covar, mult_ngrams, overlap, sim_text, nearby = \
                                    choose_effective_ngram(tmp_ngrams, tmp_locations, tmp_counts, tmp_means, tmp_covars,
                                                           text_sim_thresh=text_sim_thresh, nearby_dist=nearby_dist)
            
            if mult_ngrams:
                multiple_ngram_cnt += 1

            if ngram_mean is not None:
                error_distance = haversine(ngram_mean[0], ngram_mean[1], float(doc.get_lat()), float(doc.get_lng()))
                
                if has_prediction:
                    error.append(error_distance)
                    n_tweets_predicted += 1            

                    if point_included_by_eclipse(ngram_mean, ngram_covar, 
                                                 [float(doc.get_lat()), float(doc.get_lng())], 2.0):
                        precise_count += 1


            
            else:
                error_distance = np.nan
            

            ngrams_string = u'__'.join(tmp_ngrams)


            effective_ngrams_line = list(testing[n].get_data()) + [ngram_mean[0], ngram_mean[1], 
                                                                       error_distance, has_prediction, curr_ngram, mult_ngrams, 
                                                                       overlap, sim_text, nearby, ngrams_string]

            effective_ngrams_file.write(list_to_tab_del_uni_line(effective_ngrams_line))

    try:
        pct_with_mult_ngram = 100.0 * multiple_ngram_cnt / float(n_tweets_predicted)
    except:
        pct_with_mult_ngram = np.nan
    
    
    para = {
        'maximum_area': max_area,
        'min_ratio': min_ratio,
        'geo_aware_ngram_cnt': geo_aware_cnt,
        'non_geo_aware_ngram_cnt': non_geo_aware_cnt,
        'mean_error': np.mean(error),
        'std_error': np.std(error),
        'median_error': np.median(error),
        'n_tweets_predicted': n_tweets_predicted,
        'train_cnt': len(training),
        'test_cnt': len(testing),
        'pct_with_mult_ngram': pct_with_mult_ngram,
        'precision' : 100.0 * precise_count / n_tweets_predicted
        }


    pctls = range(0, 101, 1)
    for pctl in pctls:
        try:
            para["error_pctl_" + str(pctl) ] = np.percentile(error, pctl)
        except:
            pass


    results_filename = '%s%s_results' % (out_dir, out_tag)

    print '\nprinting results'
    with codecs.open(results_filename, 'w', 'utf-8') as results_file:
        results_file.write(unicode(para))

def list_to_tab_del_uni_line(data):
    return u'\t'.join([unicode(x) for x in data]) + u'\n'

def compute_max_pairwise_dist(locs):
    distances = []
    for i, loc_i in enumerate(locs):
        for j, loc_j in enumerate(locs[i+1:]):
            distances.append(haversine(loc_i[0], loc_i[1], loc_j[0], loc_j[1]))

    return max(distances)


def choose_effective_ngram(ngrams, locations, counts, means, covars, 
                            text_sim_thresh=0.5, nearby_dist=0.5):
    
    if len(ngrams) == 0:
        return tuple([False, None, [np.nan, np.nan], None] + [False]*4)
    
    if len(ngrams) == 1:
        return tuple([True] + ngrams + means + covars + [False]*4)
    
    overlap = overlapping(ngrams)
    sim_text = similar_text(ngrams, text_sim_thresh)
    nearby = locations_nearby(locations, nearby_dist)

    index = counts.index(max(counts))

    # use prediction only when all ngrams are nearby
    has_prediction = nearby

    return has_prediction, ngrams[index], means[index], covars[index], True, overlap, sim_text, nearby


def parse_line(line):
    out = []    
    for dat in line.split(','):
        out.append(dat.replace("u'", '').replace("'", '').replace('[', '').replace(']', '').replace('\n', '').strip())
    return out

def similar_text(ngrams, text_sim_thresh=0.5):

    if len(ngrams) <= 1:
        raise Exception('needs more than one ngram')
    
    # check all pairwise ngrams have similar text
    similar = True
    for i, ngram_i in enumerate(ngrams):
        for j, ngram_j in enumerate(ngrams):
            if i != j:
                similar = ngram_similar_ngram(ngram_i, 
                    ngram_j, thresh=text_sim_thresh) and similar
    return similar

def overlapping(ngrams):
    
    if len(ngrams) <= 1:
        raise Exception('needs more than one ngram')
        
    # check all pairwise ngrams are overlapping
    overlap = True
    for i, ngram_i in enumerate(ngrams):
        for j, ngram_j in enumerate(ngrams):
            if i != j:
                overlap = ngram_overlap_ngram(ngram_i, 
                    ngram_j) and overlap
    return overlap

def locations_nearby(locations, nearby_dist):
    
    if len(locations) <= 1:
        raise Exception('needs more than one location')
        
    # check all pairwise ngrams are overlapping
    nearby = True
    for i, location_i in enumerate(locations):
        for j, location_j in enumerate(locations):
            if i != j:
                nearby = location_near_location(location_i, 
                    location_j, nearby_dist) and nearby
    return nearby

def ngram_similar_ngram(A, B, thresh):
    return  difflib.SequenceMatcher(None, A, B).ratio() > thresh

def ngram_overlap_ngram(A, B):
    A_set = set(len_two_tokens(A))
    B_set = set(len_two_tokens(B))
    return  len(A_set.intersection(B_set)) > 0

def location_near_location(A, B, nearby_dist):
    return haversine(A[0], A[1], B[0], B[1]) <= nearby_dist

def len_two_tokens(ngram):
    word_list = ngram.split()
    out = []
    for i in range(len(word_list) - 1):
        out.append('%s %s' % (word_list[i], word_list[i+1]))
    return out

if __name__ == "__main__":
    

    parser = ArgumentParser()
    parser.add_argument("max_area", nargs='?', default=4.0, type=float)
    parser.add_argument("min_ratio", nargs='?', default=0.8, type=float)
    parser.add_argument("--train_set", default='twitter_all', type=str)
    parser.add_argument("--test_set", default='twitter_all', type=str)
    parser.add_argument("--model_dir", default='./models/test/', type=str)
    parser.add_argument("--text_sim_thresh", default=0.6, type=float)
    parser.add_argument("--nearby_dist", default=0.5, type=float)
    parser.add_argument("--pickle_prefix", default='', type=str)
    parser.add_argument("--write_train_data", action='store_true')
    args = parser.parse_args()

    DATA_DIR = '%sdata/' % args.model_dir

    training, testing, _, X_testing, ngrams, _ = \
                download_util.load_data_from_pickle(args.train_set, 
                                                    args.test_set, 
                                                    DATA_DIR, 
                                                    args.pickle_prefix)
    X_testing = X_testing.tocsr()

    ngram_counts =  np.squeeze(np.array(X_testing.sum(0)))
    evaluate_model(args.max_area, args.min_ratio, args.train_set, args.test_set, 
                   args.model_dir, training, testing, X_testing, ngrams, ngram_counts, 
                   args.text_sim_thresh, args.nearby_dist,
                   write_train_data=args.write_train_data)
