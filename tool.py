__author__ = 'eddiexie'

import math
import numpy as np
from math import radians, cos, sin, asin, sqrt, log
from convexhull import convexHull

def haversine(lat1, lng1, lat2, lng2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])

    # haversine formula
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # 6367 km is the radius of the Earth
    km = 6367 * c
    return km
    pass


def elipse_area(mean_, cov_, n_sigma):

    lat_in_km = 111.04
    lon_in_km = 84.13

    eigenvalues, eigenvectors = np.linalg.eig(cov_)
    idx = np.argsort(eigenvalues)[::-1]

    if eigenvalues[0] == eigenvalues[1]:
        assert(eigenvectors[0,0] != 0.0)

    if eigenvectors[idx[0], 0] == 0.0:
        return np.nan
        
    t_angle = eigenvectors[idx[0], 1]*1.0/eigenvectors[idx[0], 0]  #y/x
    angle = math.atan(t_angle)
    if angle < 0:
        angle = math.pi + angle

    major = math.sqrt(eigenvalues[idx[0]]) * n_sigma / 2.0
    minor = math.sqrt(eigenvalues[idx[1]]) * n_sigma / 2.0

    abs_major = math.sqrt((major * math.sin(angle) * lat_in_km)**2 + (major * math.cos(angle) * lon_in_km)**2)
    abs_minor = math.sqrt((minor * math.sin(angle) * lat_in_km)**2 + (minor * math.cos(angle) * lon_in_km)**2) 
    #print abs_major, abs_minor
    
    assert(major >= minor)

    """The squared relative lengths of the principal axes are given by the corresponding eigenvalues, from wikipedia"""
    return abs_minor * abs_major * math.pi

def eclipse_area_less_than(mean_, cov_, n_sigma, area=100):
    """ mean_ is a pair of (lat, lng) for the center of the eclipse. i.e mean_ vector. cov_ is the cov matrix,
    I am approximating the earth as a perfect sphere and one lat and one lng = 111 km, area is in unit of square km"""
    return elipse_area(mean_, cov_, n_sigma) < area

def point_included_by_eclipse(mean_, cov_, point, n_sigma):
    #WARNING, PROBABLY WRONG FUNCTION
    """ mean_ is a pair of (lat, lng) for the center of the eclipse. i.e mean_ vector. cov_ is the cov matrix,
    I am approximating the earth as a perfect sphere and one lat and one lng = 111 km, area is in unit of square km"""

    eigenvalues, eigenvectors = np.linalg.eig(cov_)
    idx = np.argsort(eigenvalues)[::-1]
    if eigenvalues[0] == eigenvalues[1]:
        assert(eigenvectors[0,0] != 0.0)
    
    # do not divide by zero
    if eigenvectors[idx[0], 0] == 0.0:
        return False

    t_angle = eigenvectors[idx[0], 1]*1.0/eigenvectors[idx[0], 0]  #y/x
    angle = math.atan(t_angle)
    if angle<0:
        angle = math.pi + angle

    relative_major_length = math.sqrt(eigenvalues[idx[0]])
    relative_minor_length = math.sqrt(eigenvalues[idx[1]])
    assert(relative_major_length>=relative_minor_length)

    ratio = relative_major_length*1.0/relative_minor_length
    assert(ratio>=0)

    x = point[0] - mean_[0]
    y = point[1] - mean_[1]

    # http://math.stackexchange.com/questions/108270/what-is-the-equation-of-an-ellipse-that-is-not-aligned-with-the-axis?rq=1
    term1 = (x*math.cos(angle) + y*math.sin(angle)) / ratio
    term2 = (y*math.cos(angle) - x*math.sin(angle))
    
    absolute_length_minor_axis = math.sqrt(eigenvalues[idx[1]]) * n_sigma
    # ideal_length is the length of minor_length when the point is on the ellipse
    ideal_length = math.sqrt(term1**2 + term2**2)

    if ideal_length > absolute_length_minor_axis:
        return False
    return True



def accuracy_of_word(gmm, area, n_sigma):
    accuracy = 0.0
    for i in range(gmm.means_.shape[0]):
        if eclipse_area_less_than(gmm.means_[i], gmm.covars_[i], n_sigma=n_sigma, area=area):
            accuracy += gmm.weights_[i]
            return True

    return accuracy > 0


def smallest_area_to_cover_new(coordinates, gmm):

    min_sigma = min_sigma_to_cover(coordinates, gmm)
    return elipse_area(gmm.means_[0], gmm.covars_[0], min_sigma)

def smallest_area_to_cover(gmm, n_sigma):
    # Do binary search on accuracy_of_word here and get an area
    low = 0.0
    high = 1000000.0
    eps = 0.000001
    rec = -1
    while(high-low > eps):
        mid = (low+high)/2
        if accuracy_of_word(gmm, mid, n_sigma):
            rec = mid
            high = mid
        else:
            low = mid + eps
    return rec


def point_included(n_sigma, point=None, mean_=None, cov_=None):
    return point_included_by_eclipse(mean_, cov_, point, n_sigma)

def min_sigma_to_cover(points_, gmm):
    
    points = convexHull(points_)

    min_sigma = 0.0
    if len(gmm.means_) > 1:
        raise Exception('muliple component gmm not supported')

    for point in points:
        sigma = bisection(point_included, point=point, mean_=gmm.means_[0], 
                          cov_=gmm.covars_[0])

        min_sigma = max(sigma, min_sigma)
        #print '%s,%s' % point 

    for point in points:
        assert bisection(point_included, point=point, mean_=gmm.means_[0], 
                          cov_=gmm.covars_[0]) <= min_sigma + 1e-5

    return min_sigma


def test_func(val, kw1=None, kw2=None):
    print kw1, kw2
    return val >= 0.123

def bisection(f, high=1e6, low=0.0, eps=1e-6, **kwargs):

    while (high - low) > eps:
        mid = (high + low) / 2.0
        if f(mid, **kwargs) == True:
            high = mid
        else:
            low = mid

    return mid

class TweetSimilarity():

    def __init__(self, size=5):

        self.SHINGLE_SIZE = size
        self.cache = {}

    def __get_shingles(self, text, ith_tweet):
        if ith_tweet in self.cache:
            return self.cache[ith_tweet]

        text = text.lower()
        shingles = set()

        for i in range(0, len(text)- self.SHINGLE_SIZE +1):
            shingles.add(text[i:i+self.SHINGLE_SIZE])

        self.cache[ith_tweet] = shingles

        return shingles


    def __jaccard(self, set1, set2):
        x = len(set1.intersection(set2))
        y = len(set1.union(set2))
        return x / float(y)

    def comp_similarity(self, text1, text2, first_index, second_index):
        """Given two tweet texts, compute the similarity between them"""


        shingles1 = self.__get_shingles(text1, first_index)
        shingles2 = self.__get_shingles(text2, second_index)

        return self.__jaccard(shingles1, shingles2)

def _compute_similarity_mat(text_list, shingle_size=4):
    """compute pairwise jaccard similarity for text in a list,
    returns numpy matrix with similarity scores"""

    # initialize
    tweet_sim = TweetSimilarity(size=shingle_size)
    n = len(text_list)
    similarity_mat = np.empty((n, n))
    similarity_mat[:] = np.nan

    # loop over pairs
    for j in xrange(n):

        for i in xrange(j+1):
            if ("I'm at " in text_list[i]) or ("I'm at " in text_list[j]):
                continue

            if ("http" in text_list[i]) or ("http" in text_list[j]):
                continue

            elif i == j:
                continue

            else:
                similarity_mat[i, j] = tweet_sim.comp_similarity(text_list[i], text_list[j], i, j)
                similarity_mat[j, i] = similarity_mat[i, j]

    return similarity_mat

def _is_similar_text_to_the_rest(text_list, jaccard_tresh=0.3):
    """determine which elements of text_list are similar (jaccard) to other elements,
    returns boolean array with same size as text list"""

    sim_mat = _compute_similarity_mat(text_list, shingle_size=4)
    sim_score = np.nanmean(sim_mat, 1)

    return sim_score > jaccard_tresh

def identify_similar_spam_tweets(tweets, jaccard_tresh=0.3):
    """filters elements from text list that are similar (jaccard) to other elements of the list"""

    text_list = [d.get_text() for d in tweets]

    return _is_similar_text_to_the_rest(text_list, jaccard_tresh=jaccard_tresh)
