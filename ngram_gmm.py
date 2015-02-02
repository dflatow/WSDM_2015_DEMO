__author__ = 'eddiexie'
"""
in function fit(), the logic is implemented via reading the source code of gmm.py in
https://github.com/reidpr/quac
"""

from sklearn.feature_extraction.text import CountVectorizer
from tool import smallest_area_to_cover, smallest_area_to_cover_new
from tool import point_included_by_eclipse
from sklearn.mixture import GMM
from tool import haversine, min_sigma_to_cover, elipse_area
import numpy as np
import sys
from convexhull import area_covered_by_points, convexHull

def repeat_fit(training_coordinates, X_training, ngram_index, max_area=5.0, 
               min_ratio = 0.8, iteration_limit=10, n_components=1,
               covariance_type='full', min_covariance=1e-6, min_obs=10):

    """This method repeatitatively fit gaussian for each ngrams, 
    and return label of data whether it's included for
    that Gaussian. Only applicable after run model.fit"""


    gmm = None
    rows = X_training[:, ngram_index].nonzero()[0]
    # mark whether certain data point shall be included in the gaussian
    current_mark = [True for x in range(len(rows))] 

    coordinate_index_pairs = [(training_coordinates[rows[i]], i) for i in range(len(rows))]

    is_geo_info = False
    pre_ratio = None

    iteration = 1
    while iteration <= iteration_limit:
        iteration += 1

        gmm = GMM(n_components=n_components, covariance_type=covariance_type, 
                  min_covar=min_covariance)

        current_data = [g[0] for g in 
                        coordinate_index_pairs if current_mark[g[1]]==True]

        if len(current_data) <= min_obs:
            gmm = None
            break

        gmm.fit(current_data)

        # MAKE THIS INTO A HELPER FUNCTION
        in_sigma_count = 0
        n_data_point = 0

        for i in range(len(coordinate_index_pairs)):
            if current_mark[coordinate_index_pairs[i][1]]:
                n_data_point += 1
                if point_included_by_eclipse(gmm.means_[0], gmm.covars_[0], 
                                             coordinate_index_pairs[i][0], 2.0):
                    in_sigma_count += 1
                else:
                    current_mark[coordinate_index_pairs[i][1]] = False

        current_data = [g[0] for g in 
                        coordinate_index_pairs if current_mark[g[1]]==True]

        #gmm.fit(current_data)
        # Ratio of points in the final ellipse
        ratio = in_sigma_count*1.0/len(rows) 
        #old_area = smallest_area_to_cover(gmm, 2.0)

        #new_area = smallest_area_to_cover_new(current_data, gmm)
        area = elipse_area(gmm.means_[0], gmm.covars_[0], 2.0)
        #area = area_covered_by_points(current_data)
        
        #if iteration == 2:
        #    print area, new_area, old_area

        #print min([x[0] for x in current_data])
        #print max([x[0] for x in current_data])


        #print min([x[1] for x in current_data])
        #print max([x[1] for x in current_data])
        #raise


        
        if ratio < min_ratio:
            break

        if area <= max_area:
            is_geo_info = True
            break

        if ratio == pre_ratio:
            break
        else:
            pre_ratio = ratio

            
    #print gmm.covars_[0]
    #print gmm.means_[0]
    coordinates = [c[0] for c in coordinate_index_pairs]
    return is_geo_info, gmm, coordinates, current_mark
"""

class Model():
    def __init__(self, error_exp=4, min_weight_ratio=0.001, n_components=1):
                 
        self.gmms = []
        self.weights = []
        self.error_exp = error_exp
        self.min_weight_ratio = min_weight_ratio
        self.n_components = n_components
        self.training_coordinates = None

    def get_parameters(self):
        # return the parameters associated with this model
        return {
            'min_weight_ratio': self.min_weight_ratio,
            'n_components': self.n_components
        }

    def __repr__(self):
        para = self.get_parameters()
        return 'min_weight_ratio=%f, n_component=%d' % (
            para['min_weight_ratio'],
            para['n_component']
        )

    def fit(self, training, X_training):
        # for all the ngrams, fit gmm for each one.

        #self.X = X_training
        #self.training_coordinates = [d.get_coordinate_pair() for d in training]
        self.n_doc, self.n_word = X_training.shape
   
        self.n_doc, self.n_word = self.X.shape
        self.best_predictions = []
        last_print = None
        
        print 'training initial gaussians...'
        for word in range(self.n_word):
            
            percent_finished = round((float(word)*100.0) / self.n_word)
            if (percent_finished % 10 == 0) and (last_print != percent_finished):
               print '%2.0f pct finished' % percent_finished 
               last_print = percent_finished


            rows = [pair for pair in self.X[:, word].nonzero()[0]]

            if len(rows) == 0:
                continue

            coordinates = [(training[k].get_lat(), 
                            training[k].get_lng()) for k in rows]

            gmm = GMM(n_components=self.n_components, covariance_type='full', 
                      min_covar=1e-7)

            gmm.fit(coordinates)
            self.gmms.append(gmm)

            # compute weights here "SAE error"
            # SAE error is computed for each gmm via 
            # averaging all the components in that gmm. see line 201 in gmm.py
            # the author's original code

            best_prediction = np.average(gmm.means_, axis=0, weights=gmm.weights_)
            self.best_predictions.append(best_prediction)

            distances = []
            for r in rows:
                dis = haversine(best_prediction[0], best_prediction[1], 
                                float(training[r].get_lat()), 
                                float(training[r].get_lng()))

                distances.append(dis)

            mean_error = np.mean(distances)

            weight = min(1, 1.0/(1+mean_error ** self.error_exp))
            self.weights.append(weight)
        
        print 'done fitting gaussians'

        self.n_gmms = len(self.gmms)
"""
