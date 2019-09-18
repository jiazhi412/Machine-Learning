import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    # TP = 0
    # FN = 0
    # FP = 0
    # TN = 0
    # for i in range(len(real_labels)):
    #     if predicted_labels[i] == 1 and real_labels[i] == 1:
    #         TP = TP + 1
    #     elif predicted_labels[i] == 1 and real_labels[i] == 0:
    #         FP = FP + 1
    #     elif predicted_labels[i] == 0 and real_labels[i] == 1:
    #         FN = FN + 1
    #     elif predicted_labels[i] == 0 and real_labels[i] == 0:
    #         TN = TN + 1
    # if TP == 0:
    #     precision = 0
    #     recall = 0
    #     F1 = 0
    # else:
    #     precision = TP / (TP + FP)
    #     recall = TP / (TP + FN)
    #     F1 = 2 * precision * recall / (precision + recall)
    den = 0
    num = 0
    for i in range(len(real_labels)):
        den = den + real_labels[i] * predicted_labels[i]
        num = num + real_labels[i] + predicted_labels[i]
    F1 = den * 2 / num
    return F1


class Distances:
    @staticmethod
    # TODO
    def canberra_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sum = 0
        for i in range(len(point1)):
            mol = abs(point1[i] - point2[i])
            den = (abs(point1[i]) + abs(point2[i]))
            if mol == 0:
                sum = sum + mol
            else:
                sum = sum + mol/den
        return sum

    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sum = 0
        for i in range(len(point1)):
            sum = sum + (abs(point1[i] - point2[i])) ** 3
        res = sum ** (1/3)
        return res

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sum = 0
        for i in range(len(point1)):
            sum = sum + (abs(point1[i] - point2[i])) ** 2
        res = sum ** (1 / 2)
        return res

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sum = 0
        for i in range(len(point1)):
            sum = sum + point1[i] * point2[i]
        return sum

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        inner_product = Distances.inner_product_distance(point1,point2)
        mod1 = 0
        mod2 = 0
        for i in range(len(point1)):
            mod1 += point1[i] ** 2
            mod2 += point2[i] ** 2
        mod1 **= 1/2
        mod2 **= 1/2
        res = 1 - (inner_product / (mod1 * mod2))
        return res

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        euclidean = Distances.euclidean_distance(point1,point2)
        res = -1 * np.exp(-1/2 * (euclidean ** 2))
        return res



class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        num_k = 0
        F1_stats = []
        distance_funcs_lists = ['canberra','minkowski', 'euclidean', 'gaussian', 'inner_prod', 'cosine_dist']

        for i in range(len(distance_funcs_lists)):
            k = 1
            while k < 30 and k <= len(x_train):
                classfier = KNN(k, distance_funcs[distance_funcs_lists[i]])
                classfier.train(x_train, y_train)
                preds = classfier.predict(x_val)
                F1_stats.append(f1_score(y_val, preds))
                k += 2
                num_k += 1

        num_k //= 6
        ind = F1_stats.index(max(F1_stats))
        self.best_k = (ind % num_k) * 2 + 1
        self.best_distance_function = distance_funcs_lists[ind // num_k]
        self.best_model = KNN(self.best_k, distance_funcs[self.best_distance_function])
        self.best_model.train(x_train, y_train)


    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        num_k = 0
        F1_stats = []
        scaling_classes_lists = ['min_max_scale','normalize']
        distance_funcs_lists = ['canberra', 'minkowski', 'euclidean', 'gaussian', 'inner_prod','cosine_dist']
        for j in range(len(scaling_classes_lists)):
            scaler1 = scaling_classes[scaling_classes_lists[j]]()
            x_train_scale = scaler1(x_train)
            x_val_scale = scaler1(x_val)
            for i in range(len(distance_funcs_lists)):
                k = 1
                while k < 30 and k <= len(x_train):
                    classfier = KNN(k, distance_funcs[distance_funcs_lists[i]])
                    classfier.train(x_train_scale, y_train)
                    preds = classfier.predict(x_val_scale)
                    F1_stats.append(f1_score(y_val, preds))
                    k += 2
                    num_k += 1

        num_k //= 12
        ind = F1_stats.index(max(F1_stats))
        self.best_k = ((ind % (num_k*6)) % num_k) * 2 + 1
        self.best_distance_function = distance_funcs_lists[(ind % (num_k*6)) // num_k]
        self.best_model = KNN(self.best_k, distance_funcs[self.best_distance_function])
        self.best_scaler = scaling_classes_lists[ind // (num_k * 6)]
        scaler1 = scaling_classes[self.best_scaler]()
        x_train_scale = scaler1(x_train)
        self.best_model.train(x_train_scale, y_train)


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features = np.array(features)
        for i in range(len(features)):
            sum = 0
            for j in range(len(features[i])):
                sum += features[i][j] ** 2
            root = sum ** (1/2)
            if root == 0:
                pass
            else:
                for j in range(len(features[i])):
                    features[i][j] = features[i][j] / root
        return features


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.first_call = True
        self.maxs = None
        self.mins = None

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features = np.array(features)
        if self.first_call:
            maxs = []
            mins = []
            for i in range(len(features[0])):
                max = features[0][i]
                min = features[0][i]
                for j in range(len(features)):
                    if features[j][i] > max:
                        max = features[j][i]
                    if features[j][i] < min:
                        min = features[j][i]
                maxs.append(max)
                mins.append(min)
            self.maxs = maxs
            self.mins = mins

        for i in range(len(features[0])):
            MAX = self.maxs[i]
            MIN = self.mins[i]
            if MAX == MIN:
                for j in range(len(features)):
                    features[j][i] = 0
            else:
                for j in range(len(features)):
                    features[j][i] = (features[j][i] - MIN) / (MAX - MIN)
        self.first_call = False
        return features
