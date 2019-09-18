from knn import KNN
from data import data_processing
from utils import Distances, HyperparameterTuner, NormalizationScaler, MinMaxScaler

def main():
    distance_funcs = {
        'canberra': Distances.canberra_distance,
        'minkowski': Distances.minkowski_distance,
        'euclidean': Distances.euclidean_distance,
        'gaussian': Distances.gaussian_kernel_distance,
        'inner_prod': Distances.inner_product_distance,
        'cosine_dist': Distances.cosine_similarity_distance,
    }

    x_train, y_train, x_val, y_val, x_test, y_test = data_processing()

    print('x_train shape = ', x_train.shape)
    print('y_train shape = ', y_train.shape)

    classfier = KNN(k=5,distance_function=distance_funcs['canberra'])
    classfier.train(x_train,y_train)
    pred = classfier.predict(x_val)
    s = 0


if __name__ == '__main__':
    main()