from knn import KNN
from data import data_processing
from utils import Distances, HyperparameterTuner, NormalizationScaler, MinMaxScaler

def main():

    x_train, y_train, x_val, y_val, x_test, y_test = data_processing()

    print('x_train shape = ', x_train.shape)
    print('y_train shape = ', y_train.shape)

    scaler = MinMaxScaler()
    x_train = scaler(x_train)
    x_val = scaler(x_val)

    scaler = MinMaxScaler()
    x_test = scaler(x_test)


if __name__ == '__main__':
    main()