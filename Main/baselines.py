import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import sklearn as sklearn
import multiprocessing

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import RandomTreesEmbedding
from sklearn.neural_network import MLPRegressor
# from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import Imputer
from sklearn import metrics
# import statsmodels.api as sm

from keras.models import Sequential
from keras.layers import Dense

plt.style.use("ggplot")
n_cores = multiprocessing.cpu_count() # Getting the number of cores for multiprocessing


def get_gaussian_process_regressor():
    gp = GaussianProcessRegressor()
    return [gp], ['Gaussian Process']


def get_mlp_regressor(num_hidden_units=51):
    mlp = MLPRegressor(hidden_layer_sizes=num_hidden_units)
    return [mlp], ['Multi-Layer Perceptron']


def get_ensemble_models():
    rf = RandomForestRegressor(
        n_estimators=51, min_samples_leaf=5, min_samples_split=3, random_state=42,
        n_jobs=int(0.8*n_cores))
    bag = BaggingRegressor(n_estimators=51, random_state=42, n_jobs=int(0.8*n_cores))
    extra = ExtraTreesRegressor(n_estimators=71, random_state=42, n_jobs=int(0.8*n_cores))
    ada = AdaBoostRegressor(random_state=42)
    grad = GradientBoostingRegressor(n_estimators=101, random_state=42)
    classifier_list = [rf, bag, extra, ada, grad]
    classifier_name_list = ['Random Forests', 'Bagging',
                            'Extra Trees', 'AdaBoost', 'Gradient Boost']
    return classifier_list, classifier_name_list


def print_evaluation_metrics(trained_model, trained_model_name, X_test, y_test):
    print('--------- For Model: ', trained_model_name, ' ---------\n')
    predicted_values = trained_model.predict(X_test)
    print("Mean absolute error: ",
          metrics.mean_absolute_error(y_test, predicted_values))
    print("Median absolute error: ",
          metrics.median_absolute_error(y_test, predicted_values))
    print("Mean squared error: ", metrics.mean_squared_error(
        y_test, predicted_values))
    print("R2: ", metrics.r2_score(y_test, predicted_values))
    plt.scatter(y_test, predicted_values, color='black')
    # plt.plot(x, y_pred, color='blue', linewidth=3)
    plt.title(trained_model_name)
    plt.xlabel('$y_{test}$')
    plt.ylabel('$y_{predicted}/y_{test}$')
    plt.savefig('%s.png' %trained_model_name, bbox_inches='tight')
    print("---------------------------------------\n")


def print_evaluation_metrics2(trained_model, trained_model_name, X_test, y_test):
    print('--------- For Model: ', trained_model_name, ' --------- (Train Data)\n')
    predicted_values = trained_model.predict(X_test)
    print("Mean absolute error: ",
          metrics.mean_absolute_error(y_test, predicted_values))
    print("Median absolute error: ",
          metrics.median_absolute_error(y_test, predicted_values))
    print("Mean squared error: ", metrics.mean_squared_error(
        y_test, predicted_values))
    print("R2: ", metrics.r2_score(y_test, predicted_values))
    plt.scatter(y_test, predicted_values/y_test, color='black')
    # plt.plot(x, y_pred, color='blue', linewidth=3)
    plt_name = trained_model_name + " (Train Data)"
    plt.title(plt_name)
    plt.xlabel('$y_{test}$')
    plt.ylabel('$y_{predicted}/y_{test}$')
    plt.savefig('%s.png' %plt_name, bbox_inches='tight')
    print("---------------------------------------\n")


def LinearModel(X_train, y_train, X_val, y_val):
    regr = linear_model.LinearRegression(n_jobs=int(0.8*n_cores)).fit(X_train, y_train)
    y_pred = regr.predict(X_val)

    # print('--------- For Model: LinearRegression --------- \n')
    # print('Coefficients: \n', regr.coef_)
    print("Mean squared error: %.2f" % mean_squared_error(y_val, y_pred))
    print("R2: ", sklearn.metrics.r2_score(y_val, y_pred))

# =============================================================================
#     plt.scatter(y_val, y_pred/y_val, color='black')
#     # plt.plot(x, y_pred, color='blue', linewidth=3)
#     plt.title('Linear Model Baseline')
#     plt.xlabel('$y_{test}$')
#     plt.ylabel('$y_{predicted}/y_{test}$')
#     plt.savefig('Linear Model Baseline.png', bbox_inches='tight')
# =============================================================================
    
    return


def simple_neural_network(X_train, y_train, X_val, y_val):
    pass
    """
    X_train_normalized = X_train.values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(X_train_normalized)

    # Run the normalizer on the dataframe
    x = pd.DataFrame(x_scaled)

    y_train_normalized = y_train.values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    y_scaled = min_max_scaler.fit_transform(y_train_normalized)

    # Run the normalizer on the dataframe
    y = pd.DataFrame(y_scaled)

    X_val_normalized = X_val.values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    x_val_scaled = min_max_scaler.fit_transform(X_val_normalized)

    # Run the normalizer on the dataframe
    x_v = pd.DataFrame(x_val_scaled)

    y_val_normalized = y_val.values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    y_val_scaled = min_max_scaler.fit_transform(y_val_normalized)

    # Run the normalizer on the dataframe
    y_v = pd.DataFrame(y_val_scaled)
    """


    model = Sequential()
    model.add(Dense(units=34, activation='relu', input_dim=763))
    model.add(Dense(units=20, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=3, activation='sigmoid'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x, y, epochs = 5, batch_size=264)
    print("finished fitting")
    print_evaluation_metrics(model, "NN", x_v, y_v)
    print_evaluation_metrics2(model, "NN", x, y)
    return


def TreebasedModel(X_train, y_train, X_val, y_val):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    y_train = np.squeeze(y_train)
    y_val = np.squeeze(y_val)

    classifier_list, classifier_name_list = get_ensemble_models()
    for classifier, classifier_name in zip(classifier_list, classifier_name_list):
        classifier.fit(X_train, y_train)
        print_evaluation_metrics2(classifier, classifier_name, X_train, y_train)
        print_evaluation_metrics(classifier, classifier_name, X_val, y_val)
    return

def kmeans(X_train, y_train, X_val, y_val):
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, verbose=0, n_jobs=int(0.8*n_cores)).fit(X_train)
    c_train = kmeans.predict(X_train)
    c_pred = kmeans.predict(X_val)
    centroids = kmeans.cluster_centers_
    for i in range(n_clusters):
        print('--------analyzing cluster %d--------' %i)
        train_mask = c_train==i
        std_train = np.std(y_train[train_mask])
        mean_train = np.mean(y_train[train_mask])
        print("# examples & price mean & std for training set within cluster %d is:(%d, %.2f, %.2f)" %(i, train_mask.sum(), np.float(mean_train), np.float(std_train)))
        pred_mask = c_pred==i
        std_pred = np.std(y_val[pred_mask])
        mean_pred = np.mean(y_val[pred_mask])
        print("# examples & price mean & std for validation set within cluster %d is:(%d, %.2f, %.2f)" %(i, pred_mask.sum(), np.float(mean_pred), np.float(std_pred)))
        if pred_mask.sum() == 0:
            print('Zero membered test set! Skipping the test and training validation.')
            continue
        LinearModel(X_train[train_mask], y_train[train_mask], X_val[pred_mask], y_val[pred_mask])
        print('--------Finished analyzing cluster %d--------' %i)
    
    
    return c_pred, centroids


if __name__ == "__main__":

    X_train = pd.read_csv('../Data/data_cleaned_train_X.csv')
    y_train = pd.read_csv('../Data/data_cleaned_train_y.csv')

    X_val = pd.read_csv('../Data/data_cleaned_val_X.csv')
    y_val = pd.read_csv('../Data/data_cleaned_val_y.csv')

    print("--------------------Linear Regression--------------------")
    LinearModel(X_train, y_train, X_val, y_val)
    # print("--------------------Tree-based Model--------------------")
    # TreebasedModel(X_train, y_train, X_val, y_val)
    # print("--------------------Neural Net--------------------")
    # simple_neural_network(X_train, y_train, X_val, y_val)
    print("--------------------KMeans Clustering--------------------")
    c_pred, centroids = kmeans(X_train, y_train, X_val, y_val)
    
    pass
