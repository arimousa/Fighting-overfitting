import itertools

import matplotlib.pyplot as plt
import numpy as np
import openml as openml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC

from mpl_toolkits.mplot3d import Axes3D


def neural_network():
    seed = 42
    np.random.seed(seed)
    dataset_id = 31

    # load data
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )

    # preprocessing
    label_encoder = LabelEncoder()
    enc = OneHotEncoder(handle_unknown='ignore')

    y = label_encoder.fit_transform(y)
    encoder = enc.fit(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    X_train = enc.transform(X_train)
    X_test = enc.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=seed,
                                                      stratify=y_train)

    y_hat = SVC(random_state=seed).fit(X_train, y_train).predict(X_val)
    print(f'Default hp accuracy {accuracy_score(y_val, y_hat)}')

    # SMBO
    # initial random hps
    low = 0.01
    high = 2.0
    hp_configuration = np.random.uniform(low, high, size=(5, 2))

    responses = []
    for c, gamma in hp_configuration:
        y_hat = SVC(random_state=seed, C=c, gamma=gamma).fit(X_train, y_train).predict(X_val)
        responses.append(accuracy_score(y_val, y_hat))
    responses = np.array(responses)

    surrogate = GaussianProcessRegressor()
    surrogate.fit(hp_configuration, responses)

    hyperparameter_values = np.array(list(itertools.product(np.linspace(low, high, 100), repeat=2)))
    initial_surrogate_response = surrogate.predict(hyperparameter_values)
    search_values = np.array(list(itertools.product(np.linspace(low, high, 1000), repeat=2)))

    beta = 0.1
    for i in range(50):
        means, stds = surrogate.predict(search_values, return_std=True)

        acq_values = means + beta * stds
        idx = np.argmax(acq_values)
        c, gamma = search_values[idx]
        y_hat = SVC(random_state=seed, C=c, gamma=gamma).fit(X_train, y_train).predict(X_val)
        responses = np.append(responses, accuracy_score(y_val, y_hat))
        hp_configuration = np.vstack((hp_configuration, [[c, gamma]]))
        surrogate.fit(hp_configuration, responses)

    final_surrogate_response = surrogate.predict(hyperparameter_values)

    final_idx = np.argmax(responses)
    print(f'best hp configuration: {hp_configuration[final_idx]} with performance{responses[final_idx]} ')

    c_hp = []
    gama_hp = []
    for hyperparameter_configuration in hyperparameter_values:
        c_hp.append(hyperparameter_configuration[0])
        gama_hp.append(hyperparameter_configuration[1])

    c_hp = np.asarray(c_hp)
    gama_hp = np.asarray(gama_hp)
    initial_surrogate_response = np.asarray(initial_surrogate_response)
    final_surrogate_response = np.asarray(final_surrogate_response)

    # plotting
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(c_hp, gama_hp, initial_surrogate_response)
    ax.set_title('Initial surrogate')
    ax.set_xlabel('C')
    ax.set_ylabel('Gama')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(c_hp, gama_hp, final_surrogate_response)
    ax.set_title('Final surrogate')
    ax.set_xlabel('C')
    ax.set_ylabel('Gama')
    plt.show()


if __name__ == '__main__':
    neural_network()
