import itertools

import numpy as np
import openml as openml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC


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
    hyperparameter_values = np.array(list[itertools.product(np.linspace(low, high, 100), repeat=2)])
    initial_surrogate_response = surrogate.predict(hyperparameter_values)
    search_values = np.array(list(itertools.product(np.linspace(low, high, 1000), repeat=2)))

    beta = 0.1


if __name__ == '__main__':
    neural_network()
