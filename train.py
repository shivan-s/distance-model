"""Train model."""

import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


def train_model(df):
    x_train, x_test, y_train, y_test = train_test_split(
        df["face area (pixel)"].values,
        df["distance (m)"].values,
        test_size=0.3,
        random_state=42,
    )

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train.reshape(-1, 1))
    x_test = scaler.transform(x_test.reshape(-1, 1))

    model = DecisionTreeRegressor()
    model.fit(x_train.reshape(-1, 1), y_train)
    return model


def save_model(model):
    pickle.dump(model, open("model.pkl", "wb"))
