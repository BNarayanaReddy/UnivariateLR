import numpy as np
import argparse
import pandas as pd
import pickle


class UniLinRegression:
    def __init__(self, train_path, test_path):
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        train_headerss = train.keys()
        test_headerss = test.keys()
        x_train = train[train_headerss[0]]
        self.max_x = np.max(x_train)
        self.x_train = x_train / self.max_x
        y_train = train[train_headerss[1]]
        self.y_train = y_train / 1000
        x_test = test[test_headerss[0]]
        self.x_test = x_test / np.max(x_test)
        y_test = test[test_headerss[1]]
        self.y_test = y_test / 1000
        self.m = self.x_train.shape[0]

    def compute_cost(self, X, Y, w, b):
        """
        Mean Squarred Error loss is used
        """

        cost = 0
        m = X.shape[0]
        for i in range(m):
            loss = (self.predict(X[i], w, b) - Y[i]) ** 2
            cost += loss
        return cost / (2 * m)

    def predict(self, x, w, b):
        """
        Simple Linear function with a 2 trainable params
        """
        return w * x + b

    def compute_gradients(self, w, b):
        """
        Simpler math so need to do any bigger computations so no need for the
        backpropagation algorithm
        """
        dj_dw = 0
        dj_db = 0
        m = self.m
        for i in range(m):
            err = self.predict(self.x_train[i], w, b) - self.y_train[i]
            dj_dw_i = err * self.x_train[i]
            dj_db_i = err
            dj_dw += dj_dw_i
            dj_db += dj_db_i

        return dj_dw / m, dj_db / m

    def fit(self, lr, epochs):
        w = 0
        b = 0
        X = self.x_train
        Y = self.y_train
        self.params = []
        for i in range(epochs):
            cost = self.compute_cost(X, Y, w, b)
            gradients = self.compute_gradients(w, b)
            w = w - lr * gradients[0]
            b = b - lr * gradients[1]
            self.params.append((w, b))
            if i % 10 == 0:
                print(
                    "Epochs : ",
                    i,
                    "|",
                    "Loss : ",
                    cost,
                    "Gradients : ",
                    gradients,
                    "Params : ",
                    w,
                    b,
                )

    def test_accuracy(self):
        X_test = self.x_test
        Y_test = self.y_test
        w, b = min(self.params)
        model_params = (w, b)

        cost = self.compute_cost(X_test, Y_test, w, b)
        return cost, model_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
			Univariate Linear Regression model -
			Use csv with 2 cols : feature followed by label
			Ensure to include Area_sqft, Price as the headers or keys
			""",
    )

    parser.add_argument(
        "--train_csvfile",
        default="/home/narayana/learn/ml/lin_reg/data/train.csv",
        type=str,
        help="Pass path of the train csv file",
    )
    parser.add_argument(
        "--test_csvfile",
        default="/home/narayana/learn/ml/lin_reg/data/test.csv",
        type=str,
        help="Pass path of the test csv file",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.03,
        type=float,
        help="Learning rate to be utilized by the gradient descent algorithm implemented for model training",
    )
    parser.add_argument("--epochs", default=10000, type=int)
    args = parser.parse_args()
    model = UniLinRegression(args.train_csvfile, args.test_csvfile)
    model.fit(args.learning_rate, args.epochs)
    eval_loss, model_params = model.test_accuracy()
    data = {"Loss": eval_loss, "state_dict": model_params, "norm_x": model.max_x}
    # Assuming you have a trained model named 'model'
    with open("model.pkl", "wb") as file:
        pickle.dump(data, file)
