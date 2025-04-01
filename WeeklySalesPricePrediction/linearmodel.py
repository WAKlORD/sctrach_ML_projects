import numpy as np


def mse(input, predicted):
    return np.mean((input - predicted)**2)

class LinearModel:
    def __init__(self):
        self.Lmd = 0
        self.lrng_rate = 0

    @staticmethod
    def gradient_descent_lasso(Input, output, epochs: int, learning_rate: float, lambda_: float):
        n_samples, n_features = Input.shape
        W_g = np.zeros(n_features)
        bias = 0.0

        print("Current learning rate, lambda:", learning_rate, lambda_)

        for i in range(epochs):
            y_pred = Input @ W_g + bias
            error = mse(output, y_pred)

            gradient_w = (Input.T @ (y_pred - output)) / n_samples
            gradient_b = np.sum(y_pred - output) / n_samples

            lasso_term = lambda_ * np.sign(W_g)

            W_g -= learning_rate * (gradient_w + lasso_term)
            bias -= learning_rate * gradient_b

            if i % 1000 == 0:
                print(f"Epoch {i}, MSE: {error}")

        return W_g, bias

    def validation(self, Input, output, fold: int,Lambdas,lrng_rates):

        x_5 = np.array_split(Input, fold)
        y_5 = np.array_split(output, fold)

        best_score = (float('inf'), None, None)  # Initialize best score

        for Lambda in Lambdas:
            for Learning_rate in lrng_rates:
                for i in range(len(x_5)):
                    filtered_splits_x = [part for r, part in enumerate(x_5) if r != i]
                    x = np.concatenate(filtered_splits_x)
                    x_val = x_5[i]

                    filtered_splits_y = [part for r, part in enumerate(y_5) if r != i]
                    y = np.concatenate(filtered_splits_y)
                    y_val = y_5[i]

                    w_g, b_g = self.gradient_descent_lasso(x, y,10000,Learning_rate,Lambda)

                    y_pred = x_val @ w_g + b_g

                    mse_val = mse(y_val, y_pred)

                    print(f"The current MSE is: {mse_val}")

                    if mse_val < best_score[0]:
                        best_score = (mse_val, Lambda, Learning_rate)
                        self.Lmd = Lambda
                        self.lrng_rate = Learning_rate

        return self.Lmd,self.lrng_rate
        print(f"Best hyperparameters - Lambda: {self.Lmd}, Learning Rate: {self.lrng_rate}")





