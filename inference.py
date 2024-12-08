import pickle


def predict(x, w, b):
    y_hat = w * x + b
    return y_hat


with open("model.pkl", "rb") as file:
    model = pickle.load(file)
    params = model["state_dict"]
    eval_loss = model["Loss"]
    norm_x = model["norm_x"]
print(model, params, norm_x)


test_input = 2500
scaled_input = test_input / norm_x
predicted_price = predict(scaled_input, *params) * 10000
print("Expected Price : ", predicted_price)
