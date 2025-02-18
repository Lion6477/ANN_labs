from keras.models import load_model
import numpy as np

def predict(x, y):
    model = load_model("saturn_model.keras")
    audit = np.array([[x, y]])
    audit_output = model.predict(audit)
    return np.argmax(audit_output)

if __name__ == '__main__':

    model = load_model("saturn_model.keras")

    print(f"Predict 0 =: {predict(10, 2)}")
    print(f"Predict 1 =: {predict(2, 0)}")
    print(f"Predict 2 =: {predict(17,0)}")


