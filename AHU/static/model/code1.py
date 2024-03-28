import numpy as np
from keras.models import load_model

model = load_model('model.h5')

inp = np.array([[50.3, 20.50, 20.0, 35.5, 40.0, 50.0, 80.4, 30.4, 22.0, 40.50, 90.0]])

y_pred_prob = model.predict(inp)

print(y_pred_prob)

