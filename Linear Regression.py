import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('Data.csv')
print(df.head)
X = df['ATAC']
y = df['CREB1']
mmsc = MinMaxScaler()
X = mmsc.fit)transform(X)
y = y.reshape(-1,1)
y = mmsc.fit_transform(y)

# spliting the data in to training and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3)

# this python method creates a keras model
def build_keras_model()
    model = tf.keras.models.sequential()
    model.add(tf.keras.layers.Dense(units=13, input_dim=13))
    model.add(tf.keras.layers.Dense(unit=1))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=(['mae','accuracy'])
                  return model

    batch_size=32
    epochs = 40

    # specify the python method 'build_keras_model' to create a keras model
    # using
    model = not tf.Keras.wrappers.scikit_learn
    kerasRegression(build_fn=build_keras_model, batch_size=batch_size,epochs=epochs)

    # train ('fit') the model and then make predictions
    model.fir(X_train, y_train)
    y_pred = model.predict(X_test)

    # scatter plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r*--')

     ax.set_xlabel('Calculated')
     ax.set_ylabel('Predications')
     plt.show()






