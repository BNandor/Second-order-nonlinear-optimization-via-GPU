import math
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.losses import MeanAbsoluteError

def plot_history(history, key):
  plt.plot(history.history[key])
  plt.plot(history.history['val_'+key])
  plt.xlabel("Epochs")
  plt.ylabel(key)
  plt.legend([key, 'val_'+key])
  plt.show()

def scale_datasets(x_train, x_test,columnsToScale):
  """
  Standard Scale test and train data
  Z - Score normalization
  """
  scaling=columnsToScale
  if len(columnsToScale) == 0:
    scaling=x_train.columns
  standard_scaler = StandardScaler()
  x_train_scaled = pd.DataFrame(
      standard_scaler.fit_transform(x_train),
      columns=scaling
  )
  x_test_scaled = pd.DataFrame(
      standard_scaler.transform(x_test),
      columns=scaling
  )
  return x_train,x_test
  # return x_train_scaled, x_test_scaled

def trainANN(train_data, test_data,targetColumns,columnsToScale):
    x_train, y_train = train_data.drop(targetColumns, axis=1), train_data[targetColumns]
    x_test, y_test = test_data.drop(targetColumns, axis=1), test_data[targetColumns]
    x_train_scaled, x_test_scaled = scale_datasets(x_train, x_test,columnsToScale)
    hidden_units1 = 4
    hidden_units2 = 480
    hidden_units3 = 256
    learning_rate = 0.01
    model = Sequential([
        Dense(hidden_units1, kernel_initializer='normal', activation='relu'),
        Dropout(0.2),
        # Dense(hidden_units2, kernel_initializer='normal', activation='relu'),
        # Dropout(0.2),
        # Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
        # Dropout(0.2),
        Dense(len(targetColumns), kernel_initializer='normal', activation='linear')
    ])
    # loss function
    msle = MeanAbsoluteError()
    model.compile(
        loss=msle, 
        optimizer=Adam(learning_rate=learning_rate), 
        metrics=[msle]
    )
    # train the model
    history = model.fit(
        x_train_scaled.values,
        y_train.values,
        epochs=500,
        batch_size=50,
        validation_split=0.1
    )
    # plot_history(history, 'mean_squared_logarithmic_error')
    return (model,x_test_scaled,y_test)

# train,test=train_test_split(dataSet, test_size=trainTestRatio)
def testHousing():
    train_data = pd.read_csv("../samples/data/housing_train.csv")
    test_data = pd.read_csv("../samples/data/housing_test.csv")
    model,x_test_scaled,y_test=trainANN(train_data,test_data,["median_house_value"],[])
    x_test_scaled['prediction'] = model.predict(x_test_scaled)
    x_test_scaled['real']=y_test
    print(x_test_scaled[['prediction','real']])

# testHousing()