import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.preprocessing import StandardScaler 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time
tf.random.set_seed = 17
df = pd.read_csv("preprocessed_df.csv")
df_silica_target=df['Silica_Concentrate']
df_silica_target = pd.DataFrame(data = df_silica_target,columns = ["Silica_Concentrate"])
df_dropped_test=df.drop(columns=['date','Silica_Concentrate'])
x_train,x_test,y_train,y_test=train_test_split(df_dropped_test,df_silica_target,test_size=0.2, random_state=17)  
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
dump(scaler, "scaler.joblib")
print(x_train)
print(np.mean(x_train))
time.sleep(10)
optimizer=Adam(learning_rate=0.01)
ANN_model=keras.Sequential()
ANN_model.add(Dense(22,input_dim=22, kernel_initializer='normal',activation='relu'))
ANN_model.add(Dense(256,activation='relu')) 
ANN_model.add(Dense(256,activation='relu')) 
ANN_model.add(Dense(256,activation='relu')) 
ANN_model.add(Dense(256,activation='relu'))  
ANN_model.add(Dense(1,activation='linear'))
ANN_model.compile(loss='mse', optimizer='Adam', metrics = [keras.metrics.RootMeanSquaredError(name='rmse')])
model = ANN_model.fit(x_train,y_train,epochs=10)
result=ANN_model.evaluate(x_train,y_train)
print(f"Silicon model train result {result}")
ANN_model.save("silicon_ySS.h5")
model = keras.models.load_model("silicon_ySS.h5")
x_test2 = scaler.transform(x_test)
print("Loading model")
result=model.evaluate(x_test2,y_test)
print(result)
print("Ytest", y_test)