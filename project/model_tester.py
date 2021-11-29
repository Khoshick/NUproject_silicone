from tensorflow import keras
import pandas as pd
from joblib import load
df = pd.read_csv("preprocessed_df.csv")
df_silica_target=df['Silica_Concentrate']
df_silica_target = pd.DataFrame(data = df_silica_target,columns = ["Silica_Concentrate"])
df_dropped_test=df.drop(columns=['date','Silica_Concentrate'])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df_dropped_test,df_silica_target,test_size=0.2, random_state=17)  
ANN_model = keras.models.load_model("silicon_ySS.h5")
print(ANN_model.summary())
#loading standard scaler parameters
scaler = load("scaler.joblib")
#train data
x_train = scaler.transform(x_train)
y_pred = ANN_model.predict(x_train)
y_pred = pd.DataFrame(y_pred)
y_pred.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
x_train = pd.DataFrame(x_train)
x_train.reset_index(drop=True, inplace=True)
columns_results = list(df_dropped_test.columns)
columns_results.append("actual")
columns_results.append("predicted")
df_results = pd.concat([x_train,y_train,y_pred], axis = 1, ignore_index=True)
df_results.columns = columns_results
df_results.to_csv("df_train_results_with_unscaled_predictions_ss.csv", index = False)
#Test data
x_test = scaler.transform(x_test)
y_pred = ANN_model.predict(x_test)
x_test = pd.DataFrame(x_test)
x_test.reset_index(drop=True, inplace=True)
y_pred = pd.DataFrame(y_pred)
y_pred.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
columns_results = list(df_dropped_test.columns)
columns_results.append("actual")
columns_results.append("predicted")
df_results = pd.concat([x_test,y_test,y_pred], axis = 1)
df_results.columns = columns_results
df_results.to_csv("df_test_results_with_unscaled_predictions_ss.csv", index = False)