import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from tensorflow import keras
from joblib import load
import seaborn as sns
from PIL import Image

st.title('Silica percent predictor')
st.markdown('The goal of this project is to remove the wastage of Iron ore by predicting the impurity in advance for the set mining process parameters')
st.subheader('Froth floatation Process schematic')
st.image("https://upload.wikimedia.org/wikipedia/commons/d/d0/Flotation_cell.jpg")
st.sidebar.header('Enter the set process parameters')
# df = pd.read_csv("preprocessed_df.csv")
# df_silica_target=df['Silica_Concentrate']
# df_silica_target = pd.DataFrame(data = df_silica_target,columns = ["Silica_Concentrate"])
# df_dropped_test=df.drop(columns=['date','Silica_Concentrate','Iron_Concentrate'])
# x_train,x_test,y_train,y_test=train_test_split(df_dropped_test,df_silica_target,test_size=0.2, random_state=17)  
with st.sidebar.form(key="form1"):
 Ironfeed = st.number_input('Percent Iron Feed', value=60.18)
 SilicaFeed = st.number_input('Percent Silica Feed', value=9.34)
 StarchFlow	= st.number_input('Starch Flow (m3/hr)', value=2206.01)
 AminaFlow	= st.number_input('Amina Flow (m3/hr)', value=387.953)
 OrePulpFlow = st.number_input('Ore Pulp Flow (m3/hr)', value=399.134)
 OrePulppH	= st.number_input('Ore Pulp pH', value=9.57526)
 OrePulpDensity = st.number_input('Ore Pulp Density (kg/cm3)', value=1.52685)
 aircol1= st.number_input('Air Column 1 (m3/hr)', value=199.695)
 aircol2= st.number_input('Air Column 2 (m3/hr)', value=194.925)
 aircol3= st.number_input('Air Column 3 (m3/hr)', value=200.282)
 aircol4= st.number_input('Air Column 4 (m3/hr)', value=295.096)
 aircol5= st.number_input('Air Column 5 (m3/hr)', value=306.4)
 aircol6= st.number_input('Air Column 6 (m3/hr)', value=249.893)
 aircol7= st.number_input('Air Column 7 (m3/hr)', value=251.256)
 FlotationColumn01Level= st.number_input('Flotation Column 01 Level (m3/hr)', value=851.7092)
 FlotationColumn02Level= st.number_input('Flotation Column 02 Level (m3/hr)', value=773.2705)
 FlotationColumn03Level= st.number_input('Flotation Column 03 Level (m3/hr)', value=882.909)
 FlotationColumn04Level= st.number_input('Flotation Column 04 Level (m3/hr)', value=444.242)
 FlotationColumn05Level= st.number_input('Flotation Column 05 Level (m3/hr)', value=450.434)
 FlotationColumn06Level= st.number_input('Flotation Column 06 Level (m3/hr)', value=483.979)
 FlotationColumn07Level= st.number_input('Flotation Column 07 Level (m3/hr)', value=438.374)
#  Ironconcentrate = st.number_input('Iron Concentrate', value=66.57)
#  yourexpectedvalue = st.number_input('Your Expected Value', value=1.6)
#  st.subheader("Scatter plot")
#  cols1 = ['date', 'Iron_feed', 'Silica_Feed', 'Starch_Flow', 'Amina_Flow', 'Ore_Pulp_Flow', 'Ore_Pulp_pH', 'Ore_Pulp_Density' , 'air_col1' , 'air_col2' , 'air_col3' , 'air_col4' , 'air_col5' , 'air_col6' , 'air_col7', 'Flotation_Column_01_Level', 'Flotation_Column_02_Level', 'Flotation_Column_03_Level', 'Flotation_Column_04_Level', 'Flotation_Column_05_Level', 'Flotation_Column_06_Level', 'Flotation_Column_07_Level', 'Iron_Concentrate', 'Silica_Concentrate'    ] # one or more
#  cols  = [ 'Iron_feed', 'Silica_Feed', 'Starch_Flow', 'Amina_Flow', 'Ore_Pulp_Flow', 'Ore_Pulp_pH', 'Ore_Pulp_Density' , 'air_col1' , 'air_col2' , 'air_col3' , 'air_col4' , 'air_col5' , 'air_col6' , 'air_col7', 'Flotation_Column_01_Level', 'Flotation_Column_02_Level', 'Flotation_Column_03_Level', 'Flotation_Column_04_Level', 'Flotation_Column_05_Level', 'Flotation_Column_06_Level', 'Flotation_Column_07_Level', 'Iron_Concentrate', 'Silica_Concentrate'    ]
#  selectbar_1 = st.selectbox(label='X axis',options=cols1)
#  selectbar_2 = st.selectbox(label='y axis',options=cols)
#  st.subheader("Histogram")
#  selectbar_3 = st.selectbox(label='column',options=cols)
 st.form_submit_button(label='Enter')
# st.set_option('deprecation.showPyplotGlobalUse', False)
# st.subheader("Scatter plot")
# sns.relplot(x=selectbar_1, y=selectbar_2, data=df)
# st.pyplot()
# st.subheader("Histogram")
# sns.histplot(df[selectbar_3])
# st.pyplot()
# st.subheader("Heat map")
# plt.figure(figsize=(50, 50))
# sns.heatmap(df.corr(), annot=True)
# st.pyplot()
input= pd.DataFrame(np.array([[Ironfeed,SilicaFeed,StarchFlow,AminaFlow,OrePulpFlow,OrePulppH,OrePulpDensity,aircol1,aircol2,aircol3,aircol4,aircol5,aircol6,aircol7,FlotationColumn01Level,FlotationColumn02Level,FlotationColumn03Level,FlotationColumn04Level,FlotationColumn05Level,FlotationColumn06Level,FlotationColumn07Level]]),columns=['Iron_feed','Silica_Feed','Starch_Flow','Amina_Flow','Ore_Pulp_Flow','Ore_Pulp_pH','Ore_Pulp_Density','air_col1','air_col2','air_col3','air_col4','air_col5','air_col6','air_col7','Flotation_Column_01_Level','Flotation_Column_02_Level','Flotation_Column_03_Level','Flotation_Column_04_Level','Flotation_Column_05_Level','Flotation_Column_06_Level','Flotation_Column_07_Level'])
# input = normalize(input) 
print("Formula Input", input)
scaler = load("scaler.joblib")
ANN_model = keras.models.load_model("silicon_ySS.h5")
input = scaler.transform(input)
predict=ANN_model.predict(input)
st.subheader('Silica Percent =')
st.markdown('Change the process parameters if the below value is greater than 2.5')
st.write(predict)

