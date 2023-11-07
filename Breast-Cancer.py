import streamlit as st
import pandas as pd
import joblib as jb
import time


parameter_list=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
parameter_input_values=[]
parameter_description=['Size of tumor', 'Surface Roughness', 'Tumor Boundary Length ', 'Tumor Area','Smoothness of Tumor ','Compactness of Tumor','Concavity of Tumor', 'Points of Concavity','Symmetry of Tumor ', 'Fractal Dimension']
parameter_default_values=[17.99, 10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471, 0.2419,0.07871]

with st.spinner('Fetching Latest ML Model'):
    # Use pickle to load in the pre-trained model
    model = joblib.load("Magic Curve Model.pkl")
    time.sleep(1)
    st.success('Model V8 Loaded!')


st.title('My Breast Cancer Prediction App \n\n')

for parameter,parameter_df,parameter_desc in zip(parameter_list,parameter_default_values,parameter_description):
    #print (parameter,parameter_df,parameter_desc)
    st.subheader('Input value for '+parameter)
    parameter_input_values.append(st.number_input(parameter_desc,key=parameter,value=float(parameter_df)))
        
parameter_dict=dict(zip(parameter_list, parameter_input_values)) 

st.write('\n','\n')
st.title('Your Input Summary')

st.write(parameter_dict)

st.write('\n','\n')

def predict(input_predict):
    values = input_predict['data'] 

    input_variables = pd.DataFrame([values],
                                columns=parameter_list, 
                                dtype=float,
                                index=['input'])    
    
    # Get the model's prediction
    prediction = model.predict(input_variables)
    print("Prediction: ", prediction)
    prediction_proba = model.predict_proba(input_variables)[0][1]
    print("Probabilities: ", prediction_proba)

    ret = {"prediction":float(prediction),"prediction_proba": float(prediction_proba)}
    
    return ret

if st.button("Click Here to Predict"):

    PARAMS={'data':list(parameter_dict.values())}
    
    r=predict(PARAMS)
    
    st.write('\n','\n')
    
    prediction_proba=r.get('prediction_proba')
    prediction_proba_format = str(round(float(r.get('prediction_proba')),1)*100)+'%'
    
    prediction_value=r.get('prediction')
    
    prediction_bool='Positive' if float(prediction_proba) > 0.4 else 'Negative'
    
    st.write(f'Your Breast Cancer Prediction is: **{prediction_bool}** with **{prediction_proba_format}** confidence')
