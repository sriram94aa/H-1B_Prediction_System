import streamlit as st
import pandas as pd
from sklearn.ensemble  import GradientBoostingClassifier
import datetime
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline


st.write("""# H-1B Prediction""")
st.header('User Input Values')


def user_input_features():
     
    # Agent Representing Employer
    agent = st.selectbox('Is Agent representing your employer', ('Yes', 'No'))
    if agent == 'Yes':
        agent = 1
    else:
        agent = 0
    # Total Workers
    tot_Work = st.text_input('Enter the Total number of Workers', 0)

    # Full Time Position
    fullTime = st.radio('Are you Employed Full-Time?', ('Yes', 'No'))
    if fullTime == 'Yes':
        fullTime = 1
    else:
        fullTime = 0
    # Wage
    wage = st.text_input('Enter your Annual Income', 0)
    # H1b Dependant
    dep = st.radio('Do you have a Dependent?', ('Yes', 'No'))
    if dep == 'Yes':
        dep = 1
    else:
        dep = 0
    # Violator
    vio = st.radio('Have you violated any rules?', ('Yes', 'No'))
    if vio == 'Yes':
        vio = 1
    else:
        vio = 0
    # Employment Start date
    start = st.date_input("Enter Employment Start Date:" , datetime.date(2021, 4, 27))
    
    # Employment End date
    end = st.date_input("Enter Employment End Date:" , datetime.date(2021, 4, 27))
    
    days = abs(end - start).days
    
    #Occupation
    occ = st.selectbox('Please select your Occupation', ('Computer Occupations','Others','Architecture & Engineering','Financial Occupation','Management Occupation','Medical Occupations','Education Occupations','Advance Sciences','Business Occupation','Mathematical Occupations','Marketing Occupation'))
    if occ == 'Computer Occupations':
        occ = 0.961930
    elif occ == 'Others':
        occ = 0.939235
    elif occ == 'Architecture & Engineering':
        occ = 0.951285
    elif occ == 'Financial Occupation':
        occ = 0.950806
    elif occ == 'Management Occupation':
        occ = 0.958884
    elif occ == 'Medical Occupations':
        occ = 0.942589
    elif occ == 'Education Occupations':
        occ = 0.943956
    elif occ == 'Advance Sciences':
        occ = 0.954090
    elif occ == 'Business Occupation':
        occ = 0.937729
    elif occ == 'Mathematical Occupations':
        occ = 0.966245
    elif occ == 'Marketing Occupation':
        occ = 0.857143
     
    data = {'AGENT_REPRESENTING_EMPLOYER': agent,	
            'TOTAL_WORKERS': tot_Work,	
            'FULL_TIME_POSITION': fullTime,
            'PREVAILING_WAGE':wage,
            'H1B_DEPENDENT': dep,
            'WILLFUL_VIOLATOR':vio,
            'Total_Days_Of_Employment': days,
            'Occ_Encoded':occ}
    features = pd.DataFrame(data, index=[0])
    return features


def pred(new_df):
        df_train = pd.read_csv('StreamlitCSV.csv')
        
        X = df_train.drop('CASE_STATUS', axis=1)
        y = df_train['CASE_STATUS']
        
        X_train,X_test,y_train,y_test = train_test_split(X ,y ,test_size = 0.30,random_state=1)
        
        resample = SMOTEENN()
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
        pipeline = Pipeline(steps=[('r', resample), ('m', model)])
        pipeline.fit(X_train, y_train)
        final = pipeline.predict(new_df)
        st.header('Anticipated H1-B Status')
        if final[0] == 1:
            st.image('accepted.jpg', width=100)
        else:
            st.image('reject.jpg', width=150)
df = user_input_features()

if st.button('Submit'):
    pred(df)


