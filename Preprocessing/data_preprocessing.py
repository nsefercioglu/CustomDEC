import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('/insurance_data.csv')
originaldf = df

df = df.drop(['Customer ID','Location','Behavioral Data','Purchase History','Insurance Products Owned','Risk Profile','Segmentation Group'], axis=1)


risk_mapping = {'Female': 0.45, 'Male': 0.55}
df['Gender'] = df['Gender'].map(risk_mapping)

df = pd.get_dummies(df, columns=['Marital Status'], drop_first=False)

df = pd.get_dummies(df, columns=['Occupation'], drop_first=False)

education_mapping = {
    'High School Diploma': 4,
    'Associate Degree': 3,
    "Bachelor's Degree": 2,
    "Master's Degree": 1,
    'Doctorate': 0
}
df['Education Level'] = df['Education Level'].map(education_mapping)

df = pd.get_dummies(df, columns=['Geographic Information'], drop_first=False)

df = pd.get_dummies(df, columns=['Interactions with Customer Service'], drop_first=False)

df = pd.get_dummies(df, columns=['Policy Type'], drop_first=False)

df = pd.get_dummies(df, columns=['Customer Preferences'], drop_first=False)

df = pd.get_dummies(df, columns=['Preferred Communication Channel'], drop_first=False)

df = pd.get_dummies(df, columns=['Preferred Contact Time'], drop_first=False)

df = pd.get_dummies(df, columns=['Preferred Language'], drop_first=False)

driving_record_mapping = {
    'Clean': 0,
    'Minor Violations': 1,
    'Accident': 2,
    'Major Violations': 3,
    'DUI': 4
}
df['Driving Record'] = df['Driving Record'].map(driving_record_mapping)

df = pd.get_dummies(df, columns=['Life Events'], drop_first=False)

df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
df['Policy Renewal Date'] = pd.to_datetime(df['Policy Renewal Date'])
df['Days with Insurance'] = (df['Policy Renewal Date'] - df['Policy Start Date']).dt.days
df = df.drop(['Policy Start Date', 'Policy Renewal Date'], axis=1)

scaler = MinMaxScaler()
minmax_collumns = ['Age', 'Gender', 'Income Level', 'Education Level', 'Claim History', 'Coverage Amount', 'Premium Amount', 'Deductible', 'Previous Claims History', 'Credit Score', 'Driving Record','Days with Insurance']
df[minmax_collumns] = scaler.fit_transform(df[minmax_collumns])

df.to_csv('/preprocessed_insurance_data.csv', index=False)
