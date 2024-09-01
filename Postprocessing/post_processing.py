import pandas as pd

df = pd.read_csv('/clustered.csv')
df = df.drop(['Geographic Information_Andaman and Nicobar Islands',
'Geographic Information_Andhra Pradesh',
'Geographic Information_Arunachal Pradesh',
'Geographic Information_Assam', 'Geographic Information_Bihar',
'Geographic Information_Chandigarh',
'Geographic Information_Chhattisgarh',
'Geographic Information_Dadra and Nagar Haveli',
'Geographic Information_Daman and Diu', 'Geographic Information_Delhi',
'Geographic Information_Goa', 'Geographic Information_Gujarat',
'Geographic Information_Haryana',
'Geographic Information_Himachal Pradesh',
'Geographic Information_Jharkhand', 'Geographic Information_Karnataka',
'Geographic Information_Kerala', 'Geographic Information_Lakshadweep',
'Geographic Information_Madhya Pradesh',
'Geographic Information_Maharashtra', 'Geographic Information_Manipur',
'Geographic Information_Meghalaya', 'Geographic Information_Mizoram',
'Geographic Information_Nagaland', 'Geographic Information_Odisha',
'Geographic Information_Puducherry', 'Geographic Information_Punjab',
'Geographic Information_Rajasthan', 'Geographic Information_Sikkim',
'Geographic Information_Tamil Nadu', 'Geographic Information_Telangana',
'Geographic Information_Tripura',
'Geographic Information_Uttar Pradesh',
'Geographic Information_Uttarakhand',
'Geographic Information_West Bengal',
'Interactions with Customer Service_Chat',
'Interactions with Customer Service_Email',
'Interactions with Customer Service_In-Person',
'Interactions with Customer Service_Mobile App',
'Interactions with Customer Service_Phone','Policy Type_Business',
'Policy Type_Family', 'Policy Type_Group', 'Policy Type_Individual',
'Customer Preferences_Email', 'Customer Preferences_In-Person Meeting',
'Customer Preferences_Mail', 'Customer Preferences_Phone',
'Customer Preferences_Text', 'Preferred Communication Channel_Email',
'Preferred Communication Channel_In-Person Meeting',
'Preferred Communication Channel_Mail',
'Preferred Communication Channel_Phone',
'Preferred Communication Channel_Text',
'Preferred Contact Time_Afternoon', 'Preferred Contact Time_Anytime',
'Preferred Contact Time_Evening', 'Preferred Contact Time_Morning',
'Preferred Contact Time_Weekends', 'Preferred Language_English',
'Preferred Language_French', 'Preferred Language_German',
'Preferred Language_Mandarin', 'Preferred Language_Spanish',
'Life Events_Childbirth', 'Life Events_Divorce',
'Life Events_Job Change', 'Life Events_Marriage',
'Life Events_Retirement','Marital Status_Divorced', 'Marital Status_Married',
'Marital Status_Separated', 'Marital Status_Single',
'Marital Status_Widowed', 'Occupation_Artist', 'Occupation_Doctor',
'Occupation_Engineer', 'Occupation_Entrepreneur', 'Occupation_Lawyer',
'Occupation_Manager', 'Occupation_Nurse', 'Occupation_Salesperson',
'Occupation_Teacher','Age', 'Gender', 'Income Level', 'Education Level','Credit Score','Claim History'], axis=1)

#'Age', 'Gender', 'Income Level', 'Education Level','Credit Score'

def calculate_iqr(x):
    Q1 = x.quantile(0.40)
    Q3 = x.quantile(0.60)
    IQR = Q3 - Q1
    return Q1, Q3, IQR

# Function to filter data points within the IQR range
def filter_iqr(df):
    for column in df.columns:
        if column != 'Cluster':
            Q1, Q3, IQR = calculate_iqr(df[column])
            df = df[(df[column] >= Q1) & (df[column] <= Q3)]
    return df

# Filter data points within the IQR range for each cluster
filtered_df = df.groupby('Cluster').apply(filter_iqr).reset_index(drop=True)

# Calculate mean for each cluster on the filtered data
mean_stats = filtered_df.groupby('Cluster').mean()

# Rename the index using the dictionary
cluster_names = {
    0: 'Cluster 0',
    1: 'Cluster 1',
    2: 'Cluster 2',
    3: 'Cluster 3',
    4: 'Cluster 4'
}
mean_stats.rename(index=cluster_names, inplace=True)

# Save the result to a CSV file
mean_stats.to_csv('/cluster_stats.csv', index=True)