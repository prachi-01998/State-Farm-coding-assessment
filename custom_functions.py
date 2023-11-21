import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Function for feature engineering
def feature_engineering(train_data):
    '''This function takes care of feature engineering. It takes the data 
    as input and eliminates any unnecessary symbols and converting the 
        datatypes of a few columns as needed. Also gets rid of null values in the categorical datas In this function the dataset is 
    split into train, validation, and test datasets.'''

    train_val = train_data.copy(deep=True)

    # Fixing the money and percents
    train_val['x12'] = train_val['x12'].str.replace('$', '', regex = True)
    train_val['x12'] = train_val['x12'].str.replace(',', '')
    train_val['x12'] = train_val['x12'].str.replace(')', '', regex = True)
    train_val['x12'] = train_val['x12'].str.replace('(', '-', regex = True)
    train_val['x12'] = train_val['x12'].astype(float)
    train_val['x63'] = train_val['x63'].str.replace('%', '')
    train_val['x63'] = train_val['x63'].astype(float)
    
    # Handle missing values in categorical columns
    categorical_columns = train_val.select_dtypes(include=['object']).columns
    train_val[categorical_columns] = train_val[categorical_columns].fillna('Unknown')
    
    return train_val

# Function for mean imputation from the Train set
def mean_imputation(train_data, obj_cols):
    '''This function helps in dealing with missing values by 
    using the SimpleImputer class from the sklearn library. It also
    uses the StandardScaler class from sklearn library to standardize 
    the data.'''

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    train_imputed = pd.DataFrame(imputer.fit_transform(train_data.drop(columns=obj_cols)),
    columns=train_data.drop(columns=obj_cols).columns)
    std_scaler = StandardScaler()
    train_imputed_std = pd.DataFrame(std_scaler.fit_transform(train_imputed),columns=train_imputed.columns)
    
    return train_imputed_std


# Function to create dummy variables
def create_dummies(data, imputed_std, train = True):
    '''This function creates the one-hot encodings for the 
    categorical values using the pandas library.'''

    # Define all possible categories for columns
    all_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    all_countries = ['germany', 'asia', 'america', 'japan', 'nan']  # Add all possible countries
    all_months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    all_genders = ["Male", "Female"]  # Add all possible genders

    # Iterate over categorical columns
    for col in data.select_dtypes(include='object').columns:
        if col == 'x5':
            categories = all_days
        elif col == 'x31':
            categories = all_countries
        elif col == 'x81':
            categories = all_months
        elif col == 'x82':
            categories = all_genders
        else:
            categories = data[col].unique()

        # Ensure that the column is of categorical dtype
        data[col] = pd.Categorical(data[col], categories=categories, dtype='category')

        # Create dummy variables
        dummies = pd.get_dummies(data[col], prefix=col, prefix_sep='_', dummy_na=True)

        # Concatenate dummy variables with imputed_std
        imputed_std = pd.concat([imputed_std, dummies], axis=1, sort=False)

    if train:
        imputed_std = pd.concat([imputed_std, data['y']], axis=1, sort=False)

    return imputed_std
