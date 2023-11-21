import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import bokeh
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys


# Function to load data
def load_data(file_path):
    '''This function helps loading the data from the csv files.
    It takes the location of the train and the test files as inputs 
    and returns the raw train and test datasets'''
    data = pd.read_csv(file_path)
    
    return data


# Function to perform initial data exploration
def initial_data_exploration(data_train):
    '''This function performs the initial data exploration. 
    It also prints the versions of the different libraries 
    being used. It takes raw data as input.'''
    print("python version " + sys.version)
    print('numpy version ' + np.__version__)
    print('pandas version ' + pd.__version__)
    print('sklern version ' + sklearn.__version__)
    print('bokeh version ' + bokeh.__version__)
    print('statsmodels version ' + sm.__version__)
    print('matplotlib version ' + matplotlib.__version__)
    print('seaborn version ' + sns.__version__)

    # Describing the target variable
    print(Counter(data_train.y))

    # Overview of data types
    print("object dtype:", data_train.columns[data_train.dtypes == 'object'].tolist())
    print("int64 dtype:", data_train.columns[data_train.dtypes == 'int'].tolist())
    print("The rest of the columns have float64 dtypes.")


# Function to investigate object columns
def investigate_object(data):
    '''This function explores the columns with object type data in the 
    dataset. It is used to find the unique values in these columns'''    

    col_obj = data.columns[data.dtypes == 'object']

    for i in range(len(col_obj)):
        if len(data[col_obj[i]].unique()) > 13:
            print(col_obj[i] + ":", "Unique Values:", np.append(data[col_obj[i]].unique()[:13], "..."))
        else:
            print(col_obj[i] + ":", "Unique Values:", data[col_obj[i]].unique())

    del col_obj


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


def data_split(data):
    '''This function is used to split the data into the train, 
    validation and test datasets. It makes these datasets 
    from the train dataset that we have.'''
    # Creating train/val/test sets
    x_train, x_val, y_train, y_val = train_test_split(data.drop(columns=['y']), data['y'], test_size=0.1, random_state=13)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=4000, random_state=13)

    # Smashing sets back together
    train = pd.concat([x_train, y_train], axis=1, sort=False).reset_index(drop=True)
    val = pd.concat([x_val, y_val], axis=1, sort=False).reset_index(drop=True)
    test = pd.concat([x_test, y_test], axis=1, sort=False).reset_index(drop=True)

    return train, val, test


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


def visualize_data(train_data):
    '''This function is used to make a correlation heatmap. 
    If we see a few number of variables correlated with the target, 
    then we will use an L1 penalty. If we see a lot of variables 
    correlated with target then we will use an L2 penalty.'''

    sns.set(style='white')

    corr = train_data.corr()

    plt.figure(figsize=(12, 12))
    sns.set(font_scale=1)
    sns.heatmap(data=corr,
                center=0,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, linewidth=0.5)


# Function for exploratory logistic regression
def exploratory_logistic_regression(train_data):
    '''This function performs the exploratry logistic regression 
    with l1 penalty. Since the results from the correlation 
    indicated having fewer correlated variables, it seemed 
    appropriate to use the l1 penalty.'''

    exploratory_LR = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
    exploratory_LR.fit(train_data.drop(columns=['y']), train_data['y'])
    exploratory_results = pd.DataFrame(train_data.drop(columns=['y']).columns).rename(columns={0: 'name'})
    exploratory_results['coefs'] = exploratory_LR.coef_[0]
    exploratory_results['coefs_squared'] = exploratory_results['coefs'] ** 2
    var_reduced = exploratory_results.nlargest(25, 'coefs_squared')

    return var_reduced


# Function for preliminary logistic regression
def preliminary_logistic_regression(train_data, variables):
    '''The L1 process creates biased parameter estimates. As a result, 
    this function builds a final model without biased estimators.'''

    logit = sm.Logit(train_data['y'], train_data[variables])
    result = logit.fit()
    return result


# Function for evaluating the model
def evaluate_model(model, variables, imputed_std, target_col):
    '''This function computes the C-statistics for the dataset provided.'''
    #     Outcomes = pd.DataFrame(model.predict(imputed_std[variables])).rename(columns={0: 'probs'})
#     Outcomes['target'] = imputed_std[target_col]
#     C_Statistics = roc_auc_score(Outcomes['target'], Outcomes['probs'])
#     Outcomes['prob_bin'] = pd.qcut(Outcomes['probs'], q=20)
    probabilities = model.predict(imputed_std[variables])
    outcomes = pd.DataFrame(probabilities, columns=['probs'])
    outcomes['target'] = imputed_std[target_col]
    C_Statistics = roc_auc_score(outcomes['target'], outcomes['probs'])
    outcomes['prob_bin'] = pd.qcut(outcomes['probs'], q=20)

    return C_Statistics, outcomes

def probab_bins(Outcomes):
    '''This function shows the different probability bins'''

    print(Outcomes.groupby(['prob_bin'])['target'].sum())


# Function for saving the trained model
def save_model(model, file_path):
    '''This function helps save the trained model to the file path given.'''

    joblib.dump(model, file_path)


# Main function for data processing and model training
def main(train_file):
    # Load data
    raw_train = load_data(train_file)
    
    #getting the Categorical columns
    cols_with_obj_dtype = raw_train.columns[raw_train.dtypes == 'object'].tolist()
    cols_with_obj_dtype.append('y')
    Cate_cols = cols_with_obj_dtype

    # Initial data exploration
    initial_data_exploration(raw_train)

    # Feature engineering
    train_data = feature_engineering(raw_train)
    
    #splitting the data into train, validation, test dataset
    train, val, test = data_split(train_data)
    
    # Mean imputation
    train_imputed_std = mean_imputation(train, Cate_cols)

    # Create dummy variables
    data_train = create_dummies(train, train_imputed_std, train = True)
    
    # Data visualization
    visualize_data(data_train)

    # Train the Exploratory logistic regression model
    exploratory_variables = exploratory_logistic_regression(data_train)
    
    variables = exploratory_variables['name'].to_list()
    # Train preliminary logistic regression model
    result_preliminary = preliminary_logistic_regression(data_train,variables)

    # preparing the validation set
    val_imputed_std = mean_imputation(val, Cate_cols)
    val_imputed_std = create_dummies(val, val_imputed_std)

    # preparing the validation set
    test_imputed_std = mean_imputation(test, Cate_cols)
    test_imputed_std = create_dummies(test, test_imputed_std)

    # evaluate the models
    C_stat_train, Outcomes_train = evaluate_model(result_preliminary, variables, train_imputed_std, 'y')
    C_stat_val, Outcomes_val = evaluate_model(result_preliminary, variables, val_imputed_std, 'y')
    C_stat_test, Outcomes_test = evaluate_model(result_preliminary, variables, test_imputed_std, 'y')
    print("Train C-Statistics: ", C_stat_train, "Validation C-Statistics: ", C_stat_val, "Test C-Statistics: ", C_stat_test)

    # Finalized training model
    train_and_val = pd.concat([train_imputed_std, val_imputed_std])
    all_train = pd.concat([train_and_val, test_imputed_std])
    final_model = preliminary_logistic_regression(all_train, variables)

    C_Stat_final_train, Outcomes_final = evaluate_model(final_model, variables, all_train, 'y')

    # The probability bins
    probab_bins(Outcomes_final)

    # Save the trained model to a file
    save_model(final_model, model_save_path)


if __name__ == "__main__":
    train_file_path = r"C:\Users\prach\Downloads\MLE 26\exercise_26_train.csv"
    #test_file_path = 'exercise_26_test.csv'
    model_save_path = 'trained_model_2.joblib'
    main(train_file_path)
