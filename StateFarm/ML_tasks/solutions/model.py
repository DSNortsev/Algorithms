import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import pickle
from sklearn.model_selection import train_test_split
# import sys
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# import statsmodels.api as sm
# #loading visualization library
# import bokeh

import collections as ccc

# Declare file path
basedir = os.path.abspath(os.path.dirname(__file__))
train_data_dir = os.path.join(basedir, 'data', 'exercise_26_train.csv')
test_data_dir = os.path.join(basedir, 'data', 'exercise_26_test.csv')
models_dir = os.path.join(basedir, 'models')

# CATEGORICAL_FEATURES = {'y':[0 , 1],
#                         'x5': ['tuesday' 'saturday' 'thursday' 'sunday' 'wednesday' 'monday' 'friday'],
#                         'x31': ['germany' 'asia' 'america' 'japan', 'nan'],
#                         'x81': [], 'x82'}
CATEGORICAL_FEATURES = ['x5', 'x31', 'x81', 'x82']

TOTAL_FEATURES = 25

HIGH_RANKING_FEATURES = []

HIGH_RANKING_FEATURES = ['x5_saturday', 'x81_July', 'x81_December', 'x31_japan', 'x81_October', 'x5_sunday', 'x31_asia',
                         'x81_February', 'x91', 'x81_May', 'x5_monday', 'x81_September', 'x81_March', 'x53', 'x81_November',
                         'x44', 'x81_June', 'x12', 'x5_tuesday', 'x81_August', 'x81_January', 'x62', 'x31_germany',
                         'x58', 'x56']
DUMMY_CATEGORIES = {}

# Load dataset



def build_and_train():
    # Load CSV files
    raw_data = pd.read_csv(train_data_dir)
    # Normalize the data
    df_processed = pre_processing(raw_data, train=True)
    print(df_processed)
    # Create Logistic Regression model
    logit = sm.Logit(df_processed['y'], df_processed[HIGH_RANKING_FEATURES])
    # Fit the model
    result = logit.fit()
    result.summary()
    print(get_important_features(df_processed))

    # print(DUMMY_CATEGORIES)
    # pickle.dump(result, open(models_dir + '/logit.pkl', 'wb'))
    # model = pickle.load(open(models_dir + '/logit.pkl', 'rb'))
    # raw_data = pd.read_csv(test_data_dir)
    # raw_data = raw_data.iloc[:2]
    # print(raw_data[CATEGORICAL_FEATURES])
    # df_processed = pre_processing(raw_data)
    # # print(df_processed.filter(regex='x82_'))
    # # print(*df_processed.columns)
    # # print(CATEGORICAL_FEATURES)
    # print(df_processed[HIGH_RANKING_FEATURES])
    # print(model.predict(df_processed[HIGH_RANKING_FEATURES]))


def get_important_features(df):
    exploratory_lr = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
    exploratory_lr.fit(df.drop(columns=['y']), df['y'])
    exploratory_results = pd.DataFrame(df.drop(columns=['y']).columns).rename(columns={0: 'name'})
    exploratory_results['coefs'] = exploratory_lr.coef_[0]
    exploratory_results['coefs_squared'] = exploratory_results['coefs'] ** 2
    var_reduced = exploratory_results.nlargest(25, 'coefs_squared')
    return var_reduced['name'].to_list()



def pre_processing(df, train=False):
    # Use global variable when training the data
    if train:
        global DUMMY_CATEGORIES

    df_processed = df.copy(deep=True)

    # 1. Fixing the money and percents
    col_fix = {'x12': [('$', ''), (',', ''), (')', ''), ('(', '-')],
               'x63': [('%', '')]}

    for key, val in col_fix.items():
        for s_char in val:
            df_processed[key] = df_processed[key].str.replace(*s_char)
        df_processed[key] = df_processed[key].astype(float)

    # # Add categorical features to global variable with all trained values. Needed for One-hot encoding
    # for col in df_processed.columns[df_processed.dtypes == 'object']:
    #     CATEGORICAL_FEATURES[col] = df_processed[col].unique().tolist()
    # CATEGORICAL_FEATURES['y'] = df_processed['y'].unique().tolist()

    if train:
        # Add categorical features to global variable with all trained values. Needed for One-hot encoding
        # for col in df_processed.columns[df_processed.dtypes == 'object']:
        #     CATEGORICAL_FEATURES[col] = df_processed[col].unique().tolist()
        drop_features = CATEGORICAL_FEATURES + ['y']
    else:
        drop_features = CATEGORICAL_FEATURES
    print(drop_features)


    # 2. Replace NaN with mean and normalize the data
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df_processed.drop(columns=drop_features)),
                              columns=df_processed.drop(columns=drop_features).columns)
    std_scaler = StandardScaler()
    df_imputed_std = pd.DataFrame(std_scaler.fit_transform(df_imputed), columns=df_imputed.columns)

    # 3 Create dummies
    for col in CATEGORICAL_FEATURES:
        if train:
            dump = pd.get_dummies(df_processed[col], drop_first=True, prefix=col,
                              prefix_sep='_', dummy_na=True)
            DUMMY_CATEGORIES[col] = [col.split('_')[1] for col in dump.columns[:-1]]
        else:
            df_processed[col] = df_processed[col].astype(CategoricalDtype(DUMMY_CATEGORIES[col]))
            dump = pd.get_dummies(df_processed[col], prefix=col,
                              prefix_sep='_', dummy_na=True)
        print(dump.columns)
        df_imputed_std = pd.concat([df_imputed_std, dump], axis=1, sort=False)
    print(DUMMY_CATEGORIES)

    # Add y feature back to dataframe when training the model
    if train:
        df_imputed_std = pd.concat([df_imputed_std, df_processed.y], axis=1, sort=False)

    return df_imputed_std

build_and_train()
