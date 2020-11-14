import pandas as pd
import sys
import os
import pickle
from sklearn.preprocessing import LabelEncoder
import urllib.request


def download_file():
    url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
    urllib.request.urlretrieve(url, 'model_data.csv')


def featurize():
    df = pd.read_csv('model_data.csv')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', None)
    missing_values = df.isnull().sum() / len(df)
    temp_df = pd.DataFrame({'column': df.columns, 'missing_values': missing_values})
    temp_df.sort_values('missing_values', inplace=True, ascending=False)
    cut_off_columns = temp_df[temp_df['missing_values'] > 0.50].index.tolist()
    df.drop(columns=cut_off_columns, inplace=True)
    iso_code = df[df['iso_code'].isna()]
    df.drop(index=iso_code.index, inplace=True)
    continent = df[df['continent'].isna()]
    df.drop(index=continent.index, inplace=True)

    # uncomment save df to csv for further analysis
    # analysts may be interested in pca et al

    # df.to_csv('.\Data\model_data.csv')

    # impute and encode categoric cols
    encoder = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).copy()
    countries = df['location'].str.lower()
    countries.fillna('missing',inplace=True)
    for col in cat_cols:
        df[col].fillna('missing', inplace=True)
        df[col] = encoder.fit_transform(df[col])
    country_mapper = dict(zip(countries,df['location']))
    with open('.\static\Data\country_mapper.pickle','wb') as write:
        pickle.dump(country_mapper,write,protocol=pickle.HIGHEST_PROTOCOL)
    # impute numeric cols
    num_cols = df.select_dtypes(include=['float64']).copy()
    for col in num_cols:
        df[col].fillna((df[col].mean()), inplace=True)

    df.to_pickle('.\static\Data\dataframe.pickle')


def main():
    password = input('Welcome admin! Please enter the password:\n')
    if password == '1234':
        action = input('Login successful!\nWhat would you like to refresh the db ? Y/N\n')
        if action.lower() == 'y':
            download_file()
            featurize()
            os.remove('model_data.csv')
            print("Database successfully refreshed and dataframe stored")
        else:
            print('Exiting system')
            sys.exit(0)
    else:
        print('Sorry wrong password, terminating program')
        sys.exit(1)


if __name__ == '__main__':
    main()
