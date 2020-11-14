from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import pandas as pd
import pickle


def load_data(label):
    df = pd.read_pickle(r'./static/Data/dataframe.pickle')
    print(df.columns.get_loc('new_cases'))
    y = df['new_cases']
    X = df.drop(columns=['new_cases'])
    if label:
        return X, y
    else:
        return X


def train_xgb():
    X, y = load_data(label=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=151)
    d_mat_train = xgb.DMatrix(X_train, y_train)
    params = {'eta': 0.1, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'objective': 'reg:squarederror',
              'max_depth': 5, 'alpha': 10, 'min_child_weight': 1}
    with open(r'./static/Models/model_params.pickle', 'wb') as write:
        pickle.dump(params, write, protocol=pickle.HIGHEST_PROTOCOL)
    booster = xgb.train(params, d_mat_train)
    d_mat_test = xgb.DMatrix(X_test)
    booster.save_model(r'./static/Models/xgb_model.json')

def train_svm():
    X, y = load_data(label=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=151)
    regressor = SVR()
    regressor.fit(X_train,y_train)
    with open(r'./static/Models/model_svm.pickle', 'wb') as write:
        pickle.dump(regressor, write, protocol=pickle.HIGHEST_PROTOCOL)

def train_rfr():
    X, y = load_data(label=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=151)
    regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)
    with open(r'./static/Models/model_rfr.pickle', 'wb') as write:
        pickle.dump(regressor, write, protocol=pickle.HIGHEST_PROTOCOL)


def train_dTree():
    X, y = load_data(label=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=151)
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)
    with open(r'./static/Models/model_dtree.pickle', 'wb') as write:
        pickle.dump(regressor, write, protocol=pickle.HIGHEST_PROTOCOL)

def train_knn():
    X, y = load_data(label=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=151)
    regressor = KNeighborsRegressor(n_neighbors=2)
    regressor.fit(X_train, y_train)
    with open(r'./static/Models/model_knn.pickle', 'wb') as write:
        pickle.dump(regressor, write, protocol=pickle.HIGHEST_PROTOCOL)


def test(Country,Model):
    preds = 0
    if Model.lower() == 'xgb':
        file = open(r'./static/Models/model_params.pickle', 'rb')
        params = pickle.load(file)
        file.close()
        bst = xgb.Booster(params)
        bst.load_model(r'./static/Models/xgb_model.json')
        index = get_index(Country)
        X = load_data(False)
        X_test = X[X.location == index].iloc[-1].values.reshape(1, -1)
        X_test = xgb.DMatrix(X_test)
        preds = bst.predict(X_test)[0]
    elif Model.lower() == 'svm':
        file = open(r'./static/Models/model_svm.pickle', 'rb')
        regressor = pickle.load(file)
        file.close()
        index = get_index(Country)
        X = load_data(False)
        X_test = X[X.location == index].iloc[-1].values.reshape(1, -1)
        preds = regressor.predict(X_test)[0]
    elif Model.lower() == 'knn':
        file = open(r'static\Models\model_knn.pickle', 'rb')
        regressor = pickle.load(file)
        file.close()
        index = get_index(Country)
        X = load_data(False)
        X_test = X[X.location == index].iloc[-1].values.reshape(1, -1)
        preds = regressor.predict(X_test)[0]
    elif Model.lower() == 'rfr':
        file = open(r'static\Models\model_rfr.pickle', 'rb')
        regressor = pickle.load(file)
        file.close()
        index = get_index(Country)
        X = load_data(False)
        X_test = X[X.location == index].iloc[-1].values.reshape(1, -1)
        preds = regressor.predict(X_test)[0]
    elif Model.lower() == 'dtree':
        file = open(r'static\Models\model_dtree.pickle', 'rb')
        regressor = pickle.load(file)
        file.close()
        index = get_index(Country)
        X = load_data(False)
        X_test = X[X.location == index].iloc[-1].values.reshape(1, -1)
        preds = regressor.predict(X_test)[0]
    return preds



def get_index(Country):
    Country = Country.lower()
    file = open('./static/Data/country_mapper.pickle', 'rb')
    mapper = pickle.load(file)
    file.close()
    return mapper[Country]



def main():
    # Uncomment following lines to train the models
    # train_svm()
    # train_knn()
    # train_rfr()
    # train_dTree()
    train_xgb()
    test('India','xgb')


if __name__ == '__main__':
    main()
