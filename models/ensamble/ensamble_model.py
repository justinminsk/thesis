import logging
import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

logging.basicConfig(filename='logs.txt', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

def get_data():
    ensamble_df = pd.read_parquet("ensamble_data.pq")

    iex_df = pd.read_parquet("iex_data/date_iex_data.parquet")

    ensamble_df = ensamble_df.rename(columns={"iex_pred_x": "wallstreet_pred","iex_pred_y":"iex_pred"})

    merged_df = pd.merge(ensamble_df, iex_df, on="date_col")

    merged_df = merged_df.set_index("date_col")

    scaler = MinMaxScaler(feature_range = (0, 1))
    scaled_df = scaler.fit_transform(merged_df)

    x_data = scaled_df[:,:3]
    y_data = scaled_df[:,-1]

    train_split = 0.8
    num_train = int(train_split * merged_df.shape[0])

    x_train = x_data[0:num_train]
    x_test = x_data[num_train:]

    y_train = y_data[0:num_train]
    y_test = y_data[num_train:]

    return x_train, y_train, x_test, y_test

def grid_svr(x_train, y_train):
    params = {"kernel": ["sigmoid"], "gamma": [0.001, 0.005, 0.0001, "auto"], "C": [0.001, 0.005, 0.0001]}
    grid_search_cv = GridSearchCV(SVR(), params, n_jobs=-1, verbose=1, cv=5)
    grid_search_cv.fit(x_train, y_train)
    logging.info("SVR Grid Search:")
    logging.info(grid_search_cv.best_estimator_)
    logging.info(" ")
    logging.info("Best Score:")
    logging.info(grid_search_cv.best_score_)
    logging.info(" ")

    svr_model = grid_search_cv.best_estimator_
    joblib.dump(svr_model, 'en_models/svr_model.joblib')

def grid_dtr(x_train, y_train):
    params = {"max_depth": [2, 3], "min_samples_split": [11, 12, 13], "min_samples_leaf": [1, 2]}
    grid_search_cv = GridSearchCV(DecisionTreeRegressor(), params, n_jobs=-1, verbose=1, cv=5)
    grid_search_cv.fit(x_train, y_train)
    logging.info("DecisionTreeRegressor Grid Search:")
    logging.info(grid_search_cv.best_estimator_)
    logging.info(" ")
    logging.info("Best Score:")
    logging.info(grid_search_cv.best_score_)
    logging.info(" ")

    dtr_model = grid_search_cv.best_estimator_
    joblib.dump(dtr_model, 'en_models/dtr_model.joblib')

def grid_rfr(x_train, y_train):
    params = {"n_estimators": [5, 10, 15], "max_depth": [2, 3], "min_samples_split": [9, 10, 11], "min_samples_leaf": [6, 7, 8]}
    grid_search_cv = GridSearchCV(RandomForestRegressor(), params, n_jobs=-1, verbose=1, cv=5)
    grid_search_cv.fit(x_train, y_train)
    logging.info("RandomForestRegressor Grid Search:")
    logging.info(grid_search_cv.best_estimator_)
    logging.info(" ")
    logging.info("Best Score:")
    logging.info(grid_search_cv.best_score_)
    logging.info(" ")

    rfr_model = grid_search_cv.best_estimator_
    joblib.dump(rfr_model, 'en_models/rfr_model.joblib')

def grid_gbr(x_train, y_train):
    params = {"learning_rate": [0.1, 0.01, 0.001], "n_estimators": [5, 10, 15, 50], "min_samples_split": list(range(5,11)), "min_samples_leaf": list(range(5,11)), "max_depth": list(range(2,6))}
    grid_search_cv = GridSearchCV(GradientBoostingRegressor(), params, n_jobs=-1, verbose=1, cv=5)
    grid_search_cv.fit(x_train, y_train)
    logging.info("GradientBoostingRegressor Grid Search:")
    logging.info(grid_search_cv.best_estimator_)
    logging.info(" ")
    logging.info("Best Score:")
    logging.info(grid_search_cv.best_score_)
    logging.info(" ")

    gbr_model = grid_search_cv.best_estimator_
    joblib.dump(gbr_model, 'en_models/gbr_model.joblib')


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = get_data()
    grid_svr(x_train, y_train)
    grid_dtr(x_train, y_train)
    grid_rfr(x_train, y_train)
    grid_gbr(x_train, y_train)
