import logging
import os
import datetime

import dill
import json

from os import listdir
from os.path import isfile, join

import pandas as pd

path = os.environ.get('PROJECT_PATH', '..')


def get_test_files() -> pd.DataFrame:
    test_folder_path = path + '/data/test'
    test_files = [f for f in listdir(test_folder_path) if isfile(join(test_folder_path, f))]
    lst = []
    for file_name in test_files:
        with open(test_folder_path + '/' + file_name, "r") as read_file:
            lst.append(json.load(read_file))

    return pd.DataFrame(lst)


def predict():
    model_filename = f'{path}/data/models/cars_pipe_{datetime.datetime(2022, 8, 27, 15, 17).strftime("%Y%m%d%H%M")}.pkl'

    with open(model_filename, 'rb') as file:
        model = dill.load(file)

    df_test = get_test_files()
    predicts = model.predict(df_test)
    df_test['price_category'] = predicts
    df_test.to_csv(path + '/data/predictions/predictions.csv', index=False)


if __name__ == '__main__':
    predict()

