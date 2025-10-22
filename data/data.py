import os

import pandas as pd
from scipy.io.arff import loadarff
from ucimlrepo import fetch_ucirepo

DATA_DIR = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
    "experiments",
    "data",
    "externals",
)


def load_pima_data():
    """
    Load PIMA diabates data set from data folder
    The name of the file shoulde be : diabetes.csv
    """
    filename = "diabetes.csv"
    try:
        df_diabete = pd.read_csv(os.path.join(DATA_DIR, filename))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"""Pima dataset not found. It must be downloaded and
                                placed in the folder {DATA_DIR} under the name {filename}"""
        )

    X_pima = df_diabete.drop(["Outcome"], axis=1).to_numpy()
    y_pima = df_diabete[["Outcome"]].to_numpy().ravel()  # be consistent with X
    return X_pima, y_pima


def load_phoneme_data():
    """
    Load Phoneme diabates data set from data folder
    The name of the file shoulde be : phoneme.csv
    """
    filename = "phoneme.csv"
    try:
        df_phoneme = pd.read_csv(
            os.path.join(DATA_DIR, filename),
            names=["Aa", " Ao", " Dcl", " Iy", " Sh", " Class"],
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"""Phoneme dataset not found. It must be downloaded and
                                placed in the folder {DATA_DIR} under the name {filename}"""
        )

    X_phoneme = df_phoneme.drop([" Class"], axis=1).to_numpy()
    y_phoneme = df_phoneme[[" Class"]].to_numpy().ravel()
    return X_phoneme, y_phoneme


def load_california_data():
    """
    Load California data set from data folder
    The name of the file shoulde be : california.arff
    """
    filename = "california.arff"
    raw_cal_housing = loadarff(os.path.join(DATA_DIR, filename))
    df_cal_housing = pd.DataFrame(raw_cal_housing[0])
    df_cal_housing.replace({"price": {b"True": 1, b"False": 0}}, inplace=True)
    X_cal_housing = df_cal_housing.drop(["price"], axis=1).to_numpy()
    y_cal_housing = df_cal_housing[["price"]].to_numpy().ravel()

    return X_cal_housing, y_cal_housing


def load_houses_sales_data():
    """
    Load House Sales of Léo Grinsztajn data set from data folder
    The name of the file shoulde be : house_sales_leo.arff
    """
    filename = "house_sales_leo.arff"
    raw_house_sales = loadarff(os.path.join(DATA_DIR, filename))
    df_house_sales = pd.DataFrame(raw_house_sales[0])
    # df_cal_housing.replace({"price": {b"True": 1, b"False": 0}}, inplace=True)
    X_house_sales = df_house_sales.drop(["price"], axis=1).to_numpy()
    y_house_sales = df_house_sales[["price"]].to_numpy().ravel()

    return X_house_sales, y_house_sales


def load_wine_data():
    """
    Load wine data set from ucimlrepo
    You should have installl ucimlrepo
    """
    # fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)
    X = wine_quality.data.features
    y = wine_quality.data.targets
    df_wine = pd.concat([X, y], axis=1)

    dict_mapping = {5: 0, 6: 0, 8: 1}
    df_wine = df_wine[df_wine["quality"].isin([5, 6, 8])].copy()
    df_wine.replace({"quality": dict_mapping}, inplace=True)
    X_wine = df_wine.drop(["quality"], axis=1).to_numpy()
    y_wine = df_wine[["quality"]].to_numpy().ravel()
    return X_wine, y_wine


def load_titatnic_data():
    """
    Load PIMA diabates data set from data folder
    The name of the file shoulde be : diabetes.csv
    """
    filename = "Titanic-Dataset.csv"
    try:
        df_titanic = pd.read_csv(os.path.join(DATA_DIR, filename))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"""Titanic dataset not found. It must be downloaded and
                                placed in the folder {DATA_DIR} under the name {filename}"""
        )
    df_titanic.dropna(subset=["Age"], inplace=True)  ## Dropped NA over Age
    dict_mapping_sex = {"male": 0, "female": 1}
    df_titanic = df_titanic.replace({"Sex": dict_mapping_sex})
    X_titanic = df_titanic.drop(
        ["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1
    ).to_numpy()
    y_titanic = df_titanic[["Survived"]].to_numpy().ravel()  # be consistent with X
    return X_titanic, y_titanic


def load_titatnic_benard_data():
    """
    Load PIMA diabates data set from data folder
    The name of the file shoulde be : diabetes.csv
    """
    filename = "titanic.csv"
    try:
        df_titanic = pd.read_csv(os.path.join(DATA_DIR, filename))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"""Titanic dataset not found. It must be downloaded and
                                placed in the folder {DATA_DIR} under the name {filename}"""
        )
    # df_titanic.dropna(subset=['Age'],inplace=True) ## Dropped NA over Age
    dict_mapping_sex = {"male": 0, "female": 1}
    df_titanic = df_titanic.replace({"Sex": dict_mapping_sex})
    X_titanic = df_titanic.drop(["Survived", "Name"], axis=1).to_numpy()
    y_titanic = df_titanic[["Survived"]].to_numpy().ravel()  # be consistent with X
    return X_titanic, y_titanic


def load_adult_data():
    """
    Load Adult data set from ucimlrepo
    You should have installl ucimlrepo
    """
    # fetch dataset
    adult = fetch_ucirepo(id=2)

    # data (as pandas dataframes)
    X = adult.data.features
    y = adult.data.targets
    df_adult = pd.concat([X, y], axis=1)

    dict_mapping = {">50K": 1, ">50K.": 1, "<=50K": 0, "<=50K.": 0}
    df_adult.replace({"income": dict_mapping}, inplace=True)
    dict_mapping_sex = {"Male": 0, "Female": 1}
    df_adult = df_adult.replace({"sex": dict_mapping_sex})
    df_adult.fillna("Unknow", inplace=True)
    X_adult = df_adult.drop(["income"], axis=1).to_numpy()
    y_adult = df_adult[["income"]].to_numpy().ravel()
    return X_adult, y_adult
