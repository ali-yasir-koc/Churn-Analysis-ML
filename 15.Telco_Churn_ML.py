########################## TELCO CHURN ANALYSIS- MACHINE LEARNING ###########################
# This is a machine learning project.
# It consists of EDA, feature engineering and model parts.
# The aim is selection of the best algorithm for churn customer classification.
# Telco churn data includes information about a fictitious telecom company that provided 7043 California
# home phone and Internet services in the third quarter.
# Shows which install services you've ordered, gone or signed up for.


########################## Importing Library and Settings  ###########################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings

warnings.simplefilter(action="ignore")

path = "C:\\Users\\hseym\\OneDrive\\Masaüstü\\Miuul\\datasets"
os.chdir(path)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


########################## Loading  The Date  ###########################
def load_data(dataframe):
    return pd.read_csv(dataframe)


churn = load_data("TelcoCustomerChurn.csv")
df = churn.copy()
df.head()


########################## Summary of  The Data  ###########################
def columns_info(dataframe):
    columns, dtypes, unique, nunique, nulls = [], [], [], [], []

    for cols in dataframe.columns:
        columns.append(cols)
        dtypes.append(dataframe[cols].dtype)
        unique.append(dataframe[cols].unique())
        nunique.append(dataframe[cols].nunique())
        nulls.append(dataframe[cols].isnull().sum())

    return pd.DataFrame({"Columns": columns,
                         "Data_Type": dtypes,
                         "Unique_Values": unique,
                         "Number_of_Unique": nunique,
                         "Missing_Values": nulls})


columns_info(df)


########################## Structural Changes   ###########################
df.columns = [col.upper() for col in df.columns]
df.drop(columns = ["CUSTOMERID"], inplace = True)

df['TOTALCHARGES'] = pd.to_numeric(df['TOTALCHARGES'], errors = 'coerce')

df["CHURN"] = df["CHURN"].apply(lambda x: 1 if x == "Yes" else 0)


########################## Grab to Columns ###########################
def grab_col_names(dataframe, cat_th = 10, car_th = 20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

df[cat_cols].head()


########################## Outlier Analysis ###########################
def outlier_thresholds(dataframe, col_name, q1 = 0.10, q3 = 0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis = None):
        return True
    else:
        return False


for col in num_cols:
    print(check_outlier(df, col))


########################## Missing Value Analysis ###########################
df.isnull().sum()

df["TOTALCHARGES"].fillna(df[df["TOTALCHARGES"].isnull()]["MONTHLYCHARGES"], inplace = True)

df.isnull().sum()

df_temp = df.copy()



########################## Feature Engineering  ###########################
df.head()
""" 1 """
# Age of the customer in company
df["TENURE"].describe()
df["TENURE"].hist()
bins = [-1, 12, 24, 36, 48, 60, 72]
df["NEW_CUSTOMER_AGE_LEVEL"] = pd.cut(df["TENURE"], bins, labels = [1, 2, 3, 4, 5, 6])
df["NEW_CUSTOMER_AGE_LEVEL"] = df["NEW_CUSTOMER_AGE_LEVEL"].astype("int")

""" 2 """
# Monthly payment rate by customer age level
df["NEW_AGE_CHARGES_RATIO"] = df["NEW_CUSTOMER_AGE_LEVEL"] / df["MONTHLYCHARGES"]

""" 3 """
# Contract status of the elderly
SENIOR_CONTRACTS = pd.Series(["no_senior_monthly", "no_senior_yearly", "no_senior_two_year",
                              "senior_monthly", "senior_yearly", "senior_two_year"], dtype = "category")
df["NEW_SENIOR_CONTRACTS"] = SENIOR_CONTRACTS

df.loc[(df["SENIORCITIZEN"] == 0) & (df["CONTRACT"] == "Month-to-month"), "NEW_SENIOR_CONTRACTS"] = SENIOR_CONTRACTS[0]
df.loc[(df["SENIORCITIZEN"] == 0) & (df["CONTRACT"] == "One year"), "NEW_SENIOR_CONTRACTS"] = SENIOR_CONTRACTS[1]
df.loc[(df["SENIORCITIZEN"] == 0) & (df["CONTRACT"] == "Two year"), "NEW_SENIOR_CONTRACTS"] = SENIOR_CONTRACTS[2]
df.loc[(df["SENIORCITIZEN"] == 1) & (df["CONTRACT"] == "Month-to-month"), "NEW_SENIOR_CONTRACTS"] = SENIOR_CONTRACTS[3]
df.loc[(df["SENIORCITIZEN"] == 1) & (df["CONTRACT"] == "One year"), "NEW_SENIOR_CONTRACTS"] = SENIOR_CONTRACTS[4]
df.loc[(df["SENIORCITIZEN"] == 1) & (df["CONTRACT"] == "Two year"), "NEW_SENIOR_CONTRACTS"] = SENIOR_CONTRACTS[5]

""" 4 """
# Service columns were filled with 1 for yes and 0 for no and no service.
service_cols = ['MULTIPLELINES', 'ONLINESECURITY', 'ONLINEBACKUP', 'DEVICEPROTECTION',
                'TECHSUPPORT', 'STREAMINGTV', 'STREAMINGMOVIES', "PHONESERVICE"]

for col in service_cols:
    df["NEW_" + col] = np.where(df[col] == "Yes", 1, 0)

# Internet Service column was filled with 1 for fiber and DSL  and 0 for no.
df["NEW_INTERNETSERVICE"] = np.where(df["INTERNETSERVICE"] == "No", 0, 1)

""" 5 """
# Phone or internet or both
df["NEW_MAIN_SERVICE"] = np.where((df["NEW_PHONESERVICE"] == 1) & (df["NEW_INTERNETSERVICE"] == 0), 0,
                                  np.where((df["NEW_PHONESERVICE"] == 0) & (df["NEW_INTERNETSERVICE"] == 1), 1, 2))

""" 6 """
# Total number of Internet services utilized
df["NEW_NET_SERVICE_RATE"] = df["NEW_STREAMINGMOVIES"] + df["NEW_ONLINESECURITY"] + df["NEW_ONLINEBACKUP"] + \
                         df["NEW_DEVICEPROTECTION"] + df["NEW_TECHSUPPORT"] + df["NEW_STREAMINGTV"]

""" 7 """
# Stream services beneficiaries
df["NEW_STREAM_TOTAL"] = df["NEW_STREAMINGTV"] + df["NEW_STREAMINGMOVIES"]

""" 8 """
# Customers with a 1 or 2-year contract are identified as Engaged.
df["NEW_ENGAGED"] = df["CONTRACT"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

""" 9 """
# People who do not receive any support, backup or protection services
df["NEW_NO_SUPPORT"] = df.apply(lambda x: 1 if (x["ONLINEBACKUP"] != "Yes") or
                                               (x["DEVICEPROTECTION"] != "Yes") or
                                               (x["TECHSUPPORT"] != "Yes") else 0, axis = 1)

""" 10 """
# Young customers with a monthly contract
df["NEW_YOUNG_NOT_ENGAGED"] = df.apply(lambda x: 1 if (x["NEW_ENGAGED"] == 0) and
                                                      (x["SENIORCITIZEN"] == 0) else 0, axis = 1)

""" 11 """
# Does the person make automatic payments?
df["NEW_FLAG_AUTO-PAYMENT"] = df["PAYMENTMETHOD"].apply(lambda x: 1 if x in ["Bank transfer (automatic)",
                                                                             "Credit card (automatic)"] else 0)

""" 12 """
# Average Monthly Charges
df["AVG_CHARGES"] = df["TOTALCHARGES"] / (df["TENURE"])
df.loc[df["TENURE"] == 0, "AVG_CHARGES"] = 0

""" 13 """
# Raise Rate
df["RAISE"] = ((df["MONTHLYCHARGES"] - df["AVG_CHARGES"]) / df["AVG_CHARGES"]) * 100
df.loc[df["TENURE"] == 0, "RAISE"] = 0

""" 14 """
# Charge per service
df["NEW_AVG_SERVICE_FREE"] = df.apply(lambda x:
                                      (x["MONTHLYCHARGES"] / x['NEW_NET_SERVICE_RATE'])
                                      if x["NEW_NET_SERVICE_RATE"] != 0
                                      else 0, axis = 1)
""" 15 """
# Those who use paperless invoices and electronic checks
df["NEW_EP"] = df.apply(lambda x: 1 if (x["PAPERLESSBILLING"] == "Yes") &
                                       (x["PAYMENTMETHOD"] == "Electronic check ") else 0, axis = 1)

df.head()
df.drop(columns = ['MULTIPLELINES', 'ONLINESECURITY', 'ONLINEBACKUP', 'DEVICEPROTECTION',
                   'TECHSUPPORT', 'STREAMINGTV', 'STREAMINGMOVIES', "PHONESERVICE"], inplace = True)
df.head()


########################## Label Encoding  ###########################
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


bin_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype not in ["int64", "float64", "int32"]]
df[bin_cols].head()

for col in bin_cols:
    label_encoder(df, col)

df[bin_cols].head()

""" 16 """
# Extra human situation existing at home
df["NEW_EXTRA_PEOPLE"] = df["PARTNER"] + df["DEPENDENTS"]
df.head()


########################## Rare Encoding  ###########################
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end = "\n\n\n")


new_cat_col = [col for col in df.columns if df[col].dtype in ["category", "object"]]
rare_analyser(df, "CHURN", new_cat_col)


########################## One-Hot Encoding  ###########################
ohe_cols = new_cat_col.copy()

def one_hot_encoder(dataframe, categorical_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first)
    return dataframe


dff = one_hot_encoder(df, ohe_cols)
dff.head()
df.shape
dff.shape
dff.info()

dff.columns = [col.upper() for col in dff.columns]


########################## Standardization ###########################
standard_cols = [col for col in dff.columns if dff[col].dtype in ["float64"]]
dff[standard_cols].head()

ss = StandardScaler()
for col in standard_cols:
    dff[col] = ss.fit_transform(dff[[col]])
dff.head()


########################## Model Selection ###########################
y = dff["CHURN"]
X = dff.drop(["CHURN"], axis=1)

models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")


########################## Random Forests ###########################
rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_
rf_best_grid.best_score_


rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)


cv_results = cross_validate(rf_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()


##########################  GBoost  ###########################
xgboost_model = XGBClassifier(random_state=17)

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

best_xgbost = {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 500}
xgboost_final = xgboost_model.set_params(**best_xgbost, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()


##########################  LightGBM  ###########################
lgbm_model = LGBMClassifier(random_state=17)

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

best_lgbm = {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'n_estimators': 300}
lgbm_final = lgbm_model.set_params(**best_lgbm, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()


########################## CatBoost ###########################
catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

best_catboost = {'depth': 6, 'iterations': 500, 'learning_rate': 0.01}
catboost_final = catboost_model.set_params(**best_catboost, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()


########################### Feature Importance  ###########################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)
