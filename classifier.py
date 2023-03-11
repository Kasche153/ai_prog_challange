import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, chi2
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score, cross_validate ,ShuffleSplit
import numpy
 



dataframe = pd.read_csv("output.csv")

file = open("evaluation.txt", "w")

test = pd.read_csv("EvaluateOnMe-6.csv")
le = LabelEncoder()

# dataframe["x6"] = le.fit_transform(dataframe.x6.values)

#print(dataframe.to_string())


# for col in dataframe:

#     if(dataframe[col].dtype == "float64"):
#         print("prev", dataframe[col].max())
#         l_q = dataframe[col].quantile(0.01)
#         h_q = dataframe[col].quantile(0.99)
        
#         print("after", dataframe[col].max())


        


# print(dataframe.to_string())

def clean_data(data: pd.DataFrame):
    
    for col in data:
        print(data[col].dtype)

    return data


def cap_data(df):
    for col in df.columns:
        print("capping the ",col)
        if (((df[col].dtype)=='float64') | ((df[col].dtype)=='int64')):
            percentiles = df[col].quantile([0.01,0.99]).values
            df[col][df[col] <= percentiles[0]] = percentiles[0]
            df[col][df[col] >= percentiles[1]] = percentiles[1]
        else:
            df[col]=df[col]
    return df


clean_data(dataframe)


def process_data(data: pd.DataFrame):

    for col in data:
        print(data[col].dtype)
    # x.drop("x5", axis=1,inplace=True)
    data["x1"].astype("float")
    # data["x11"].replace("F", "False",inplace=True)
    # data["x12"].replace("F", "False", inplace=True)
    # data["x11"].replace("Tru", "True", inplace=True)
    # data["x12"].replace("Flase", "False", inplace=True)
    data.drop("x3", inplace=True, axis=1)
    data["x6"] = le.fit_transform(data["x6"].values)
    data["x11"] = le.fit_transform(data["x11"].values)
    data["x12"] = le.fit_transform(data["x12"].values)
    return data


y = dataframe["y"]
x = dataframe.drop('y', inplace=False, axis=1)



x = cap_data(x)

x = process_data(x)
# x.to_csv("test.csv", encoding='utf-8')

# print(x["x12"].to_string())
# dataframe["x1"].astype("float")



#x.drop("x5", axis=1,inplace=True)


# print(set(x["x6"].values.tolist()))
print(set(x["x11"].values.tolist()))
print(set(x["x12"].values.tolist()))





# print(x.to_string())

#print(x.to_string())

k_folds = KFold(n_splits = 2)


# classifier = DecisionTreeClassifier()
#classifier = BaggingClassifier()
#classifier = GaussianNB()
#classifier = LogisticRegression()
classifier = KNeighborsClassifier()
# classifier = RandomForestClassifier(n_estimators=100)
# classifier = AdaBoostClassifier()



test = process_data(test)
classifier.fit(x, y)

predictions =  classifier.predict(test)

for pred in predictions:
    file.write(pred + "\n")
    
scores = cross_val_score(classifier, x, y, cv=k_folds)
print(scores.mean(), scores.std())
