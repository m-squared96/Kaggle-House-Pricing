#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr


class RandomForest(object):

    """
    Takes the train.csv data from the Kaggle Housing Prices competition and
    trains a random forest regression model, with a view to predicting values
    in the test dataset
    """

    def __init__(self):

        """
        Imports data and deals with missing values, correlation coefficients
        and dummy variables
        """

        data = pd.read_csv("train.csv")
        data = data.fillna(value=0)

        #data = data[data["SalePrice"] <= 400000]

        data["OverallComb"] = data["OverallQual"] * data["OverallCond"]

        remod_list = []

        for i,j in zip(data["YearBuilt"], data["YearRemodAdd"]):
            if i != j:
                remod_list.append(1)

            elif i == j:
                remod_list.append(0)

        data["RemodFlag"] = remod_list

        num_data = data.select_dtypes(include=[np.number])
        oth_data = data.drop(num_data.columns, axis=1)

        data_dummies = self.dummy_replace(oth_data)

        self.data = pd.concat([num_data, data_dummies], axis=1)
        print(self.data)
        
    def data_prepare(self):

        """
        Drops uncorrelated columns and calls the data_split method
        :param crit_score: minimum correlation coefficient between a given array and SalePrice
        """

        x = self.data.drop(["Id", "SalePrice"], axis=1)
        y = np.log(self.data["SalePrice"])

        self.data_split(x,y)

    def dummy_replace(self,data):

        """
        Replaces non-numeric columns with dummy variables
        :param data: Pandas.DataFrame
        """

        data_predummy = data.drop(data.select_dtypes(include=[np.number]).columns, axis=1)
        data_dummy = pd.get_dummies(data_predummy)

        data = pd.concat([data.drop(data_predummy.columns, axis=1),data_dummy], axis=1)

        return data

    def data_split(self,x,y):

        """
        Splits data into training and testing sets and scales
        :param x: Data excluding target variable
        :param y: Target variable
        """

        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(x,y,
                test_size=0.3, random_state=101)

        self.scaler = StandardScaler().fit(self.xtrain)
        self.xtrain_scaled = pd.DataFrame(self.scaler.transform(self.xtrain),
                index=self.xtrain.index.values, columns=self.xtrain.columns.values)

        self.xtest_scaled = pd.DataFrame(self.scaler.transform(self.xtest),
                index=self.xtest.index.values, columns=self.xtest.columns.values)

    def rfr_construct(self, est):

        """
        Constructs a random forest regression model with a given number of estimators
        :param estimators: n_estimators kwarg in the sklearn.ensemble.RandomForestRegressor
        """

        self.regressor = RandomForestRegressor(n_estimators=est, random_state=101, criterion="mse")
        self.regressor.fit(self.xtrain_scaled, self.ytrain)

        self.predictions = self.regressor.predict(self.xtest_scaled)

        self.test_score = r2_score(self.ytest, self.predictions)
        self.spearman = spearmanr(self.ytest, self.predictions)
        self.pearson = pearsonr(self.ytest, self.predictions)

        self.errors = 100*(np.exp(self.predictions) - np.exp(self.ytest))/np.exp(self.ytest)

        self.abs_errors = abs(self.errors)
        self.mean_error = self.abs_errors.mean()

        return self.test_score, self.mean_error

    def plotter(self):

        plt.figure()
        plt.scatter(np.exp(self.predictions), np.exp(self.ytest))
        plt.xlabel("Predicted Values")
        plt.ylabel("Actual Values")
        plt.title("Prediction vs Actual Values")

        plt.figure()
        plt.scatter(np.exp(self.predictions), np.exp(self.errors))
        plt.xlabel("Predicted Values")
        plt.ylabel("Difference wrt Actual")
        plt.title("Errors")

        plt.show()

    def importance(self):
        importances = tuple(self.regressor.feature_importances_)
        self.feature_weights = [(feature, round(importance, 5)) for feature, importance in zip(self.xtrain_scaled.columns, importances)]

        for pair in self.feature_weights:
            print(pair)

    def test_prepare(self):

        test = pd.read_csv("test.csv").fillna(value=0)
        test["OverallComb"] = test["OverallQual"] * test["OverallCond"]

        remod_list = []
        for i,j in zip(test["YearBuilt"], test["YearRemodAdd"]):
            if i != j:
                remod_list.append(1)

            elif i == j:
                remod_list.append(0)

        test["RemodFlag"] = remod_list

        num_data = test.select_dtypes(include=[np.number])
        oth_data = test.drop(num_data.columns, axis=1)

        data_dummies = self.dummy_replace(oth_data)
        test = pd.concat([num_data, data_dummies], axis=1)

        data_cols = self.data.drop("SalePrice", axis=1)

        for test_col in test.columns:
            if test_col not in data_cols:
                test.drop(test_col, axis=1)

        for train_col in data_cols:
            if train_col not in test.columns:
                test[train_col] = np.zeros((len(test["Id"]),1))

        self.test = test

    def predict(self):
        
        self.test_ids = self.test["Id"]

        self.test_scaler = StandardScaler().fit(self.test.drop("Id", axis=1))
        self.test_scaled = pd.DataFrame(self.test_scaler.transform(self.test.drop("Id", axis=1)),
                index=self.test.index.values, columns=self.test.drop("Id", axis=1).columns.values)

        if len(self.test_scaled.columns) == len(self.xtrain.columns):

            self.test_predictions = self.regressor.predict(self.test_scaled)
            submission_frame = pd.DataFrame([])
            submission_frame["Id"] = self.test_ids
            submission_frame["SalePrice"] = self.test_predictions

            submission_frame.to_csv("predicted_house_prices.csv", index=False)

        else:
            print("Submission document not outputted: columns do not match up")

            extra_column_list = []

            for col in self.test_scaled.columns:
                if col not in self.xtrain_scaled:
                    extra_column_list.append(col)

            print("Culprits:")
            print(extra_column_list)

            self.test_scaled = self.test_scaled.drop(extra_column_list, axis=1)
            print("Attempting again...")
            
            try:
                self.test_predictions = self.regressor.predict(self.test_scaled)
                submission_frame = pd.DataFrame([])
                submission_frame["Id"] = self.test_ids
                submission_frame["SalePrice"] = np.exp(self.test_predictions)

                submission_frame.to_csv("predicted_house_prices.csv", index=False)
                print("Output file produced")

            except:
                print("Failed again - aborting")



class ModelTest():

    """
    Generates and tests a series of RandomForest models and uses the most
    successful to produce submission file
    """

    def __init__(self):

        self.rf = RandomForest()

    def minfinder(self):

        model_num = int(input("Enter number of models to be tested:   "))
        estimators = np.linspace(500,2500,num=model_num)

        self.results = pd.DataFrame([],columns=["Estimators", "Test Score", "Mean Error"])
        count = 0

        print("\nInitialising regression models")

        self.rf.data_prepare()

        for i in estimators:
            
            test, merr = self.rf.rfr_construct(int(i))
            inter_results = pd.DataFrame([[i, test, merr]],
                    columns=["Estimators", "Test Score", "Mean Error"])

            self.results = pd.concat([self.results, inter_results], ignore_index=True)
            count += 1
            print("Progress:", str(count*100/model_num), "%")

        print(self.results)
        plot_option = str(input("Generate plot of errors?(y/n)")).lower()
        if plot_option in ("y", "yes", "ye", "ys"):
            self.errorplot()

    def errorplot(self):

        plt.figure()
        plt.plot(self.results["Estimators"], self.results["Test Score"], ls="--")
        plt.xlabel("Critical Correlation Score")
        plt.ylabel("Test Score")
        plt.title("Critical Correlation vs Test Score")

        plt.show(block=False)

    def outputter(self):

        best_est = self.results["Estimators"][self.results["Mean Error"].idxmin()]

        print("Best number of estimators:", str(int(best_est)))
        print("Best MAE:", str(self.results["Mean Error"].min()))

        self.rf.data_prepare()
        _,_ = self.rf.rfr_construct(int(best_est))

        self.rf.plotter()
        self.rf.importance()
        self.rf.test_prepare()
        self.rf.predict()

def main():
    tester = ModelTest()
    tester.minfinder()
    tester.outputter()

main()