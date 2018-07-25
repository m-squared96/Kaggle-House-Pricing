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

        self.data = pd.read_csv("train.csv")
        self.data = self.data.fillna(value=0)
        self.data = self.dummy_replace(self.data)
        
    def data_prepare(self, crit_score):

        """
        Drops uncorrelated columns and calls the data_split method
        :param crit_score: minimum correlation coefficient between a given array and SalePrice
        """

        x = self.data.drop("SalePrice", axis=1)
        y = self.data["SalePrice"]

        self.data_split(x,y)

    def dummy_replace(self,data):

        """
        Replaces non-numeric columns with dummy variables
        :param data: Pandas.DataFrame
        """

        data_predummy = data.drop(data.select_dtypes(include=[np.number]).columns, axis=1)
        data_dummy = pd.get_dummies(data_predummy)

        data = pd.concat([data.drop(data_predummy.columns, axis=1),
            data_dummy], axis=1)

        return data

    def data_split(self,x,y):

        """
        Splits data into training and testing sets and scales
        :param x: Data excluding target variable
        :param y: Target variable
        """

        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(x,y,
                test_size=0.3, random_state=101)

        scaler = StandardScaler().fit(self.xtrain.select_dtypes(include=[np.number]))

        self.xtrain_scaled = pd.DataFrame(scaler.transform(self.xtrain),
                index=self.xtrain.index.values, columns=self.xtrain.columns.values)

        self.xtest_scaled = pd.DataFrame(scaler.transform(self.xtest),
                index=self.xtest.index.values, columns=self.xtest.columns.values)

    def rfr_construct(self):

        """
        Constructs a random forest regression model with a given number of estimators
        :param estimators: n_estimators kwarg in the sklearn.ensemble.RandomForestRegressor
        """

        self.regressor = RandomForestRegressor(n_estimators=1000, random_state=101, criterion="mse")
        self.regressor.fit(self.xtrain_scaled, self.ytrain)

        self.predictions = self.regressor.predict(self.xtest_scaled)

        self.test_score = r2_score(self.ytest, self.predictions)
        self.spearman = spearmanr(self.ytest, self.predictions)
        self.pearson = pearsonr(self.ytest, self.predictions)

        self.errors = self.predictions - self.ytest

        self.abs_errors = abs(self.errors)
        self.mean_error = self.abs_errors.mean()

        return self.test_score, self.mean_error

    def plotter(self):

        plt.figure()
        plt.scatter(self.predictions, self.ytest)
        plt.xlabel("Predicted Values")
        plt.ylabel("Actual Values")
        plt.title("Prediction vs Actual Values")

        plt.figure()
        plt.scatter(self.predictions, self.errors)
        plt.xlabel("Predicted Values")
        plt.ylabel("Absolute Difference wrt Actual")
        plt.title("Errors")

        plt.show()

    def importance(self):
        importances = tuple(self.regressor.feature_importances_)
        self.feature_weights = [(feature, round(importance, 5)) for feature, importance in zip(self.xtrain_scaled.columns, importances)]

        for pair in self.feature_weights:
            print(pair)        


class ModelTest():

    """
    Generates and tests a series of RandomForest models and uses the most
    successful to produce submission file
    """

    def __init__(self):

        self.rf = RandomForest()

    def minfinder(self):

        corrnum = int(input("Number of R iterations:  "))
        corr_vals = np.linspace(0.00, 0.5, num=corrnum)

        total_models = corrnum
        print("Total number of models to test:", str(total_models))

        self.results = pd.DataFrame([],columns=["R", "Estimators", "Test Score", "Mean Error"])
        count = 0

        print("\nInitialising regression models")

        for i in corr_vals:

            self.rf.data_prepare(i)

            
            test, merr = self.rf.rfr_construct()
            inter_results = pd.DataFrame([[i, 1000, test, merr]],
                    columns=["R", "Estimators", "Test Score", "Mean Error"])

            self.results = pd.concat([self.results, inter_results], ignore_index=True)
            count += 1
            print("Progress:", str(count*100/total_models), "%")

        print(self.results)
        plot_option = str(input("Generate plot of errors?(y/n)")).lower()
        if plot_option in ("y", "yes", "ye", "ys"):
            self.errorplot()

    def errorplot(self):

        plt.figure()
        plt.plot(self.results["R"], self.results["Test Score"], ls="--")
        plt.xlabel("Critical Correlation Score")
        plt.ylabel("Test Score")
        plt.title("Critical Correlation vs Test Score")

        plt.figure()
        plt.plot(self.results["R"], self.results["Mean Error"], ls="--")
        plt.xlabel("Critical Correlation Score")
        plt.ylabel("Mean Absolute Error")
        plt.title("Critical Correlation vs MAE")

        plt.show(block=False)

    def outputter(self):

        best_r = self.results["R"][self.results["Mean Error"].idxmin()]
        best_est = self.results["Estimators"][self.results["Mean Error"].idxmin()]

        print("Best R value:", str(best_r))
        print("Best number of estimators:", str(int(best_est)))
        print("Best MAE:", str(self.results["Mean Error"].min()))

        self.rf.data_prepare(best_r)
        _,_ = self.rf.rfr_construct()

        #self.rf.predict_test()
        self.rf.plotter()
        self.rf.importance()

def main():
    tester = ModelTest()
    tester.minfinder()
    tester.outputter()

def test():
    rf = RandomForest()
    rf.data_prepare()

test()