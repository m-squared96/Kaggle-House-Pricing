#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    data = pd.read_csv("train.csv")

    data["OverallComb"] = data["OverallQual"] * data["OverallCond"]

    data["SalePrice_log"] = np.log(data["SalePrice"])

    corr_matrix = data.select_dtypes(include=[np.number]).drop(["Id"], axis=1).corr(method="pearson", min_periods=1)

    var_list = [
        "SalePrice",
        "OverallComb",
        "OverallQual",
        "OverallCond",
        "SalePrice_log"
    ]
    hue_var = None

    plotter(data, var_list, hue_var)
    heatmap(corr_matrix)
    price(data)


def plotter(data, var_list, hue):

    plt.figure()
    sns.pairplot(data, vars=var_list, hue=hue)

def heatmap(data):
    
    plt.figure()
    sns.heatmap(data, annot=False, cbar=True)

def price(data):
    plt.figure()
    sns.distplot(data["SalePrice"])

    plt.figure()
    sns.distplot(np.log(data["SalePrice"]))

main()
plt.show()