# import libraries
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import sklearn
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.metrics
import sklearn.svm
import sklearn.tree
import sklearn.neighbors

def pairPlot(heartData) :
    sns.pairplot(data=heartData, hue="DEATH_EVENT")
    plt.savefig("PairPlot.pdf")

def pca(heartData) :
    # Separate out the categorical labels column from the data
    heartLabels = heartData["DEATH_EVENT"]
    heartData = heartData.drop("DEATH_EVENT", axis=1)

    # Initialize a PCA object and specify that we want the transformed data to be 2D
    pca = sklearn.decomposition.PCA(n_components=2)

    # Perform the PCA transformation
    heartData_PCA = pca.fit_transform(heartData)

    # Plot a scatterplot of the result
    sns.scatterplot(x=heartData_PCA[:,0], y=heartData_PCA[:,1], hue=heartLabels)
    plt.savefig("PCAplot.pdf")

def swarmPlot(heartData) :
    # Draw a categorical scatterplot to show each observation
    ax = sns.swarmplot(data=heartData, x="serum_creatinine", y="serum_sodium", hue="DEATH_EVENT")
    ax.set(ylabel="")
    plt.savefig("OtherPlot.pdf")

heartData = pd.read_csv("heart_failure_clinical_records_dataset.csv")
#pairPlot(heartData)
#pca(heartData)
swarmPlot(heartData)