from sklearn.datasets import load_iris
from sklearn import preprocessing
import uncertainties as unc
from lib import plots
import numpy as np
import sklearn
print(sklearn.__version__)

# Load Iris dataset
X, y = load_iris(return_X_y=True)
X_ori = preprocessing.StandardScaler().fit_transform(X)

#X=X[:,2:4]

for features in [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]:

    X=X_ori[:,features]
    
    
    # Plot dataset
    plots.plot_dataset(X, y, features)
    
    # Load Heatmap
    X_test = plots.load_Xtest(X)
    
    # # Compute margin-uncertainty based on Parzen Window
    # aleatoric, epistemic = unc.margin_uncertainties(X, y, X_test, model="ParzenWindow")
    # plots.plot_uncertainties(X_test, aleatoric, epistemic)
    
    # # Compute margin-uncertainty based on Decision Tree
    # aleatoric, epistemic = unc.margin_uncertainties(X, y, X_test, model="DecisionTree")
    # plots.plot_uncertainties(X_test, aleatoric, epistemic)
    
    # Compute margin-uncertainty based on Random Forest
    aleatoric, epistemic = unc.margin_uncertainties(X, y, X_test, model="RandomForest")
    plots.plot_uncertainties(X_test, aleatoric, epistemic, features)
    
    # # Compute margin-uncertainty based on K-Nearest Neighbors
    # aleatoric, epistemic = unc.margin_uncertainties(X, y, X_test, model="NearestNeighbors")
    # plots.plot_uncertainties(X_test, aleatoric, epistemic)
    
    # Plot dataset
    plots.plot_dataset(X, y, features)