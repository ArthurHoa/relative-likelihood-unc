# Relative-likelihood based uncertainty quantification

Relative likelihood-based epistemic and aleatoric uncertainty decomposition for both binary and multi-class classification.

## Example

```
# Load Iris dataset
X, y = load_iris(return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)
X=X[:,2:4]

# Plot dataset
plots.plot_dataset(X, y)

# Load Heatmap
X_test = plots.load_Xtest(X)

# Compute margin-uncertainty
aleatoric, epistemic = unc.margin_uncertainties(X, y, X_test)
plots.plot_uncertainties(X_test, aleatoric, epistemic)
```

<img src="images/dataset_iris.png" alt="dataset" width="200">

### Parzen Window

```
aleatoric, epistemic = unc.margin_uncertainties(X, y, X_test, model="ParzenWindow")
plots.plot_uncertainties(X_test, aleatoric, epistemic)
```

<img src="images/al_parzen.png" alt="AU_parzen" width="200"><img src="images/ep_parzen.png" alt="EU_parzen" width="200">

### Decision Tree

```
aleatoric, epistemic = unc.margin_uncertainties(X, y, X_test, model="DecisionTree")
plots.plot_uncertainties(X_test, aleatoric, epistemic)
```

<img src="images/al_tree.png" alt="AU_tree" width="200"><img src="images/ep_tree.png" alt="EU_tree" width="200">

### Random Forest

```
aleatoric, epistemic = unc.margin_uncertainties(X, y, X_test, model="RandomForest")
plots.plot_uncertainties(X_test, aleatoric, epistemic)
```

<img src="images/al_rf.png" alt="AU_rf" width="200"><img src="images/ep_rf.png" alt="EU_rf" width="200">

### Nearest Neighbors

```
aleatoric, epistemic = unc.margin_uncertainties(X, y, X_test, model="NearestNeighbors")
plots.plot_uncertainties(X_test, aleatoric, epistemic)
```

<img src="images/al_knn.png" alt="AU_KNN" width="200"><img src="images/ep_knn.png" alt="EU_KNN" width="200">


## Reference
