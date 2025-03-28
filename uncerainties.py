from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
import numpy as np
import cvxpy as cp

def margin_uncertainties(X, y, X_test, model="RandomForest"):
    """
    Compute margin uncertainties on X_test based on a specific model.
    Args:
        X: array of inputs
        y: array of classes
        X_test: array on test instances
        model: Type of model to use among ['ParzenWindow', 'DecisionTree', 'RandomForest'] (default=RandomForest)
    Returns:
        uncertainties: aleatoric and epistemic uncertainty 
    """

    aleatoric = np.zeros(X_test.shape[0])
    epistemic = np.zeros(X_test.shape[0])

    # Parzen Window
    if model == "ParzenWindow":
        cls = NearestNeighbors(n_neighbors=X.shape[0]).fit(X)
        dist, indices = cls.kneighbors(X_test)

        for i in range(X_test.shape[0]):
            valid_indices = indices[i][dist[i] < 0.75]
            neighbor_classes = y[valid_indices]
            support = np.array([np.sum(neighbor_classes == c) for c in np.unique(y)])
            aleatoric[i], epistemic[i] = compute_margin(support)

    # Decision Tree
    elif model == "DecisionTree":
        cls = DecisionTreeClassifier(min_samples_leaf=5)
        cls.fit(X, y)
        tree = cls.tree_
        leaf_indices = cls.apply(np.array(X_test))

        for i in range(X_test.shape[0]):
            support = tree.value[leaf_indices[i]][0]
            support = support * tree.n_node_samples[leaf_indices[i]]
            aleatoric[i], epistemic[i] = compute_margin(support)

    # Decision Tree
    elif model == "NearestNeighbors":
        cls = NearestNeighbors(n_neighbors=7).fit(X)
        dist, indices = cls.kneighbors(X_test)

        for i in range(X_test.shape[0]):
            support = np.ceil([np.sum(1 / dist[i][y[indices[i]] == c]) for c in range(np.max(y) + 1)])
            aleatoric[i], epistemic[i] = compute_margin(support)

    # Random Forest
    elif model == "RandomForest":
        cls = RandomForestClassifier(min_samples_leaf=5)
        cls.fit(X, y)
        size = len(cls.estimators_)
        trees = [estimator.tree_ for estimator in cls.estimators_]
        leaf_indices = [estimator.apply(np.array(X_test)) for estimator in cls.estimators_]
        leaf_depth = [np.sum(estimator.decision_path(X_test).toarray(), axis=1) - 1 for estimator in cls.estimators_]

        for i in range(X_test.shape[0]):
            aleatoric_temp = 0
            epistemic_temp = 0

            for j in range(size):
                leaf_index = leaf_indices[j][i]
                support = trees[j].value[leaf_index][0]
                support = support * trees[j].n_node_samples[leaf_index]
                support = support * leaf_depth[j][i]
                
                al, ep = compute_margin(support)
                aleatoric_temp += al
                epistemic_temp += ep

            aleatoric[i] = (aleatoric_temp / size)
            epistemic[i] = (epistemic_temp / size)

    else:
        raise ValueError("Argument 'model' must be a in the list.")

    return aleatoric, epistemic

# Cache used to run optimization faster
def cache(f):
    caches = {}
    def wrapper(*args, **kwargs):
        all_args = str(args) + str(kwargs)
        if all_args not in caches:
            caches[all_args] = f(*args,**kwargs)
        return caches[all_args]

    return wrapper

# Compute margin uncertainties
@cache
def compute_margin(support):
    """
    Compute margin uncertainties based on a support vector
    Args:
        support: array of shape (K)
    Returns:
        uncertainties: aleatoric and epistemic uncertainty 
    """
    K = len(support)
    margins_args = np.argsort(support)[::-1][0:2]

    opt = minimize_scalar(f_objective, bounds=(1/K, 1), method='bounded', args=(support, margins_args[0]))
    pl1 = opt.x

    opt = minimize_scalar(f_objective, bounds=(1/K, 1), method='bounded', args=(support, margins_args[1]))
    pl2 = opt.x

    ue = min(pl1, pl2)
    ua = 1 - max(pl1, pl2)

    return ua, ue

def f_objective(theta, support, k):
    """
    Objective fuction used to compute uncertainties
    """
    K = len(support)

    thetas = solve_theta(support, theta, k)
    thetas_star = support / np.sum(support)

    left = np.prod(thetas**support) / np.prod(thetas_star**support)
    right = ((K * theta) - 1) / (K - 1)

    res = min(left, right)
    return -res

def solve_theta(f_eps, theta_k, k):
    """
    Solves the optimization problem:
        maximize sum_{k' != k} f_eps[k'] * log(theta[k'])
        subject to:
            sum_{k' != k} theta[k'] = 1 - theta_k
            theta_k - theta[k'] >= 0 for all k' != k
            theta[k'] >= 0
    Args:
        f_eps: array of shape (K,) with values f^k'_ε(x)
        k: index k (int)
    Returns:
        Optimal theta vector of length K
    """
    K = len(f_eps)

    # Indices excluding k
    other_indices = [i for i in range(K) if i != k]
    
    # Optimization variables: theta_{k'} for k' != k
    theta_other = cp.Variable(K - 1)

    # Objective
    f_eps_other = np.array([f_eps[i] for i in other_indices])

    objective = cp.Maximize(cp.sum(cp.multiply(f_eps_other, cp.log(theta_other))))

    # Constraints
    constraints = [
        cp.sum(theta_other) == 1 - theta_k,
        theta_other >= 0,
        theta_k - theta_other >= 0
    ]

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Assemble full theta vector
    theta_full = np.zeros(K)
    theta_full[k] = theta_k
    for idx, i in enumerate(other_indices):
        theta_full[i] = theta_other.value[idx]

    return theta_full