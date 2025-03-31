import numpy as np
from matplotlib import cm as CM
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

R3 = 1.73205
S = 60
SIZE = [int(S * R3), S]
COLORS = [[0, .8, 0], [.8, 0, 0], [0, 0, .8]]

# Scatter plot uncertainties
def plot_uncertainties(X_test, aleatoric, epistemic):
    draw_unc(X_test, aleatoric**0.8, "AU")
    draw_unc(X_test, epistemic**0.8, "EU")
    plt.show()

# Scatter plot dataset figure
def draw_unc(X_test, certainties, text):
    fig = plt.figure(figsize=(6, 4), dpi=100)

    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[5, 1], wspace=0)
    subplot = fig.add_subplot(gs[0, 0])

    # Remove useless values
    gridsize=(int(SIZE[0] * 0.68), int(SIZE[1] * 0.43))
    subplot.hexbin(X_test[:, 0], X_test[:, 1], C=certainties, gridsize=gridsize, cmap=CM.jet, bins=None)

    # Remove black rectangle
    subplot.spines['top'].set_visible(False)
    subplot.spines['right'].set_visible(False)
    subplot.spines['bottom'].set_visible(False)
    subplot.spines['left'].set_visible(False)

    # Remove ticks form figure
    subplot.set(yticklabels=[], xticklabels=[]) 
    subplot.tick_params(left=False, bottom=False) 

    # Same size for each figure
    subplot.set_xlim(np.min(X_test[:, 0]) + 0.2, np.max(X_test[:, 0]) - 0.2)
    subplot.set_ylim(np.min(X_test[:, 1]) + 0.2, np.max(X_test[:, 1]) - 0.2)

    subplot.set_position([0, 0, 5/6, 1])

    ax = fig.add_subplot(gs[0, 1])
    gradient = np.linspace(1, 0, 256).reshape(-1, 1)
    ax.imshow(gradient, aspect='auto', cmap='jet', extent=[0.2, 0.8, 0.2, 0.6])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_position([0.87, 0.15, 0.09, 0.7])

    plt.text(0.873, 0.9, "Max " + text, transform=plt.gcf().transFigure)
    plt.text(0.873, 0.08, "Min " + text, transform=plt.gcf().transFigure)
    

# Scatter plot dataset figure
def plot_dataset(X, y):

    fig = plt.figure(figsize=(5, 4), dpi=100)

    subplot = fig.add_subplot(111)
    subplot.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(COLORS))
    
    # Remove black rectangle
    subplot.spines['top'].set_visible(False)
    subplot.spines['right'].set_visible(False)
    subplot.spines['bottom'].set_visible(False)
    subplot.spines['left'].set_visible(False)

    # Remove ticks form figure
    subplot.set(yticklabels=[], xticklabels=[]) 
    subplot.tick_params(left=False, bottom=False) 

    # Same size for each figure
    subplot.set_xlim(np.min(X[:, 0]) - 0.2, np.max(X[:, 0]) + 0.2)
    subplot.set_ylim(np.min(X[:, 1]) - 0.2, np.max(X[:, 1]) + 0.2)

    # Full zoom
    fig.subplots_adjust(0,0,1,1)

    plt.show()

# Load Heatmap test set
def load_Xtest(X):
    X_test = np.zeros((SIZE[0] * SIZE[1], 2))

    minx1 = np.min(X[:,0])
    minx1 = minx1 + max(minx1, -minx1) * 0.01
    maxx1 = np.max(X[:,0])
    maxx1 = maxx1 - max(maxx1, -maxx1) * 0.01
    minx2 = np.min(X[:,1])
    minx2 = minx2 + max(minx2, -minx2) * 0.01
    maxx2 = np.max(X[:,1])
    maxx2 = maxx2 - max(maxx2, -maxx2) * 0.01

    for i in range(SIZE[0]):
        for j in range(SIZE[1]):
            X_test[i + j * SIZE[0]][0] = minx1 + i * (maxx1 - minx1) / SIZE[0]
            X_test[i + j * SIZE[0]][1] = minx2 + j * (maxx2 - minx2) / SIZE[1]

    return X_test