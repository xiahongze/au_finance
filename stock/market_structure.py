"""
Adapted from this example
http://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html

Novelties:
1. historical data from yahoo finance
2. Australian market
3. Examine covariance within/outside market sectors
4. remove none-trading dates and align dates
"""

import random
import numpy as np
import pandas as pd
import yahoo_finance as yf

def dropna_n_align_date(func):
    def func_wrapper(*args, **kwargs):
        d = func(*args, **kwargs)
        indices = []
        for k in d:
            d[k].dropna(inplace=True)
            indices.append(d[k].index)
        common_index = reduce(lambda x, y: x.join(y, how="inner"), indices)
        for k in d:
            d[k] = d[k].loc[common_index]
        return d
    return func_wrapper

@dropna_n_align_date
def get_random_stocks_online(n, startDate="1990-01-01"):
    symb = random.sample(yf.stockAXS.index, n) 
    res = yf.get_selected_hist(symb, startDate)
    res = {k:res[k] for k in res if res[k] is not None}
    return res

@dropna_n_align_date
def get_random_stocks_file(n, filename):
    import pickle
    f = open(filename, "rb")
    stocks = pickle.load(f)
    symb = random.sample(stocks.keys(), n)
    f.close()
    del pickle
    return {symb[i]: stocks[symb[i]] for i in xrange(len(symb))}

def get_daily_variation(d):
    d_var = {k: d[k]["Close"] - d[k]["Open"] for k in d}
    df_var = pd.DataFrame(d_var)
    return (df_var / df_var.std(axis=0)).dropna(axis=0)
    
def discover_clusters(var):
    from sklearn import cluster, covariance
    # Learn a graphical structure from the correlations
    edge_model = covariance.GraphLassoCV()
    edge_model.fit(var)
    
    # Cluster using affinity propagation
    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    n_labels = labels.max()
    for i in xrange(n_labels + 1):
        print 'Cluster %i: %s' % (i, \
            ', '.join(var.columns[labels == i]))
    del cluster, covariance
    
    return labels, edge_model.precision_.copy()

def get_low_dim_embedding(X, n_neighbors=4):
    """
    Find a low-dimension embedding for visualization: find the best position of
    the nodes (the stocks) on a 2D plane

    We use a dense eigen_solver to achieve reproducibility (arpack is
    initiated with random vectors that we don't control). In addition, we
    use a large number of neighbors to capture the large-scale structure.
    """
    from sklearn import manifold
    
    if n_neighbors >= X.shape[1]:
        n_neighbors = X.shape[1] - 1
        print "Warning: reduced n_neighbors to %d" % n_neighbors

    node_position_model = manifold.LocallyLinearEmbedding(
        n_components=2, eigen_solver='dense', n_neighbors=n_neighbors
        )

    embedding = node_position_model.fit_transform(X.T).T
    del manifold
    return embedding
    
def plot_market_structure(names, labels, embedding, partial_correlations):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    # Visualization
    plt.figure(1, facecolor='w', figsize=(10, 8))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')

    # Display a graph of the partial correlations
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

    # Plot the nodes using the coordinates of our embedding
    plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
                cmap=plt.cm.spectral)

    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    #a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    try:
        lc = LineCollection(segments,
                            zorder=0, cmap=plt.cm.hot_r,
                            norm=plt.Normalize(0, .7 * values.max()))
        lc.set_array(values)
        lc.set_linewidths(8 * values)
    except ValueError:
        print "Warning: skip line normalization"
        lc = LineCollection(segments,
                            zorder=0, cmap=plt.cm.hot_r)
        lc.set_linewidths(1)
    ax.add_collection(lc)

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    for index, (name, label, (x, y)) in enumerate(
            zip(names, labels, embedding.T)):

        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002
        plt.text(x, y, name, size=10,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 color='black',
                 bbox=dict(facecolor='w',
                           edgecolor=plt.cm.spectral(label / float(labels.max())),
                           alpha=.6))

    plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
             embedding[0].max() + .10 * embedding[0].ptp(),)
    plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
             embedding[1].max() + .03 * embedding[1].ptp())

    plt.show()
    plt.close()
    del plt, LineCollection
    
    
if __name__ == "__main__":
    # d = get_random_stocks_online(15)
    d = get_random_stocks_file(20, "stock200.pckl")
    var = get_daily_variation(d)
    labels, partial_correlations = discover_clusters(var)
    embedding = get_low_dim_embedding(var, 3)
    plot_market_structure(var.columns, labels, embedding, partial_correlations)