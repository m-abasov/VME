import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import data_cleaning as dc


def corrMatrix(returns_weighted, returns_unweighted, labels):
    """Compute the (average) correlation matrix described in Asness (2013) on p. 948

    Inputs:
    - returns_weighted: a list of Series of volatility weigthed returns for each portfolio
    - returns_unweighted: a list of Series of unweighted returns
    - labels: list of labels for the matrix rows/columns

    Returns: the correlation matrix.
    """

    n = len(returns_weighted)
    corrs = pd.DataFrame(index=labels, columns=labels)

    # Off-diagonal correlations
    for i in range(n):
        for j in range(i + 1, n):
            corr_iter = returns_weighted[i].corr(returns_weighted[j])
            corrs.iloc[i, j] = corr_iter

    # On-diagonal correlations
    YQuarter = returns_unweighted[0].index.to_series().apply(dc.quarter)
    YQuarter.set_axis(returns_unweighted[0].index, inplace=True)

    returns_unweighted_quarterly = []
    for ret in returns_unweighted:
        ret_unweighted = ret.groupby(YQuarter).agg(sum)
        ret_unweighted = ret_unweighted.replace(0.0, np.nan)
        returns_unweighted_quarterly.append(ret_unweighted)

    for i in range(n):
        corr_temp = []
        ret = returns_unweighted_quarterly[i]
        for j in range(n):
            returnOthers = ret.sum(axis=1) - ret.iloc[:, j]
            corr_iter = returnOthers.corr(ret.iloc[:, j])
            corr_temp.append(corr_iter)
        corrs.iloc[i, i] = sum(corr_temp) / n

    return corrs


def portfolio_pca(df, idx, idx_num, idx_delta):
    """Implements PCA for a specified range of indexes within the DataFrame df"""

    idxs = [idx + k * idx_delta for k in range(idx_num)]
    covMatrix_val = df.iloc[:, idxs].cov()
    pca_val = PCA()
    pca_val.fit(covMatrix_val)

    return pca_val.components_[0]
