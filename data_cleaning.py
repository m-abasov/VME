import numpy as np


def portfolioSplits(df, date_var, portfolio_indexes, idx_num=1, idx_delta=None):
    """ Split entire dataframe into assets in different portfolios.

    Inputs:
    - df: DataFrame containing asset returns
    - date_var: a string denoting the date column in df
    - portfolio_indexes: indexes of the assets within df in each portfolio
    - idx_num: the number of assets in each portfolio (default = 1)
    - idx_delta: defines how spread out the assets are within df (default = None)

    Returns: a list of DataFrames for each portfolio.
    """

    # Get rid of the date column
    df[date_var] = df[date_var].apply(lambda x: x // 100)
    df.set_index(date_var, inplace=True)

    # Split the dataframe into portfolios
    returns = []
    if idx_num == 1:
        for i in portfolio_indexes:
            returns.append(df.iloc[:, i])
    else:
        for i in portfolio_indexes:
            idxs = [i + idx_delta * k for k in range(idx_num)]
            returns.append(df.iloc[:, idxs])
    return returns


def volatilityWeights(returns, roll_window, min_count_req=1):
    """ Implement volatility weighting for each portfolio in returns.

    Inputs:
    - returns: DataFrame containing asset returns for each portfolio
    - roll_window: rolling number of months to estimate volatility (std)
    - min_count_req: minimum number of asset returns in a given period to implement weighting

    Returns: a list of DataFrames for each weighted portfolio.
    """

    # Compute the weights
    vltWeights = []
    for ret in returns:
        stDev = ret.rolling(window=roll_window, min_periods=roll_window).std()
        stDev["Total"] = stDev.sum(axis=1, min_count=min_count_req)
        stDev.iloc[:, :-1] = stDev.iloc[:, :-1].div(stDev.Total, axis=0)
        vltWeights.append(stDev.iloc[:, :-1])

    # Implement the weigths
    returns_weighted = []
    for i in range(4):
        ret = returns[i]
        ret_weighted = ret * vltWeights[i]
        returns_weighted.append(ret_weighted)

    return returns_weighted


def quarter(date):
    """ Determines the quarter in the format YYYYQQ """
    year = date // 100
    month = date % 100
    if month < 4:
        quarter = 1
    elif month < 7:
        quarter = 2
    elif month < 10:
        quarter = 3
    else:
        quarter = 4
    return str(year) + "Q" + str(quarter)


def dfQuarterly(returns, roll_window, min_count_req=1, ignore_nan=True):
    """ Compute aggregate returns for a volatility weighted portfolios

    Inouts:
    - returns: DataFrame containing asset returns for each portfolio
    - roll_window: rolling number of months to estimate volatility (std)
    - min_count_req: minimum number of asset returns in a given period to implement weighting
    - ignore_nan: indicates whether to ignore NaNs or treat them as zeros.

    Returns: a list of Series of aggregate returns for each weighted portfolio.
    """

    # Implement volatility weighting
    returns = volatilityWeights(returns, roll_window, min_count_req=min_count_req)

    # Compute quarterly aggregate returns
    returns_quarterly = []
    YQuarter = returns[0].index.to_series().apply(quarter)
    YQuarter.set_axis(returns[0].index, inplace=True)
    for ret in returns:

        ret_quarterly = ret.groupby(YQuarter).agg(sum)  # Agregate returns quarterly
        if ignore_nan:
            ret_quarterly = ret_quarterly.replace(0.0, np.nan)

        returns_quarterly.append(
            ret_quarterly.sum(axis=1, skipna=False)
        )  # Sum the returns for each asset in the portfolio

    return returns_quarterly
