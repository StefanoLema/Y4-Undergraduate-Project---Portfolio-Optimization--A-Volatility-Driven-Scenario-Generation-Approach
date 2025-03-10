
##############################################################
# Import Required Libraries                                  #
##############################################################

# Main Notebook Libraries, only used in inline code
import matplotlib.pyplot as plt # General plotting
import matplotlib as mpl # General Plotting
from IPython.display import display # Plot display
import yfinance as yf # Data imports using Yahoo Finance
from statsmodels.tsa.stattools import adfuller # ADF evaluaiton

# For dataframe and array handling
import numpy as np
import pandas as pd

# For risk-free rate calculations
from fredapi import Fred
#fred = Fred(api_key='YOUR API KEY')

# For custom scenario generation and optimization classes
import cvxpy as cp
from scipy.optimize import minimize
from scipy.stats import t as student_t
import scipy as sp

# For PyportfolioOpt main optimization class
from pypfopt import EfficientFrontier
from pypfopt import EfficientCVaR

##############################################################
# Functions for Periodic Volatility analysis                 #
##############################################################


# Enforce Yearly Stationarity Function
def make_stationary(series, window=252):
    """
    Transforms a time series to be yearly stationary by removing the rolling mean and normalizing the volatility.

    Parameters:
    series (pd.Series): The time series data to be transformed.
    window (int): The window size for calculating the rolling mean and standard deviation. Default is 252.

    Returns:
    pd.Series: The transformed stationary time series.
    """
    rolling_mean = series.rolling(window).mean()
    detrended = series - rolling_mean
    return detrended / series.rolling(window).std()



# Compute Cross-Correlation Function (CCF)
def compute_normalized_ccf(x, y):
    """
    Compute the normalized cross-correlation function (CCF) between two time series arrays.

    Parameters:
    x (array-like): First time series array.
    y (array-like): Second time series array.

    Returns:
    tuple: A tuple containing:
        - ccf (numpy.ndarray): Cross-correlation function values.
        - lags (numpy.ndarray): Array of lag values.
    """
    ccf = sp.signal.correlate(x, y)
    norm_factor = np.linalg.norm(x) * np.linalg.norm(y)
    ccf_norm = ccf / norm_factor
    lags = sp.signal.correlation_lags(len(x), len(y))
    return ccf_norm, lags

# Periodic Volatility Sampling Algorithm
def sample_periodic_volatility(vxd_series_with_vl, n_simulations, lower_quantile, upper_quantile, min_gap_days=21, random_state=None):
    """
    Sample periodic volatility from a given time series with volatility levels.

    Parameters:
    vxd_series_with_vl (pd.Series): Time series data with volatility levels.
    n_simulations (int): Number of simulations to run for each volatility type.
    lower_quantile (float): Lower quantile threshold for categorizing low volatility.
    upper_quantile (float): Upper quantile threshold for categorizing high volatility.
    min_gap_days (int): Minimum gap in days between sampled dates. Default is 21.
    random_state (int, optional): Seed for the random number generator. Default is None.

    Returns:
    dict: A dictionary containing lists of dates for low, medium, and high volatility periods.
    """
    # Initialize lists to store the results
    low_volatility_dates = []
    medium_volatility_dates = []
    high_volatility_dates = []
    
    # Number of trading days in a 1-year period (assuming daily data, approximately 252 trading days per year)
    # Plus 21 days for out of sample validation
    period_length = 252 + 21
    
    # Set the random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    print('Simulating...')
    
    # Function to check if the new date is at least 'min_gap_days' away from any existing date
    def is_date_far_enough(new_date, existing_dates):
        return all(abs(new_date - existing_date) >= min_gap_days for existing_date in existing_dates)
    
    # Simulate until we have n_simulations of each volatility type
    while len(low_volatility_dates) < n_simulations or len(medium_volatility_dates) < n_simulations or len(high_volatility_dates) < n_simulations:
        
        # Randomly select a start date for the period
        random_start_date = np.random.choice(range(len(vxd_series_with_vl) - period_length))
        random_end_date = random_start_date + period_length
        
        # Select the subset of the data corresponding to the random period
        period_data = vxd_series_with_vl[random_start_date:random_end_date]
        
        # Calculate the average of the ^VXD column for this period
        avg_vxd = period_data.mean()
        
        # Categorize based on the average value
        if avg_vxd + 1 <= lower_quantile and len(low_volatility_dates) < n_simulations:
            # Check if the new date is at least 30 days away from existing dates
            if is_date_far_enough(random_start_date, low_volatility_dates):
                low_volatility_dates.append(random_start_date)
        elif avg_vxd - 1 >= lower_quantile and avg_vxd + 1 <= upper_quantile and len(medium_volatility_dates) < n_simulations:
            # Check if the new date is at least 30 days away from existing dates
            if is_date_far_enough(random_start_date, medium_volatility_dates):
                medium_volatility_dates.append(random_start_date)
        elif avg_vxd - 1 > upper_quantile and len(high_volatility_dates) < n_simulations:
            # Check if the new date is at least 30 days away from existing dates
            if is_date_far_enough(random_start_date, high_volatility_dates):
                high_volatility_dates.append(random_start_date)
    
    print('Simulation complete!')
    
    # Return the results as dictionaries of lists
    return {
        'low_volatility_dates': low_volatility_dates,
        'medium_volatility_dates': medium_volatility_dates,
        'high_volatility_dates': high_volatility_dates
    }
    
#############################################################
# Importing Classical ARMA-GARCH and Copula-ARMA-GARCH      #
# simulations generated in R                                #
#############################################################

def load_and_reshape(file=None, file_names=None, n_sim=1000, days=21, n_stocks=9):
    """
    Load and reshape CSV files containing simulation data.

    Parameters:
    file (str, optional): Single file name to be loaded.
    file_names (list of str, optional): List of file names to be loaded. These files must match the R generated file names.
    n_sim (int): Number of simulations. Default is 1000.
    days (int): Number of days in each simulation. Default is 21.
    n_stocks (int): Number of stocks in each simulation. Default is 9.

    Returns:
    list of np.ndarray: A list of 3D numpy arrays with shape (n_sim, days, n_stocks).
    """
    if file_names is not None:
        arrays = []
        for file_name in file_names:
            df = pd.read_csv(file_name).values
            reshaped_array = df.reshape(n_sim, days, n_stocks, order="F")
            arrays.append(reshaped_array)
        return arrays
    elif file is not None:
        df = pd.read_csv(file).values
        reshaped_array = df.reshape(n_sim, days, n_stocks, order="F")
        return reshaped_array
        


#############################################################
# Scenario Generation Class                                 #
#############################################################

class ScenarioGenerator:
    """
    A class to generate various scenarios for financial time series data using different methods.

    Attributes:
    historical_returns (pd.DataFrame): DataFrame of historical returns.
    num_scenarios (int): Number of simulation paths.
    n_days (int): Simulation horizon.
    random_state (int, optional): Seed for reproducibility.
    num_assets (int): Number of assets in the historical returns DataFrame.
    """

    def __init__(self, historical_returns: pd.DataFrame, num_scenarios: int, n_days: int, random_state: int = None):
        """
        Initialize the ScenarioGenerator with historical returns and simulation parameters.

        Parameters:
        historical_returns (pd.DataFrame): DataFrame of training returns.
        num_scenarios (int): Number of simulation paths.
        n_days (int): Simulation horizon.
        random_state (int, optional): Seed for reproducibility.
        """
        self.historical_returns = historical_returns
        self.num_scenarios = num_scenarios
        self.n_days = n_days
        self.num_assets = historical_returns.shape[1]
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def historical_data(self) -> np.ndarray:
        """
        Generate scenarios using historical data.

        Returns:
        np.ndarray: Array of scenarios with shape (num_scenarios, n_days, num_assets).
        """
        T = len(self.historical_returns)
        scenarios = []
        for _ in range(self.num_scenarios):
            start = np.random.randint(0, T - self.n_days + 1)
            block = self.historical_returns.iloc[start:start + self.n_days].values
            scenarios.append(block)
        return np.array(scenarios)  # shape: (M, n_days, n_assets)

    def bootstrap(self) -> np.ndarray:
        """
        Generate scenarios using bootstrap sampling.

        Returns:
        np.ndarray: Array of scenarios with shape (num_scenarios, n_days, num_assets).
        """
        T = len(self.historical_returns)
        scenarios = np.empty((self.num_scenarios, self.n_days, self.num_assets))
        for i in range(self.num_scenarios):
            indices = np.random.randint(0, T, size=self.n_days)
            scenarios[i] = self.historical_returns.iloc[indices].values
        return scenarios

    def block_bootstrap(self, block_size: int = 5) -> np.ndarray:
        """
        Generate scenarios using block bootstrap sampling.

        Parameters:
        block_size (int): Size of the blocks to sample. Default is 5.

        Returns:
        np.ndarray: Array of scenarios with shape (num_scenarios, n_days, num_assets).
        """
        T = len(self.historical_returns)
        scenarios = []
        for _ in range(self.num_scenarios):
            sim_path = []
            while len(sim_path) < self.n_days:
                start = np.random.randint(0, T - block_size + 1)
                block = self.historical_returns.iloc[start:start + block_size].values
                sim_path.extend(block.tolist())
            sim_path = np.array(sim_path[:self.n_days])
            scenarios.append(sim_path)
        return np.array(scenarios)

    def monte_carlo_normal(self) -> np.ndarray:
        """
        Generate scenarios using Monte Carlo simulation with a normal distribution.

        Returns:
        np.ndarray: Array of scenarios with shape (num_scenarios, n_days, num_assets).
        """
        mean_vec = self.historical_returns.mean().values
        cov_mat = self._ensure_psd(self.historical_returns.cov())
        M = self.num_scenarios * self.n_days
        sims = np.random.multivariate_normal(mean_vec, cov_mat, size=M)
        return sims.reshape(self.num_scenarios, self.n_days, self.num_assets)

    def monte_carlo_tstudent(self, df: int = 5) -> np.ndarray:
        """
        Generate scenarios using Monte Carlo simulation with a multivariate t-distribution.

        Parameters:
        df (int): Degrees of freedom for the t-distribution. Default is 5.

        Returns:
        np.ndarray: Array of scenarios with shape (num_scenarios, n_days, num_assets).
        """
        mean_vec = self.historical_returns.mean().values
        cov_mat = self._ensure_psd(self.historical_returns.cov())
        M = self.num_scenarios * self.n_days
        x = np.random.chisquare(df, M) / df
        normal_sims = np.random.multivariate_normal(mean_vec, cov_mat, size=M)
        mult_t =  mean_vec + normal_sims/np.sqrt(x)[:,None] 
        return mult_t.reshape(self.num_scenarios, self.n_days, self.num_assets)

    def _ensure_psd(self, matrix: pd.DataFrame, epsilon: float = 1e-8) -> np.ndarray:
        """
        Ensure that a covariance matrix is positive semi-definite (PSD).
        Necessary for Optimization.

        Parameters:
        matrix (pd.DataFrame): Covariance matrix.
        epsilon (float): Small value to add to the diagonal for numerical stability. Default is 1e-8.

        Returns:
        np.ndarray: PSD covariance matrix.
        """
        A = np.array(matrix)
        try:
            np.linalg.cholesky(A)
            return A
        except np.linalg.LinAlgError:
            eigvals_arr, eigvecs = np.linalg.eigh(A)
            eigvals_clipped = np.where(eigvals_arr < 0, 0, eigvals_arr)
            psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
            psd += epsilon * np.eye(A.shape[0])
            return psd


#############################################################
# Portfolio Optimization Default Class                      #
#############################################################


class PortfolioOptimizer:
    """
    This optimizer accepts a 3D array of simulated returns with shape (M, n_days, n_assets).
    It aggregates each simulation path to compute terminal returns and then computes the 
    expected terminal returns and sample covariance for portfolio optimization.
    """
    def __init__(self, simulations_3d: np.ndarray):
        """
        Initialize the PortfolioOptimizer with simulated returns.

        Parameters:
        simulations_3d (np.ndarray): 3D numpy array with shape (n_scenarios, n_days, n_assets)
        """
        self.simulations_3d = simulations_3d
        self.terminal_returns = np.prod(1 + simulations_3d, axis=1) - 1
        self.M, self.n_assets = self.terminal_returns.shape
        self.expected_returns = self.terminal_returns.mean(axis=0)
        self.cov_matrix = np.cov(self.terminal_returns, rowvar=False)

    def max_return(self, target_return):
        """
        Compute the maximum feasible return given a target return.

        Parameters:
        target_return (float): The target return.

        Returns:
        float: The feasible return.
        """
        mu = self.expected_returns
        S = self.cov_matrix
        ef = EfficientFrontier(mu, S, solver="CLARABEL")
        max_ret = ef._max_return()
        feasible_ret = min(max_ret, target_return)
        return feasible_ret

    def _safe_normalize(self, w):
        """
        Normalize the weights to ensure they sum to 1.
        Removes possible numerical errors.

        Parameters:
        w (np.ndarray): The weights to normalize.

        Returns:
        np.ndarray: The normalized weights.
        """
        if np.any(w < -1e-8):
            w = np.maximum(w, 0)
        s = np.sum(w)
        if s > 1e-6:  
            return w / s
        return np.ones(self.n_assets) / self.n_assets 

    def mean_variance_optimization(self, target_return=0.0):
        """
        Perform mean-variance optimization to achieve the target return.

        Parameters:
        target_return (float): The target return. Default is 0.0.

        Returns:
        np.ndarray: The optimized portfolio weights.
        """
        mu = self.expected_returns
        S = self.cov_matrix
        ef = EfficientFrontier(mu, S, solver="CLARABEL")
        feasible_target = self.max_return(target_return)
        w = ef.efficient_return(target_return=feasible_target)
        return np.array(list(w.values()))

    def mean_cvar_optimization(self, target_return=None):
        """
        Perform mean-CVaR optimization to achieve the target return.

        Parameters:
        target_return (float, optional): The target return. Default is None.

        Returns:
        np.ndarray: The optimized portfolio weights.
        """
        mu = self.expected_returns
        ef = EfficientCVaR(returns=self.terminal_returns, expected_returns=mu, solver="CLARABEL")
        feasible_target = self.max_return(target_return)
        w = ef.efficient_return(target_return=feasible_target)
        return np.array(list(w.values()))

    def max_sharpe(self, rf_rate=0.0017, target_return=0.0):
        """
        Perform optimization to maximize the Sharpe ratio.

        Parameters:
        rf_rate (float): The risk-free rate. Default is 0.0017.
        target_return (float): The target return. Default is 0.0.

        Returns:
        np.ndarray: The optimized portfolio weights.
        """
        mu = self.expected_returns
        Sigma = self.cov_matrix
        feasible_target = self.max_return(target_return)
        try:
            ef = EfficientFrontier(mu, Sigma, solver="CLARABEL")
        except (UserWarning, ValueError) as e:
            ef = EfficientFrontier(mu, Sigma, solver="SCS")
        ef.add_constraint(lambda w: w @ mu >= feasible_target)
        w = ef.max_sharpe(risk_free_rate=rf_rate)
        return np.array(list(w.values()))
    
#############################################################
# Portfolio Optimization Custom  Class                      #
#############################################################

class PortfolioOptimizerCustom:
    """
    This is an alternative portfolio optimization class we have customized in case optimization problems 
    want to be customized but it is not used in the project.
    
    This optimizer accepts a 3D array of simulated returns with shape (M, n_days, n_assets).
    It aggregates each simulation path to compute terminal returns (here, by summing over days)
    and then computes the expected terminal returns and sample covariance for portfolio optimization.
    """
    def __init__(self, simulations_3d: np.ndarray):
        """
        Initialize the PortfolioOptimizerCustom with simulated returns.

        Parameters:
        simulations_3d (np.ndarray): 3D numpy array with shape (n_scenarios, n_days, n_assets)
        """
        self.simulations_3d = simulations_3d
        # Aggregate simulated returns along the days dimension (axis=1) to get terminal returns per scenario
        self.terminal_returns = np.prod(1 + simulations_3d, axis=1) - 1
        self.M, self.n_assets = self.terminal_returns.shape
        self.expected_returns = self.terminal_returns.mean(axis=0)
        self.cov_matrix = np.cov(self.terminal_returns, rowvar=False)
   
    def _safe_normalize(self, w):
        """
        Normalize the weights to ensure they sum to 1.
        Removes possible numerical errors.

        Parameters:
        w (np.ndarray): The weights to normalize.

        Returns:
        np.ndarray: The normalized weights.
        """
        if np.any(w < -1e-8):  # Detect unexpected negatives (shouldn't happen)
            w = np.maximum(w, 0)
        s = np.sum(w)
        if s > 1e-6:  # Only normalize if sum is meaningful
            return w / s
        return np.ones(self.n_assets) / self.n_assets  # Fallback if the solution is completely degenerate
    
    def mean_variance_optimization(self, target_return=0.0):
        """
        Perform mean-variance optimization to achieve the target return.

        Parameters:
        target_return (float): The target return. Default is 0.0.

        Returns:
        np.ndarray: The optimized portfolio weights.
        """
        w = cp.Variable(self.n_assets)
        slack = cp.Variable(nonneg=True)
        mu = self.expected_returns
        Sigma = self.cov_matrix
        constraints = [cp.sum(w) == 1,
                       w >= 0,
                       mu @ w >= target_return - slack]
        objective = cp.Minimize(cp.quad_form(w, Sigma) + 1e6 * slack)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver="CLARABEL", verbose=False)
        if w.value is None:
            return self._safe_normalize(np.ones(self.n_assets))
        return w.value
    
    def mean_cvar_optimization(self, confidence_level=0.95, target_return=None):
        """
        Perform mean-CVaR optimization to achieve the target return.

        Parameters:
        confidence_level (float): The confidence level for CVaR. Default is 0.95.
        target_return (float, optional): The target return. Default is None.

        Returns:
        np.ndarray: The optimized portfolio weights.
        """
        w = cp.Variable(self.n_assets)
        slack = cp.Variable(nonneg=True)
        alpha = 1 - confidence_level
        scenario_returns = self.terminal_returns @ w
        losses = -scenario_returns
        VaR = cp.Variable()
        z = cp.Variable(self.M, nonneg=True)
        cvar = VaR + (1.0 / (alpha * self.M)) * cp.sum(z)
        constraints = [cp.sum(w) == 1, w >= 0] 
        if target_return is not None:
            constraints.append(self.expected_returns @ w >= target_return - slack)
        for i in range(self.M):
            constraints.append(z[i] >= losses[i] - VaR)
        objective = cp.Minimize(cvar + 1e6 * slack)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver="CLARABEL", verbose=False)
        if w.value is None:
            return self._safe_normalize(np.ones(self.n_assets))
        return w.value
    
    def cvar_shortfall_optimization(self, liability=0.0, confidence_level=0.95, target_return=None, B=None):
        """
        Perform CVaR shortfall optimization to achieve the target return.

        Parameters:
        liability (float): The liability threshold.
        confidence_level (float): The confidence level for CVaR. Default is 0.95.
        target_return (float, optional): The target return. Default is None.
        B (float, optional): The scaling factor for scenario returns. Default is None.

        Returns:
        tuple: The optimized portfolio weights, t value, and z values.
        """
        w = cp.Variable(self.n_assets, nonneg=True)
        slack = cp.Variable(nonneg=True)
        t = cp.Variable(nonneg=True)
        z = cp.Variable(self.M, nonneg=True)
        alpha = 1 - confidence_level
        scenario_returns = self.terminal_returns @ w
        obj = t + (1 / (alpha * self.M)) * cp.sum(z) + 1e6 * slack
        constraints = [cp.sum(w) == 1, w >= 0] 
        if target_return is not None:
            constraints.append(self.expected_returns @ w >= target_return - slack)
        for i in range(self.M):
            if B is not None:
                constraints.append(t + z[i] + B * scenario_returns[i] >= liability)
            else:
                constraints.append(t + z[i] + scenario_returns[i] >= liability)
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver="CLARABEL", verbose=False)
        if w.value is None:
            return self._safe_normalize(np.ones(self.n_assets)), None, None
        return w.value, t.value, z.value
    
    def max_sharpe(self):
        """
        Perform optimization to maximize the Sharpe ratio.
        Does not have a minimum return threshold.

        Returns:
        np.ndarray: The optimized portfolio weights.
        """
        mu = self.expected_returns
        Sigma = self.cov_matrix
        def sharpe(w):
            ret = np.dot(w, mu)
            vol = np.sqrt(w @ Sigma @ w + 1e-15)
            return ret / (vol + 1e-15)
        def obj(w):
            return -sharpe(w)
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bnds = [(0, 1)] * self.n_assets
        x0 = np.ones(self.n_assets) / self.n_assets
        res = minimize(obj, x0, method='SLSQP', bounds=bnds, constraints=cons)
        if not res.success:
            print("Max-Sharpe optimization failed:", res.message)
            return self._safe_normalize(x0)
        return res.x

#############################################################
# Evaluation of Logarithmic Returns and Terminal Wealth     #
#############################################################

def compute_log_returns(stock_df):
    """
    Compute the logarithmic returns of a stock DataFrame using 'Close' prices.

    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock prices with 'Close' prices.

    Returns:
    pd.DataFrame: DataFrame containing the logarithmic returns.
    """
    return np.log(stock_df / stock_df.shift(1)).dropna()

def evaluate_portfolio_performance(weights, validation_returns, initial_invest=1000):
    """
    Evaluate the performance of a portfolio given asset weights and validation returns.

    Parameters:
    weights (array-like): The weights of the assets in the portfolio.
    validation_returns (pd.DataFrame): A DataFrame containing the daily returns of the assets (n_days x n_assets).
    initial_invest (float, optional): The initial investment amount. Default is 1000.

    Returns:
    pd.Series: A Series representing the cumulative wealth over time, indexed by the dates in validation_returns.
    """
    # validation_returns is a DataFrame (n_days x n_assets)
    daily_ret = validation_returns.values @ weights
    cum_wealth = initial_invest * np.cumprod(1 + daily_ret)
    return pd.Series(cum_wealth, index=validation_returns.index)    


#############################################################
# Evaluation of Risk Free Rate                              #
#############################################################

def rf_rate_fetch(date: pd.Timestamp, current_rate=0.04):
    """
    Fetch the 3-month Treasury bill rate from FRED as a proxy for the risk-free rate.
    Note that this year is annualized and final utilized rate must match validation 
    horizon.

    Parameters:
    date (pd.Timestamp): The date for which to fetch the rate.
    current_rate (float): The current rate to use if fetching fails (default is 0.04).

    Returns:
    float: The 3-month Treasury bill rate as a decimal.
    """
    # Format the date to 'YYYY-MM-DD'
    date_str = date.strftime('%Y-%m-%d')
    
    # Fetch the 3-month Treasury bill rate from FRED
    try:
        # The FRED code for the 3-month Treasury Bill rate is 'DTB3'
        data = fred.get_series('DTB3', start_date=date_str, end_date=date_str)
        
        # If data is found, return the value
        if not data.empty:
            return np.round(data.iloc[0] / 100, 5)
        else:
            return None
    
    except Exception as e:
        print(f"Error fetching data: {e}, Current Rate being used")
        return current_rate / 100
    
#############################################################
# Optimization Mapper function                              #
#############################################################

def run_optimization(opt_name, po, target, rf_rate):
    """
    Run the specified portfolio optimization method.

    Parameters:
    opt_name (str): The name of the optimization method to use. 
                    Options are "Mean-Variance", "Mean-CVaR", and "Max-Sharpe".
    po (object): The portfolio object that contains the optimization methods.
                 Based on PortfolioOptimizer Classs
    target (float): The target return for the optimization.
    rf_rate (float): The risk-free rate used in the Max-Sharpe optimization.

    Returns:
    object: The result of the optimization method.

    Raises:
    ValueError: If an unknown optimization method is specified.
    """
    if opt_name == "Mean-Variance":
        return po.mean_variance_optimization(target_return=target)
    elif opt_name == "Mean-CVaR":
        return po.mean_cvar_optimization(target_return=target)
    elif opt_name == "Max-Sharpe":
        return po.max_sharpe(rf_rate, target)
    else:
        raise ValueError(f"Unknown optimization method {opt_name}")    
    
#############################################################
# Table Comparison Function used in Train                   #
#############################################################

def aggregate_terminal_wealth_results(terminal_wealth_table, scenario_methods, optimization_methods, starting_wealth=1000, benchmark = 'Historical'):
    """
    Aggregates simulation results stored in terminal_wealth_table and computes:
      - The average terminal wealth (aggregated across simulation chunks) for each volatility regime,
        with rows = optimization methods and columns = scenario methods, reported as returns relative to starting_wealth.
      - The ranking of each scenario method based on average terminal wealth.
    
    Parameters:
      terminal_wealth_table (dict): Dictionary with keys as volatility regimes (e.g., "low", "medium", "high")
                                    and values as dicts mapping simulation indices to DataFrames.
      scenario_methods (list of str): List of scenario generation method names.
      optimization_methods (list of str): List of optimization model names.
      starting_wealth (float): The initial wealth level below which rankings receive 0 points.
      benchmark (str): The scenario_method name to be used as benchmark for ranking. Defaults to Historical.
    
    Returns:
      combined_agg_table (pd.DataFrame): A multi-index DataFrame containing the mean terminal wealth
                                         as returns relative to starting_wealth.
      combined_ranking_table (pd.DataFrame): A multi-index DataFrame containing the final rankings
                                             for each scenario method across all simulations.
    """
    agg_tables = {}
    ranking_tables = {}

    for regime, sim_dict in terminal_wealth_table.items():
        chunk_list = []  
        valid_chunk_count = 0 

        for sim_key, df_chunk in sim_dict.items():
            df_chunk = df_chunk.reindex(index=optimization_methods, columns=scenario_methods)
            
            if df_chunk.isna().all().all():
                continue
            
            df_chunk = df_chunk.astype(float)
            chunk_list.append(df_chunk)
            valid_chunk_count += 1
        
        if valid_chunk_count == 0:
            print(f"[WARNING] No valid simulation chunks found for regime '{regime}'.")
            continue
        
        combined_df = pd.concat(chunk_list, keys=range(valid_chunk_count), names=["Simulation"])
        avg_df = combined_df.groupby(level=1).mean()
        
        avg_df = (avg_df - starting_wealth) / starting_wealth * 100
        
        agg_tables[regime] = avg_df
        
        # Rank the scenario methods for each optimization method, excluding benchmark
        ranking_df = avg_df.drop(columns=[benchmark], errors='ignore').rank(axis=1, ascending=False, method='min').astype(int)
        
        ranking_tables[regime] = ranking_df
    
    combined_agg_table = pd.concat(agg_tables, names=["Volatility Regime"])
    combined_ranking_table = pd.concat(ranking_tables, names=["Volatility Regime"])
    
    return combined_agg_table, combined_ranking_table
    
    
    
#############################################################
# Functions for Out of Sample Performance                   #
#############################################################

def volatility_evaluation(vxd_value, lower_quantile, upper_quantile):
    """
    Evaluates the volatility type based on the given VXD value and quantiles.

    Parameters:
    vxd_value (float): The VXD value to evaluate.
    lower_quantile (float): The lower quantile threshold.
    upper_quantile (float): The upper quantile threshold.

    Returns:
    str: The type of volatility ('low', 'medium', or 'high').
    """

    if vxd_value < lower_quantile:
        volatility_type = 'low'
    elif vxd_value > upper_quantile:
        volatility_type = 'high'
    else:
        volatility_type = 'medium'
    return volatility_type


def portfolio_rebalancing_test(train_df, test_df, rebalance_freq, volatility_index, opt_method, scenario_method_assignment, test_no_roll, test_roll, INITIAL_CAPITAL=1000, TRANSACTION_COST=0.0005, Months=12, lower_quantile=20, upper_quantile=30):
    """
    Perform portfolio rebalancing test over different rebalancing frequencies.
    This function evaluates the performance of a portfolio rebalancing strategy using different rebalancing frequencies.
    The rebalancing frequencies are provided as a list, but only monthly rebalancing (equivalent to [1]) is used in this implementation.
    Parameters:
    - train_df (pd.DataFrame): Training dataset containing historical asset returns.
    - test_df (pd.DataFrame): Testing dataset containing out of sample asset returns.
    - rebalance_freq (list of int): List of rebalancing frequencies (in integer representations of months).
    - volatility_index (pd.Series): Series containing volatility index values.
    - opt_method (str): Optimization method to be used for portfolio optimization.
    - scenario_method_assignment (dict): Dictionary mapping volatility types to scenario generation methods.
    - test_no_roll (dict): Dictionary containing non-rolling test scenarios from ARMA-GARCH and Copula ARMA-GARCH.
    - test_roll (dict): Dictionary containing rolling test scenarios from ARMA-GARCH and Copula ARMA-GARCH.
    - INITIAL_CAPITAL (float, optional): Initial capital for the portfolio. Default is 1000.
    - TRANSACTION_COST (float, optional): Transaction cost as a fraction of the trade value. Default is 0.0005.
    - Months (int, optional): Number of months for the test period. Default is 12.
    - lower_quantile (int, optional): Lower quantile for volatility evaluation. Default is 20.
    - upper_quantile (int, optional): Upper quantile for volatility evaluation. Default is 30.
    Returns:
    - TW (pd.DataFrame): DataFrame containing terminal wealth for each rebalancing frequency.
    - monthly_wealths (pd.DataFrame): DataFrame containing monthly wealth values for each rebalancing frequency.
    """
    monthly_wealths = pd.DataFrame(index = range(0,Months+1),columns=rebalance_freq, dtype=float)
    monthly_wealths.loc[ 0, :] = INITIAL_CAPITAL
    TW = pd.DataFrame(columns=rebalance_freq, dtype=float)
    for i in rebalance_freq:
        terminal_wealth =  1000
        if i == 12:
            previous_weights = np.zeros(train_combined.shape[1])
            train_df_last_index = train_df.index[-1]
            vxd_value = volatility_index.loc[train_df_last_index]
            volatility_type = volatility_evaluation(vxd_value, lower_quantile, upper_quantile)
            # Compute returns of scenario generation method
            s_method = scenario_method_assignment[volatility_type]
            sg = ScenarioGenerator(train_df, num_scenarios=1000, n_days=len(test_df),random_state= 12)
            # Compute Returns
            if s_method == 'Historical':
                sims_3d = sg.historical_data()
            elif s_method == 'Bootstrap':
                sims_3d = sg.bootstrap()
            elif s_method == 'BlockBootstrap':
                sims_3d = sg.block_bootstrap(block_size=5)
            elif s_method == 'MonteCarloNormal':
                sims_3d = sg.monte_carlo_normal()
            elif s_method == 'MonteCarloTStudent':
                sims_3d = sg.monte_carlo_tstudent()
            elif s_method == 'Opt(ARMA)-GARCH':
                sims_3d = test_no_roll[f'GARCH_{i}_0']
            elif s_method == 'OptCopula-(ARMA)-GARCH':
                sims_3d = test_no_roll[f'Copula_{i}_0']
            elif s_method == 'Roll-Opt(ARMA)-GARCH':
                sims_3d = test_roll[f'GARCH_{i}_0']
            elif s_method == 'Roll-OptCopula-(ARMA)-GARCH':
                sims_3d = test_roll[f'Copula_{i}_0']
            
            # Convert to regular returns
            sims_3d = np.exp(sims_3d) - 1
            # Create the PortfolioOptimizer using the full 3D array
            po = PortfolioOptimizer(sims_3d)
            
            # Here we approximate the risk free (monthly) rate 
            rf_yearly = rf_rate_fetch(train_df_last_index) 
            rf_monthly = np.round((1 + rf_yearly)**(1/12) -1,5)

           # Here we set a target return based on the mean of the training chunk
            expected_return = (1 + train_combined.tail(j).mean().mean())**j -1
            target_ret = max(expected_return, 0.0 )

            try:
                weights = run_optimization(opt_method, po, target_ret, rf_monthly)
            except Exception as e:
                print(f"[WARNING] {opt_method} failed for {s_method} in {volatility_type} conditions: {e}")
                weights = np.ones(train_df.shape[1]) / train_df.shape[1]
            
            
            cum_wealth = evaluate_portfolio_performance(weights, test_df)
                
            # Transaction cost of initial buy
            transaction_cost = TRANSACTION_COST * np.sum(np.abs(weights - previous_weights))
                
            final_wealth_opt = cum_wealth.iloc[-1]
            
            for k in range(1,i):
                intermediate_wealth = cum_wealth.iloc[21*k]
                monthly_wealths.loc[k, i] = intermediate_wealth
            
            # Compute net wealth after transaction costs
            net_return = (1 - transaction_cost) * (1 + final_wealth_opt) - 1
            terminal_wealth = net_return
            monthly_wealths.loc[12, 12] = terminal_wealth
        else:
            for j in range(0, len(test_df), i * 21):
                # Days to train from original training set
                train_0 = train_df.tail(252 - j)
                train_1 = test_df.head(j)
                train_combined = pd.concat([train_0, train_1])
                test = test_df[j+1:j+i*22]
                if j == 0:
                    previous_weights = np.zeros(train_combined.shape[1])                  
                last_index = train_combined.index[-1]  
                vxd_value = volatility_index.loc[last_index]
                volatility_type = volatility_evaluation(vxd_value, lower_quantile, upper_quantile)
                # Compute returns of scenario generation method
                s_method = scenario_method_assignment[volatility_type]
                sg = ScenarioGenerator(train_combined, num_scenarios=1000, n_days=len(test), random_state= 12)
                # Compute Returns
                if s_method == 'Historical':
                    sims_3d = sg.historical_data()
                elif s_method == 'Bootstrap':
                    sims_3d = sg.bootstrap()
                elif s_method == 'BlockBootstrap':
                    sims_3d = sg.block_bootstrap(block_size=5)
                elif s_method == 'MonteCarloNormal':
                    sims_3d = sg.monte_carlo_normal()
                elif s_method == 'MonteCarloTStudent':
                    sims_3d = sg.monte_carlo_tstudent()
                elif s_method == 'Opt(ARMA)-GARCH':
                    sims_3d = test_no_roll[f'GARCH_{i}_{j//21}']
                elif s_method == 'OptCopula-(ARMA)-GARCH':
                    sims_3d = test_no_roll[f'Copula_{i}_{j//21}']
                elif s_method == 'Roll-Opt(ARMA)-GARCH':
                    sims_3d = test_roll[f'GARCH_{i}_{j//21}']
                elif s_method == 'Roll-OptCopula-(ARMA)-GARCH':
                    sims_3d = test_roll[f'Copula_{i}_{j//21}']

                # Convert to regular returns
                sims_3d = np.exp(sims_3d) - 1
                
                # Create the PortfolioOptimizer using the full 3D array
                po = PortfolioOptimizer(sims_3d)

                # Here we set a target return based on the mean of the training chunk
                expected_return = (1 + train_combined.tail(j).mean().mean())**j -1
                target_ret = max(expected_return, 0.0 )
                
                
                # Here we approximate the risk free (for i months) rate 
                months = 12//i
                rf_yearly = rf_rate_fetch(train_combined.index[-1]) 
                rf_monthly = np.round((1 + rf_yearly)**(1/months) -1,5)
                try:
                    weights = run_optimization(opt_method, po, target_ret, rf_monthly)
                except Exception as e:
                    print(f"[WARNING] {opt_method} failed for {s_method} in {volatility_type} conditions: {e}")
                    weights = np.ones(train_combined.shape[1]) / train_combined.shape[1]
                cum_wealth = evaluate_portfolio_performance(weights, test, initial_invest=terminal_wealth)
                
                # Calculate transaction costs
                transaction_cost = TRANSACTION_COST * np.sum(np.abs(weights - previous_weights))
                
                # Calculate cumulative wealth for this period
                final_wealth_opt = cum_wealth.iloc[-1]
                
                for k in range(1,i):
                    intermediate_wealth = cum_wealth.iloc[21*k]
                    monthly_wealths.loc[j//21 + k, i] = intermediate_wealth
                
                # Compute net wealth after transaction costs
                net_return = (1 - transaction_cost) * (1 + final_wealth_opt) - 1
                terminal_wealth = net_return
                monthly_wealths.loc[i + j//21, i] = terminal_wealth
                
                # Rebalance to equal weights
                previous_weights = weights.copy()        
        TW.loc[ f"{opt_method} Opt Port", i] = terminal_wealth
        
    return TW, monthly_wealths


