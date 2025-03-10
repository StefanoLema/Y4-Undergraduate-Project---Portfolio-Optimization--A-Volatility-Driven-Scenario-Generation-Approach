###############################################################################
#                               Load Required                                 #
#                                 Packages                                    #
###############################################################################

# General Functionality, basic stats
library(tidyverse)
library(reshape2)
library(psych)
library(reshape2)
library(stats)

# Financial Plots and Data Fetching Functions
library(quantmod)

# ARMA-GARCH
library(rugarch)

# Wavelet
library(wavelets)

# Vine Copula
library(VineCopula)
library(network)
library(Matrix)

# General plotting
library(ggpubr)
library(gridExtra)


###############################################################################
#                              ARMA-GARCH                                     #
#                            Model Selection                                  #
###############################################################################

#' @title Fit a GARCH Model
#' @description Fits a GARCH model to the provided time series data using the specified model specification.
#' @param data A time series dataset to which the GARCH model will be fitted.
#' @param spec A GARCH model specification object created using `ugarchspec`.
#' @return An object representing the fitted GARCH model or NULL if an error occurs.
#' @import rugarch
fit_garch <- function(spec, data) {
  tryCatch(
    {
      ugarchfit(spec, data = data, solver = "hybrid")
    },
    error = function(e) {
      message("Error encountered: ", conditionMessage(e))
      return(NULL)  # Return NULL in case of an error
    },
    warning = function(w) {
      message("Warning encountered: ", conditionMessage(w))
      invokeRestart("muffleWarning")  # Suppress warning
    }
  )
}

#' @title Compute Information Criteria for a GARCH Model
#' @description Fits a GARCH model and computes an information criteria matrix, including AIC, BIC, SIC, and HQIC.
#' @param spec A GARCH specification object created using the `ugarchspec` function.
#' @param data A time series dataset to which the GARCH model will be fitted.
#' @return A matrix containing the values of AIC, BIC, SIC, and HQIC for the fitted model.
#' @import rugarch
model_IC <- function(spec, data) {
  fit <- fit_garch(spec, data)  # Fit the GARCH model to the data
  if (is.null(fit)) return(NULL)  # Return NULL if model fitting fails
  
  info_criteria <- infocriteria(fit)  # Extract IC from the fitted model
  
  # Create a matrix with criteria names and their corresponding values
  criteria_matrix <- matrix(
    c(info_criteria[1, 1], info_criteria[2, 1], info_criteria[3, 1], info_criteria[4, 1]),
    nrow = 4,
    dimnames = list(c("AIC", "BIC", "SIC", "HQIC"), "Value")
  )
  return(criteria_matrix)
}

#' @title Select the Best AR-GARCH Model Based on Information Criteria
#' @description Evaluates different combinations of GARCH models, distributions, and ARMA orders to find the best model based on AIC, BIC, SIC, and HQIC.
#' @param data A time series dataset to be used for model fitting.
#' @param model_list A list of GARCH model types to be tested (e.g., "sGARCH", "eGARCH", "gjrGARCH").
#' @param dist_list A list of distribution types to be tested (e.g., "norm", "std", "sstd").
#' @param arma_list A list of ARMA orders to be tested (e.g., list(c(1,1), c(0,1))).
#' @param eval_func A function to evaluate models, typically `model_IC`.
#' @return A data frame listing the best model for each information criterion (AIC, BIC, SIC, HQIC).
#' @import rugarch
ARGA_model_select <- function(data, model_list, dist_list, arma_list, eval_func) {
  results <- data.frame(Model = character(),
                        Distribution = character(),
                        ARMA = character(),
                        AIC = numeric(),
                        BIC = numeric(),
                        SIC = numeric(),
                        HQIC = numeric(),
                        stringsAsFactors = FALSE)
  
  # Iterate over all model combinations
  for (model in model_list) {
    for (dist in dist_list) {
      for (arma in arma_list) {
        spec <- ugarchspec(
          variance.model = list(model = model, garchOrder = c(1, 1)),
          mean.model = list(armaOrder = arma),
          distribution.model = dist
        )
        
        criteria <- eval_func(spec, data)
        if (!is.null(criteria)) {
          results <- rbind(results, data.frame(
            Model = model,
            Distribution = dist,
            ARMA = paste0("(", toString(arma), ")"),
            AIC = criteria["AIC", "Value"],
            BIC = criteria["BIC", "Value"],
            SIC = criteria["SIC", "Value"],
            HQIC = criteria["HQIC", "Value"]
          ))
        }
      }
    }
  }
  
  best_models <- data.frame(AIC = character(),
                            BIC = character(),
                            SIC = character(),
                            HQIC = character(),
                            stringsAsFactors = FALSE)
  
  # Find the best model for each information criterion
  for (ic in c("AIC", "BIC", "SIC", "HQIC")) {
    best_index <- which.min(results[[ic]])
    best_model <- results[best_index, ]
    best_models[1, ic] <- paste0("ARMA", best_model$ARMA, "-", 
                                 best_model$Model, "(1,1) ", 
                                 best_model$Distribution)
  }
  
  return(best_models)
}

#' @title Compute the Mode of a Vector
#' @description Finds the most frequently occurring value in a vector.
#' @param x A numeric or character vector.
#' @return The mode (most frequent value) of the vector. Returns NA if empty.
Mode <- function(x) {
  x <- na.omit(x)  # Remove NA values
  if (length(x) == 0) return(NA)  # Handle empty vectors
  ux <- unique(x)  # Get unique values
  return(ux[which.max(tabulate(match(x, ux)))])  # Find most frequent value
}

#' Determine the Best Model Based on Information Criteria
#'
#' @description Determines the best model by identifying the most commonly occurring IC value among AIC, BIC, SIC, and HQIC. If no value appears at least three times, BIC is used as the default.
#' @param row A named numeric vector containing AIC, BIC, SIC, and HQIC values.
#' @return The best information criterion value for model selection.
determine_best_ic <- function(row) {
  ic_values <- row[c("AIC", "BIC", "SIC", "HQIC")]
  for (value in ic_values) {
    if (sum(ic_values == value) >= 3) {
      return(value)  # Return if at least 3 criteria agree
    }
  }
  return(row["BIC"])  # Default to BIC if no agreement
}



###############################################################################
#             Classical ARMA-GARCH and Copula ARMA-GARCH Simulations          #
#                 (fixed and rolling) for Training Splits                     #
###############################################################################

#' @title GARCH-Copula Simulation Function
#' @description Simulates future return distributions using GARCH and R-Vine copula models.
#' This function fits GARCH models to stock return data, extracts standardized residuals,
#' and simulates future return distributions using an R-Vine copula model.
#' @param train_list A list of training data frames (historical returns).
#' @param test_list A list of test data frames (out-of-sample returns).
#' @param spec_list A list of GARCH model specifications for each stock.
#' @param volatility_label A string label for output file naming.
#' @param roll A logical value indicating whether rolling forecasts should be used (default is TRUE).
#' @param n_sim The number of simulations to run (default is 1000).
#' @return No explicit return; writes simulation results to CSV files.
#' @export

garch_copula_simulations <- function(train_list, test_list, spec_list, volatility_label, roll = TRUE, n_sim = 1000) {
  set.seed(42) # Ensure reproducibility
  
  # Loop through each period in the training list
  for (lvl in seq_along(train_list)) {
    
    # Extract training and test data for the current period
    returns_train <- data.frame(train_list[[lvl]])
    returns_test <- data.frame(test_list[[lvl]])
    
    # Get dataset dimensions
    n <- nrow(returns_train)
    days <- nrow(returns_test)
    n_stocks <- ncol(returns_train)
    
    # If rolling forecast is enabled, append zero rows for the out-of-sample period
    if (roll) {
      zeros <- returns_test
      zeros[] <- 0
      returns_train <- rbind(returns_train, zeros)
    }
    
    # Initialize arrays to store forecasted returns
    rvine_forecast <- array(0, dim = c(n_sim, days, n_stocks))
    classical_forecast <- array(0, dim = c(n_sim, days, n_stocks))
    
    # Fit GARCH models to each stock
    garch_fits <- list()
    for (i in seq_along(spec_list)) {
      stock_symbol <- symbols_list[i]
      spec <- spec_list[[i]]
      
      # Fit GARCH model with or without rolling window
      fit <- if (roll) {
        ugarchfit(spec, data = returns_train[[stock_symbol]], out.sample = days)
      } else {
        ugarchfit(spec, data = returns_train[[stock_symbol]])
      }
      
      # Handle cases where the model fails to converge
      if (fit@fit$convergence == 1) {
        spec <- ugarchspec(
          variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
          mean.model = list(armaOrder = c(0, 0)),
          distribution.model = "std"
        )
        fit <- if (roll) {
          ugarchfit(spec, data = returns_train[[stock_symbol]], out.sample = days)
        } else {
          ugarchfit(spec, data = returns_train[[stock_symbol]])
        }
      }
      
      garch_fits[[stock_symbol]] <- fit
    }
    
    # Extract standardized residuals from GARCH models
    stand_res <- sapply(garch_fits, residuals, standardize = TRUE)
    shape <- sapply(garch_fits, function(x) coef(x)["shape"])
    
    # Convert residuals into uniform distribution using probability integral transform
    UniformResiduals <- sapply(1:ncol(stand_res), function(i) {
      pdist("std", stand_res[, i], mu = 0, sigma = 1, shape = shape[i])
    })
    
    # Ensure values stay within valid probability range
    UniformResiduals <- pmin(pmax(UniformResiduals, .Machine$double.eps), 0.99999)
    
    # Fit R-Vine Copula to standardized residuals
    RVfit_BIC_Test <- RVineStructureSelect(UniformResiduals, indeptest = TRUE, familyset = NA, selectioncrit = "BIC")
    
    # Generate simulations from the fitted R-Vine Copula
    sim <- t(sapply(1:n_sim, function(j) RVineSim(1, RVfit_BIC_Test)))
    
    # Transform simulated values back to return space
    for (k in 1:n_stocks) {
      rt_residuals <- qdist("std", mu = 0, sigma = 1, sim[, k], shape = shape[k])
      
      # Forecast using GARCH model
      garch_forecast <- if (roll) {
        ugarchforecast(garch_fits[[k]], n.roll = days - 1, n.ahead = 1)
      } else {
        ugarchforecast(garch_fits[[k]], n.ahead = days)
      }
      
      # Forecast using Copula-GARCH approach
      copula_forecast <- if (roll) {
        ugarchforecast(garch_fits[[k]], n.roll = days - 1, n.ahead = 1, 
                       custom.dist = list(name = "sample", distfit = matrix(rt_residuals, ncol = n_sim)))
      } else {
        ugarchforecast(garch_fits[[k]], n.ahead = days, 
                       custom.dist = list(name = "sample", distfit = matrix(rt_residuals, ncol = n_sim)))
      }
      
      # Extract mean and volatility forecasts
      y_hat_garch <- as.numeric(garch_forecast@forecast$seriesFor)
      y_hat_copula <- as.numeric(copula_forecast@forecast$seriesFor)
      sigma_hat_garch <- as.numeric(garch_forecast@forecast$sigmaFor)
      sigma_hat_copula <- as.numeric(copula_forecast@forecast$sigmaFor)
      
      # Generate simulated returns using GARCH and Copula models
      cgarch_forecasts <- sapply(1:days, function(g) rt(n_sim, shape[k]) * sigma_hat_garch[g] + y_hat_garch[g])
      copula_forecasts <- sapply(1:days, function(g) rt_residuals * sigma_hat_copula[g] + y_hat_copula[g])
      
      classical_forecast[, , k] <- cgarch_forecasts
      rvine_forecast[, , k] <- copula_forecasts
    }
    
    # Convert forecast arrays into 2D matrices for output
    classical_forecast_2d <- matrix(classical_forecast, nrow = n_sim * days, ncol = n_stocks, dimnames = list(NULL, symbols_list))
    rvine_forecast_2d <- matrix(rvine_forecast, nrow = n_sim * days, ncol = n_stocks, dimnames = list(NULL, symbols_list))
    
    # Save forecasts to CSV files
    file_prefix <- if (roll) paste0(volatility_label, "Roll/") else paste0(volatility_label, "/")
    write.csv(classical_forecast_2d, file = paste0(file_prefix, "GARCH_Simulations_", volatility_label, "_21_days_", lvl, ".csv"), row.names = FALSE)
    write.csv(rvine_forecast_2d, file = paste0(file_prefix, "Copula_Simulations_", volatility_label, "_21_days_", lvl, ".csv"), row.names = FALSE)
    
    # Print completion message
    print(paste(lvl, "simulations completed for", volatility_label, "volatility"))
  }
}




###############################################################################
#             Classical ARMA-GARCH and Copula ARMA-GARCH Simulations          #
#                 (fixed and rolling) for Testing Split                       #
###############################################################################

#' @title Volatility Evaluation Function
#' @description Determines the volatility regime based on the provided VXD value and selects
#' the appropriate GARCH model specification.
#' @param vxd_value A numeric value representing the market volatility index.
#' @param lower_quantile A numeric value specifying the lower quantile threshold.
#' @param upper_quantile A numeric value specifying the upper quantile threshold.
#' @return A list containing the selected GARCH model specification and volatility type.
volatility_evaluation <- function(vxd_value, lower_quantile, upper_quantile) {
  if (vxd_value < lower_quantile) {
    volatility_type <- "low"
    spec <- spec_list_low
  } else if (vxd_value > upper_quantile) {
    volatility_type <- "high"
    spec <- spec_list_high
  } else {
    volatility_type <- "medium"
    spec <- spec_list_medium
  }
  return(list(spec = spec, volatility_type = volatility_type))
}

#' @title GARCH and Copula Simulations for Scenario Generation
#' @description Performs scenario generation for out-of-sample testing by fitting GARCH models
#' to asset returns, estimating standardized residuals, modeling dependencies using R-vine copulas,
#' and generating simulated return paths.
#' @param returns_train A dataframe containing in-sample asset return time series for training.
#' @param returns_test A dataframe containing out-of-sample asset return time series for validation.
#' @param spec_list A list of GARCH model specifications created using `ugarchspec`.
#' @param symbols_list A vector of asset symbols corresponding to the columns in `returns_train`.
#' @param freq A character string representing the frequency label for scenario outputs.
#' @param iteration An integer specifying the iteration index for rolling forecasts.
#' @param n_sim An integer specifying the number of simulations to generate.
#' @param roll A boolean indicating whether to use rolling window forecasting.
#' @return Saves simulated return scenarios as CSV files.
GARCH_and_Copula_sims_testing <- function(returns_train, returns_test, spec_list, symbols_list, freq = "_", iteration = 0, n_sim = 1000, roll = TRUE) {
  # Number of observations in the training set
  n <- nrow(returns_train)
  
  # Simulation horizon (out-of-sample period length)
  days <- nrow(returns_test)
  
  # Number of assets
  n_stocks <- ncol(returns_train)
  
  # Storage for R-vine and classical GARCH simulations
  rvine_forecast <- array(0, dim = c(n_sim, days, n_stocks))
  classical_forecast <- array(0, dim = c(n_sim, days, n_stocks))
  
  # Append zero rows for rolling forecasts if required
  if (roll == TRUE) {
    returns_train <- rbind(returns_train, matrix(0, nrow = days, ncol = n_stocks))
  }
  
  # Fit GARCH models
  garch_fits <- list()
  for (i in seq_along(spec_list)) {
    stock_symbol <- symbols_list[i]
    spec <- spec_list[[i]]
    fit <- ugarchfit(spec, data = returns_train[[stock_symbol]], out.sample = ifelse(roll, days, 0))
    
    if (fit@fit$convergence == 1) {
      spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                         mean.model = list(armaOrder = c(0, 0)), distribution.model = "std")
      fit <- ugarchfit(spec, data = returns_train[[stock_symbol]], out.sample = ifelse(roll, days, 0))
    }
    
    garch_fits[[stock_symbol]] <- fit
  }
  
  # Extract standardized residuals and transform into uniform residuals
  stand_res <- sapply(garch_fits, residuals, standardize = TRUE)
  shape <- sapply(garch_fits, function(x) coef(x)["shape"])
  UniformResiduals <- sapply(1:ncol(stand_res), function(i) {
    pdist("std", stand_res[, i], mu = 0, sigma = 1, shape = shape[i])
  })
  
  # Fit R-vine copula and generate simulations
  RVfit_BIC_Test <- RVineStructureSelect(UniformResiduals, indeptest = TRUE, familyset = NA, selectioncrit = "BIC")
  sim <- replicate(n_sim, RVineSim(1, RVfit_BIC_Test))
  
  # Save simulations as CSV files
  write.csv(sim, file = paste0("TestSims", ifelse(roll, "Roll", "NoRoll"), "/Copula_Simulations_Testing_RW_Frequency_", freq, "_Iteration_", iteration, ".csv"), row.names = FALSE)
}


###############################################################################
#              Regular Vine Tree Fitting for Plotting Example                 #
###############################################################################

#' @title Regular Vine Matrix fitting to Financial Data
#' @description Fits a Regular Vine (R-vine) copula model to financial return data.
#' It first applies GARCH modeling to capture volatility clustering in asset returns,
#' extracts standardized residuals, transforms them into uniform residuals, and finally
#' fits an R-vine copula structure based on Bayesian Information Criterion (BIC).
#' @param returns A dataframe containing asset return time series.
#' @param spec_list A list of GARCH model specifications created using `ugarchspec`.
#' @return An object representing the fitted R-vine copula model.
fit_tree <- function(returns, spec_list) {
  # Convert returns data into a dataframe
  returns_train <- data.frame(returns)
  
  # Fit GARCH models for each stock
  garch_fits <- list()
  for (i in seq_along(spec_list)) {
    stock_symbol <- symbols_list[i]  # Get stock symbol
    spec <- spec_list[[i]]            # Get GARCH specification
    
    # Fit the GARCH model to the stock returns
    fit <- ugarchfit(spec, data = returns_train[[stock_symbol]])
    
    # If the model fails to converge, re-fit with a simple GARCH(1,1) model
    if (fit@fit$convergence == 1) {
      spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                         mean.model = list(armaOrder = c(0, 0)), distribution.model = "std")
      fit <- ugarchfit(spec, data = returns_train[[stock_symbol]])
    }
    
    # Store the fitted model
    garch_fits[[stock_symbol]] <- fit
  }
  
  # Extract standardized residuals from each GARCH model
  stand_res <- sapply(garch_fits, residuals, standardize = TRUE)
  
  # Extract the 'shape' parameter from each GARCH model
  shape <- sapply(garch_fits, function(x) coef(x)["shape"])
  
  # Convert standardized residuals into uniform residuals using the probability distribution function
  UniformResiduals <- sapply(1:ncol(stand_res), function(i) {
    pdist("std", stand_res[, i], mu = 0, sigma = 1, shape = shape[i])
  })
  
  # Adjust extreme values to keep residuals within valid probability bounds
  if (any(UniformResiduals > 0.99999)) {
    ix <- which(UniformResiduals > 0.99999)
    UniformResiduals[ix] <- 0.99999
  }
  if (any(UniformResiduals < .Machine$double.eps)) {
    ix <- which(UniformResiduals < (1.5 * .Machine$double.eps))
    UniformResiduals[ix] <- .Machine$double.eps
  }
  
  # Fit an R-vine copula model to the transformed residuals using BIC for selection
  RVfit_BIC <- RVineStructureSelect(UniformResiduals, indeptest = TRUE, familyset = NA, selectioncrit = "BIC")
  
  return(RVfit_BIC)
}