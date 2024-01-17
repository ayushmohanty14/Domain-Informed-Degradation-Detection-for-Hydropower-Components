import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import random

# Load the sampled sensor data from the CSV file into a pandas DataFrame
raw_data = pd.read_csv("DailySampled_denoised.csv")
raw_data = raw_data.fillna(0)
data = raw_data #np.log(raw_data)
# data.to_csv("log_data.csv", index=False)


# Parameters
num_cols = len(data.columns)
TrainingSize = num_cols - 1
TestSize = 1
pred_horizon = 2*len(data)
num_inspections = len(data)
cycles_for_training = 747
failure_threshold = 16
RUL_bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Initialize the arrays
all_cols = data.columns.values.tolist()
true_failure = np.zeros([num_cols,])
logSignal_TrData = np.empty([num_inspections, TrainingSize])
logSignal_TestData = np.empty([num_inspections,])
L = logSignal_TestData
Coeff_regression = np.empty([cycles_for_training, 2])
X_k = np.empty([(num_inspections - 1), TrainingSize])
RLD_cdf = np.zeros([num_cols, num_inspections, pred_horizon])
RLD_pdf = np.zeros([num_cols, num_inspections, pred_horizon])

# Bayesian Updating of the Exponential Model
def BayesUpdate(mu0, mu1, var0, var1, var_diff, L):
    num_mu_theta_new = L[0]*var0*var1*(pred_horizon - 1) + var_diff
    den_mu_theta_new = var0 *(var1*(pred_horizon - 1) + var_diff)
    mu_theta_new = num_mu_theta_new/den_mu_theta_new

    num_mu_beta_new = (var1*np.sum(L[1:]) + mu1*var_diff)*var0 - var1*L[0]*var0
    den_mu_beta_new = var0 *(var1*(pred_horizon - 1) + var_diff)
    mu_beta_new = num_mu_beta_new/den_mu_beta_new

    num_var_theta_new = var_diff*var0*(var1*(pred_horizon - 1) + var_diff)
    den_var_theta_new = var0 *(var1*(pred_horizon - 1) + var_diff)
    var_theta_new = num_var_theta_new/den_var_theta_new

    num_var_beta_new = var_diff*var1*var0
    den_var_beta_new = var0 *(var1*(pred_horizon - 1) + var_diff)
    var_beta_new = num_var_beta_new/den_var_beta_new

    return mu_beta_new, var_beta_new, mu_theta_new, var_theta_new

## PDF of an Inverse Gaussian Distribution
def invgauss_pdf(t, mu_invGauss, shape_invGauss):

    PDF = np.sqrt(shape_invGauss/(2*math.pi*(t**3)))*np.exp(-((shape_invGauss/(2*mu_invGauss**2))*(((t - mu_invGauss)**2)/t)))

    return PDF

# Main loop
count_sensor = 0
for i_col in all_cols:
    data_train = data.loc[:, ~data.columns.isin([i_col])]

    theta_est = np.empty([TrainingSize, 1])
    beta_est = np.empty([TrainingSize, 1])

    logSignal_TrData[:, :] = data_train.to_numpy()
    logSignal_TestData = data[i_col].to_numpy()
    L[0] = logSignal_TestData[0]
    L = np.diff(logSignal_TestData)
    for idx_train in range(TrainingSize):
        X_k[:, idx_train] = np.diff(logSignal_TrData[:, idx_train])
        Coeff_regression[:, 0] = logSignal_TrData[1:cycles_for_training + 1, idx_train]
        # Coeff_regression[:, 1] = np.ones(1509)
        # beta_est[idx_train], theta_est[idx_train] = np.linalg.lstsq(Coeff_regression, X_k[:1510, idx_train], rcond=None)[0]
        theta_est[idx_train], beta_est[idx_train] = np.polynomial.polynomial.polyfit(Coeff_regression[:, 0], X_k[:cycles_for_training, idx_train], deg=1)

    mu0 = np.mean(theta_est)
    var0 = np.var(theta_est)

    mu1 = np.mean(beta_est)
    var1 = np.var(beta_est)

    mean_prior = [mu0, mu1]
    var_prior = [var0, var1]

    error_increment = np.zeros([num_inspections - 1, TrainingSize])
    BM_par_allTr = np.empty([num_inspections - 1, ])

    for idx_train in range(TrainingSize):
        for t in range(1, num_inspections - 1):
            error_increment[t, idx_train] = logSignal_TrData[t, idx_train] - logSignal_TrData[t-1, idx_train] - beta_est[idx_train]

    sample_var = np.var(error_increment, axis=1)
    BM_par_allTr = np.sqrt(sample_var)
    BM_par = np.mean(BM_par_allTr[0:cycles_for_training])
    var_diff = BM_par**2

    for count in range(num_inspections):
        if logSignal_TestData[count] >= failure_threshold:
            time_threshold = count
            break

    true_failure[count_sensor] = time_threshold
    result = [i*time_threshold for i in RUL_bins]
    point_of_update = [math.floor(i) for i in result]

    mu_theta = np.empty([len(point_of_update) + 1, ])
    mu_beta = np.empty([len(point_of_update) + 1, ])
    var_theta = np.empty([len(point_of_update) + 1, ])
    var_beta = np.empty([len(point_of_update) + 1, ])

    mu_theta[0] = mu0
    mu_beta[0] = mu1
    var_theta[0] = var0
    var_beta[0] = var1

    count = 0
    for r_k in point_of_update:
        mu_beta_new, var_beta_new, mu_theta_new, var_theta_new = BayesUpdate(mu0, mu1, var0, var1, var_diff, L)
        # print(mu_beta_new, var_beta_new, mu_theta_new, var_theta_new)
        mu0 = mu_theta_new
        var0 = var_theta_new

        mu1 = mu_beta_new
        var1 = var_beta_new

        mu_theta[count + 1] = mu_theta_new
        mu_beta[count + 1] = mu_beta_new
        var_theta[count + 1] = var_theta_new
        var_beta[count + 1] = var_beta_new

        cdf = np.zeros([pred_horizon - r_k])
        pdf = np.zeros([pred_horizon - r_k])

        # for t2 in range(1, pred_horizon - r_k):
        #     mu = logSignal_TestData[r_k] + mu_beta[count + 1]*t2
        #     var = var_beta[count + 1]*(t2**2) + var_diff*t2
        #     g_t = (mu - failure_threshold)/np.sqrt(var)
        #     cdf[t2] = norm.cdf(g_t, loc=0, scale=1)


        for t2 in range(pred_horizon - r_k):
            if t2 == 0:
                zeta = failure_threshold - mu_theta[0]
                Lambda = mu_beta[0]
            else:
                zeta = failure_threshold - logSignal_TestData[r_k]
                Lambda = mu_beta[count + 1]

            mu = zeta/Lambda
            shape = (zeta / np.sqrt(var_diff)) ** 2
            pdf[t2] = invgauss_pdf(t2 + 1, mu, shape)
            if t2 >= 1:
                cdf[t2] = cdf[t2 - 1] + pdf[t2]

        RLD_cdf[count_sensor, count, r_k:] = cdf
        RLD_pdf[count_sensor, count, r_k:] = pdf

        # if count_sensor == 1:
        #     plt.figure()
        #     plt.plot(pdf)
        #     plt.title("Update Point:" + str(RUL_bins[count]*100) + "% of failure time")
        #     plt.xlim([0, 1000])
        #     plt.grid()
        #     plt.show()

        # print(cdf)

        count += 1
    # print("\nSample Number:", count_sensor)
    # print("mu_beta:", mu_beta)
    # print("var_beta:", var_beta)
    # print("mu_theta:", mu_theta)
    # print("var_theta:", var_theta)
    count_sensor += 1


print("True Failure:", true_failure)

# Predicting median of failure time distribution
percentile_95 = np.empty([num_cols, len(RUL_bins)])
percentile_50 = np.empty([num_cols, len(RUL_bins)])
percentile_05 = np.empty([num_cols, len(RUL_bins)])
true_failure = np.empty([num_cols])

for i in range(num_cols):
    for j in range(len(RUL_bins)):
        for t in range(pred_horizon):
            if RLD_cdf[i, j, t] >= 0.95:
                percentile_95[i, j] = t
                break

for i in range(num_cols):
    for j in range(len(RUL_bins)):
        for t in range(pred_horizon):
            if RLD_cdf[i, j, t] >= 0.5:
                percentile_50[i, j] = t
                break

for i in range(num_cols):
    for j in range(len(RUL_bins)):
        for t in range(pred_horizon):
            if RLD_cdf[i, j, t] >= 0.05:
                percentile_05[i, j] = t
                break

for i in range(num_cols):
    print("\nSample Number:", i + 1)
    print("True Failure:", int(true_failure[i]))
    print("50%:", percentile_50[i, :])

true = pd.DataFrame(np.transpose(true_failure))
true.to_csv("true_failure.csv", index=False)

results = pd.DataFrame(percentile_50)
results.to_csv("prediction_results.csv", index=False)

# Relative Prediction Error
RPE = np.empty([num_cols, len(RUL_bins)])
for i in range(num_cols):
    for j in range(len(RUL_bins)):
        RPE[i, j] = 100 * (abs(percentile_50[i, j] - true_failure[i])/true_failure[i])

RPE = pd.DataFrame(np.transpose(RPE))
RPE.to_csv("pred_error.csv", index=False)

# Plotting

logSignal_TestData = data.iloc[:,7].to_numpy()

for count in range(num_inspections):
    if logSignal_TestData[count] >= failure_threshold:
        time_threshold = count
        break

true_failure[7] = time_threshold
result = [i * time_threshold for i in RUL_bins]
point_of_update = [math.floor(i) for i in result]

i = 0
for r_k in point_of_update:
    plt.figure()
    plt.plot(RLD_pdf[7, i, r_k:], color="black")
    plt.axvline(percentile_50[7, i], color="blue", linestyle="--")
    plt.axvline(true_failure[7], color="red")
    plt.legend(["Lifetime Distribution", "Predicted Failure", "True Failure"])
    plt.title("Update Point:" + str(RUL_bins[i]*100) + "% of failure time")
    plt.xlim([0, 2000])
    plt.xlabel("Number of Operational Days")
    plt.ylabel("Failure Probability")
    plt.grid()
    plt.show()
    i += 1
