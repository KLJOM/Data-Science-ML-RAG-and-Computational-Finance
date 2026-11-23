import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as stats

salaryData = pd.read_csv('Salary_Data.csv')

# printing information about the data and the data its self

print(salaryData.shape) # number of rows and columns in the data frame
print(salaryData.info()) # information about the data frame and the data types as well as null values inside the data frame
print(salaryData.head()) # first 5 rows of the data frame

# printing the data frame
print(salaryData.to_string()) # printing the exhaustive data frame

# printing the descriptive statistics on the data 
print(salaryData.describe())

# fitting the linear regression model
Y_salaryData = salaryData.Salary # Defining the Target Variable of the observed values
X_salaryData = stats.add_constant(salaryData.YearsExperience) # Defining the Independent Variable of the observed values + forces the model to have a value for the intercept

# TRAINING THE MODEL
model_SalaryData = stats.OLS(Y_salaryData, X_salaryData) # Minimizing the sum of the squares of the residuals - summarising the fitted model
results_SalaryData = model_SalaryData.fit() # fitting the line of best fit to the data for the least squares regression line
print(results_SalaryData.summary()) # printing the summary of the model

alpha_SalaryData = results_SalaryData.params[0] # intercept
beta_SalaryData = results_SalaryData.params[1] # slope
ssr_SalaryData = results_SalaryData.ssr # sum of squares of the residuals

print(f"The y-int is: {alpha_SalaryData:3f}")
print(f"The slope/co-efficient of Years of Experience is: {beta_SalaryData:3f}")
print(f"The Sum of Squared Residuals is: {ssr_SalaryData:1f}")

# PLOTTING THE OBSERVATIONS IN THE DATA FRAME CREATED BY USING READ_CSV TO READ THE CSV FILE
plt.scatter(salaryData.YearsExperience, salaryData.Salary, label = 'Observed', color = "cyan")
# line of best fit
salary_SyntheticX = np.linspace(0, salaryData.YearsExperience.max(), 50)
salary_PredictedY = alpha_SalaryData + beta_SalaryData * salary_SyntheticX

plt.plot(salary_SyntheticX, salary_PredictedY, color = "red", ls = '--', label = "Model")
plt.title("Years of Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary [$]")
plt.legend()
plt.show()



