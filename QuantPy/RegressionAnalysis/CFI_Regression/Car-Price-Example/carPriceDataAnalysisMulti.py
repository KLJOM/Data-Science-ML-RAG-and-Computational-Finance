import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as stats
import seaborn as sns 

# Loading the data

carPrice_df = pd.read_csv("CarPrice_Assignment.csv")

# Asessing the data 

print(carPrice_df.shape)
print(carPrice_df.info())
print(carPrice_df.head())
print(carPrice_df.describe())

# to check unique data types in the categorical variables 

checkUnique = carPrice_df.select_dtypes(include = "object").nunique()
print(checkUnique)

# we remove Car_ID and CarName (large number of unique values) from the data frame as it is not relevant to the analysis

carprice_df = carPrice_df.drop(columns = ['car_ID', 'CarName'])
print(carprice_df.head())

#------------------------------------------ A) BASIC DATA PREPARATION & MODEL CONSTRUCTION ------------------------------------------

# Defining the target variable and the feature variables 
Y_basic = carprice_df.price
X_basic = stats.add_constant(carprice_df[['enginesize','curbweight','peakrpm', 'citympg']])

# Fitting the model to the training data set
model_basic = stats.OLS(Y_basic, X_basic)
results_basic = model_basic.fit()

print(results_basic.summary())

# using a correlation matrix to check for multicollinearity and removing the feature variables that are highly correlated
# These would cause problems with the algorithm
correlationMatrix = carprice_df.select_dtypes(exclude = "object").corr()
print(correlationMatrix)

# Visualizing the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlationMatrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

# removing the feature variables that are highly correlated

carprice_df = carprice_df.drop(columns = ['carlength', 'carwidth', 'highwaympg'])

# ONE HOT ENCODING THE CATEGORICAL VARIABLES
'''
This section of code takes the columns which have text values, it then takes the text value for each row in the column - turns it into a number
and then turns it into a binary value (0 or 1) - this is known as one hot encoding. Turning it into binary values makes computing for the computer more efficient.
This makes the data suitable for the model to be fitted to the training data set.
'''

dummy = pd.get_dummies(carprice_df.select_dtypes(include = 'object'), drop_first = True, dtype = int) 

carprice_df = pd.concat([carprice_df.select_dtypes(exclude = 'object'), dummy], axis = 1)
print(carprice_df.head())

correlationMatrix = carprice_df.select_dtypes(exclude = "object").corr() # correlation matrix of the data frame to find and remove multicolinearity
print(correlationMatrix)

# Visualizing the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlationMatrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

# removing the feature variables that are highly correlated
'''
By reducing the number of independent variables, we can reduce the multicollinearity in the data frame and hence prevent innacurate predictions. 
'''
carprice_df = carprice_df.drop(columns = ['compressionratio', 'drivewheel_fwd', 'enginetype_rotor', 'fuelsystem_4bbl', 'fuelsystem_idi'])

#------------------------------------------ A) FITTING THE DATA TO THE MODEL (TRAINING & TESTING DATA SETS) ------------------------------------------

train_df = carprice_df.sample(frac = 0.7, random_state = 99)
test_df = carprice_df.drop(train_df.index)

print(train_df.shape)
print(test_df.shape)

Y_train = train_df.price
X_train = stats.add_constant(train_df.drop('price', axis = 1))

train_model_carprice = stats.OLS(Y_train, X_train)
train_results_carprice = train_model_carprice.fit()

print(train_results_carprice.summary())

print(f"The Sum of Squared Residuals is: {train_results_carprice.ssr}")

X_test = stats.add_constant(test_df.drop('price', axis = 1))

test_predictions = train_results_carprice.predict(X_test) # creates the predicted values for the testing data set based on the remaining columns (features) of the training data set

plt.scatter(test_predictions, Y_test) # plots the scatter for observed prices against predicted prices
plt.plot([5000, 50000], [5000, 50000], c = 'k', ls = '--')
plt.title("Observed vs Predicted Prices")
plt.xlabel("Predicted Prices")
plt.ylabel("Observed Prices")
plt.show()
