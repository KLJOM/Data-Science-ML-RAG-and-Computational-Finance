import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as stats
import seaborn as sns

carPrice_df = pd.read_csv("CarPrice_Assignment.csv")

"""
In this example we split the data into training and testing data known as the Test/Train split. 
"""

# No. rows and columns in the data frame

print(carPrice_df.shape)

# first five rows of the data frame 

carPrice_df.head()

# full data frame

carPrice_df.to_string()

# descriptive statistics of the data frame 

print(carPrice_df.describe())

# defining the variables

X_carPrice = carPrice_df.enginesize
y_carPrice = stats.add_constant(carPrice_df.price)

# defining the variables for the training and testing data (which is exlcusive of the training data)

train_df = carPrice_df.sample(frac = 0.7, random_state = 99)
test_df = carPrice_df.drop(train_df.index)
print(train_df.shape)
print(test_df.shape)

# 01 visualising the training data on a scatter plot without yet fitting the model

plt.scatter(train_df.enginesize, train_df.price)
plt.title("Engine Size vs Price")
plt.xlabel("Engine Size")
plt.ylabel("Price ($)")
plt.show()

# 01 FITTING the model to the training data set

y_train = train_df.price
x_train = stats.add_constant(train_df.enginesize)

train_model_carprice = stats.OLS(y_train, x_train)
train_results_carprice = train_model_carprice.fit() # model being fitted to the training data set with the least squares method 

print(train_results_carprice.summary())

alpha_carprice = train_results_carprice.params.iloc[0] # THE INTERCEPT IN OUR EQUATION
beta_carprice = train_results_carprice.params.iloc[1] # THE SLOPE IN OUR EQUATION
ssr_carprice = train_results_carprice.ssr # THE SUM OF SQUARED RESIDUALS IN OUR EQUATION

# 01 PRINTING THE VALUES OF THE INTERCEPT, SLOPE AND SUM OF SQUARED RESIDUALS FOR THE TRAINING DATA SET

print(f"The y-int is: {alpha_carprice:3f}")
print(f"The slope/co-efficient of Years of Experience is: {beta_carprice:3f}")
print(f"The Sum of Squared Residuals is: {ssr_carprice:1f}")

# 01 PLOTTING THE OBSERVED VALUES AND THE LINE OF BEST FIT

plt.scatter(train_df.enginesize, train_df.price, label = "Observed", color = "navy")

train_syntheticX = np.linspace(0, train_df.enginesize.max(), 50)
train_predictedY = alpha_carprice + beta_carprice * train_syntheticX

plt.plot(train_syntheticX, train_predictedY, color = "red", ls = "--", label = "Model")
plt.title("Engine Size vs Price")
plt.xlabel("Engine Size")
plt.ylabel("Price ($)")
plt.legend()
plt.show()

# 02 Fitting the model to the testing data set 

y_test = test_df.price # THIS IS THE OBSERVED PRICE VALUE
x_test = stats.add_constant(test_df.enginesize)
# The following is the PREDICTED PRICE VALUES BASED ON THE TEST DATA SET
test_pred = train_results_carprice.predict(x_test) # This is the predicted values of the testing data set that lies on the line of best fit given by the training data set

plt.scatter(test_pred, y_test)
plt.plot([5000, 50000], [5000, 50000], c = 'green', ls = '--')
plt.title("Predicted Price vs Observed Price")
plt.xlabel("Predicted Price")
plt.ylabel("Observed Price")
plt.show()

# Using Seaborn to create a heatmap of the correlation matrix of the data frame

corrCarPrice = carPrice_df.select_dtypes(include=[np.number]).corr()
corrCarPriceHeatmap = sns.heatmap(
    corrCarPrice,
    vmin = -1, vmax = 1, center = 0,
    cmap = sns.diverging_palette(50, 500, n = 500),
    square = True,

)
plt.title("Correlation Matrix of Car Price Data")
plt.xlabel("Features")
plt.ylabel("Features")
plt.show()

