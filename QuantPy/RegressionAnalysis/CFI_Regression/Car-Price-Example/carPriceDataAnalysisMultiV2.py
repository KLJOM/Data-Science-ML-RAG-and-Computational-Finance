import pandas
import numpy as np 
import matplotlib.plot as plt 
import statsmodels.api as stats
import seaborn as sns

# 01 - Loading the raw data from the csv file

"""
This would be the equivalent of taking data from FRED Database or the BEA Datbase and loading the 
imported data into the python environment.

""" 
carPrice_df = pd.read_csv("CarPrice_Assignment.csv")

# 02 - Accessing, Inspecting, and Cleaning the Data

"""
This would be the equivalent of taking the loaded data from FRED Database or the BEA Datbase and 
nowprinting the raw data to scan for NaN manually, indicating that there exists in the data frame
empty values for specific variables.

""" 
carPrice_df = pd.read_csv("CarPrice_Assignment.csv")