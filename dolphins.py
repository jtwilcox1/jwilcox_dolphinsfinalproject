# # File created by JT Wilcox 
# sources
# https://www.scaler.com/topics/pandas/how-to-install-pandas-in-python/
# https://automatetheboringstuff.com/
# https://www.activestate.com/blog/how-to-predict-nfl-winners-with-python/
# https://towardsdatascience.com/sports-reference-api-intro-dbce09e89e52
# https://stackabuse.com/linear-regression-in-python-with-scikit-learn/
# https://kidscancode.org/

# the pandas library helps analyze, clean, explore, and manipulate data
import pandas as pd
# LinearRegression is a class that is used to predict a variable based on a different variable
from sklearn.linear_model import LinearRegression
# this is the data from the past 4 dolphins seasons starting with year, wins, then losses
# these are tuples
pastseason_data = [
    (2019, 5, 11), 
    (2020, 10, 6),
    (2021, 9, 8),
    (2022, 9, 8),
]
# empty lists that will store the data for the season, wins, and losses
# these empty lists help form the dataset for the linear regression model to make the prediction
years = []
wins = []
losses = []

# to append means to add an item to a single collection type 
# so what this is really doing is grouping the year with its respective wins and losses together
for season, win, loss in pastseason_data:
    years.append(season)
    wins.append(win)
    losses.append(loss)

# this organizes the data into a list of the year, wins, and losses
data = {'Season': years, 'Wins': wins, 'Losses': losses}

# this takes the data from the previous line and makes it a DataFrame
# a DataFrame is a column that contains the year with the wins and losses
dataframe = pd.DataFrame(data)

# the X_trains represent the years in the dataframe
# reshape (-1,1) is necessary because sklearn is normally 2 columns but in this case we only need one
X_train = dataframe['Season'].values.reshape(-1, 1)
# Y_train represents the number of wins in the dataframe
Y_train = dataframe['Wins'].values

# instance of the class
model = LinearRegression()
# trains the linear regressions model by creating a relationship between X_train and Y_train
model.fit(X_train, Y_train)

# the now trained model has to make a prediction for the 2023 season
year_2023 = [[2023]]
predicted_wins = model.predict(year_2023)

# print the string and the predicted win value
print("The 2023-2024 Miami Dolphins will win", predicted_wins[0])


   





