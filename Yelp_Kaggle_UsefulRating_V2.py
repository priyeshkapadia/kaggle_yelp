# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from textstat.textstat import textstat

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

# Import and Clean Data
# 
# Data files dowloaded from Kaggle: https://www.kaggle.com/c/yelp-recruiting
# 
# The data is in the form of JSON files where each dataset is written as a seperate JSON object per line

review_df        = pd.read_json('yelp_training_set/yelp_training_set_review.json'  , lines=True)
user_df          = pd.read_json('yelp_training_set/yelp_training_set_user.json'    , lines=True)
business_df      = pd.read_json('yelp_training_set/yelp_training_set_business.json', lines=True)
checkin_df       = pd.read_json('yelp_training_set/yelp_training_set_checkin.json' , lines=True)
review_test_df   = pd.read_json('yelp_test_set/yelp_test_set_review.json'          , lines=True)
user_test_df     = pd.read_json('yelp_test_set/yelp_test_set_user.json'            , lines=True)
business_test_df = pd.read_json('yelp_test_set/yelp_test_set_business.json'        , lines=True)
checkin_test_df  = pd.read_json('yelp_test_set/yelp_test_set_checkin.json'         , lines=True)

print('Kaggle Training Set:')
print('\tNo of reviews:'   , len(review_df))
print('\tNo of users:'     , len(user_df))
print('\tNo of businesses:', len(business_df))
print('\tNo of checkins:'  , len(business_df))
print('Kaggle Test Set:')
print('\tNo of reviews:'   , len(review_test_df))
print('\tNo of users:'     , len(user_test_df))
print('\tNo of businesses:', len(business_test_df))
print('\tNo of checkins:'  , len(checkin_test_df))

# Functions for Cleaning Data
# 
# `extract_useful_votes` - returns no of useful votes from the dictionary `'votes'`
# `extract_funny_votes` - returns no of funny votes from the dictionary `'votes'`
# `extract_cool_votes` - returns no of cool votes from the dictionary `'votes'`
# `find_days_since_review` - returns the difference (in days) between the review date with the date in arguement `today_date`
# `find_votes_useful_per_day` - returns average useful votes per day
# `extract_clean_text` - returns a text string where `\n` are removed from the string in the `text` field
# `find_review_length` - returns the length of the string `clean_text`
# `find_readability` - returns the flesch reading ease score in the string `clean_text` using the `textstat` module
# `find_total_checkins` - returns total number of checkins for each business

def extract_useful_votes(row):
    return row['votes']['useful']
def extract_funny_votes(row):
    return row['votes']['funny']
def extract_cool_votes(row):
    return row['votes']['cool']
# find and return the number of days between current date and date of review
def find_days_since_review(row, today_date):
    review_date = row['date']
    today_date  = datetime.datetime.strptime(today_date_str, "%Y-%m-%d")
    return (today_date - review_date).days
def find_votes_useful_per_day(row):
    return row['review_useful_votes']/row['days_since_review']
def extract_clean_text(row):
    clean_text = row['text']
    clean_text.replace("\n", " ")
    return clean_text
def find_review_length(row):
    return len(row['clean_text'])
def find_readability(row):
    try:
        readability = textstat.flesch_reading_ease(row['clean_text'])
    except:
        print("Error finding readability for review id: ", row['review_id'])
        readability = 0
    return readability
def find_total_checkins(row):
    return np.array(list(row['checkin_info'].values())).sum()

review_df['review_useful_votes'] = review_df.apply(extract_useful_votes, axis=1)
review_df['review_funny_votes']  = review_df.apply(extract_funny_votes , axis=1)
review_df['review_cool_votes']   = review_df.apply(extract_cool_votes  , axis=1)
user_df['user_useful_votes']     = user_df.apply(extract_useful_votes  , axis=1)
user_df['user_funny_votes']      = user_df.apply(extract_funny_votes   , axis=1)
user_df['user_cool_votes']       = user_df.apply(extract_cool_votes    , axis=1)
review_df['clean_text']          = review_df.apply(extract_clean_text  , axis=1)
review_df['review_length']       = review_df.apply(find_review_length  , axis=1)
review_df['review_readability']  = review_df.apply(find_readability    , axis=1)
checkin_df['total_checkins']     = checkin_df.apply(find_total_checkins, axis=1)

today_date_str = '2013-01-19'
today_date = datetime.datetime.strptime(today_date_str, "%Y-%m-%d")
review_df['days_since_review']    = review_df.apply(find_days_since_review, args=(today_date,), axis=1)
review_df['votes_useful_per_day'] = review_df.apply(find_votes_useful_per_day, axis=1)

# Merge review data frame with user and business data frames

merged_df = pd.merge(review_df, user_df, on='user_id', how='left')
merged_df = pd.merge(merged_df, business_df, on='business_id', how='left')
merged_df = pd.merge(merged_df, checkin_df, on='business_id', how='left')

# **Assumption** - Replace missing data values with the mean value of the feature

for feature in ['average_stars', 'review_count_x', 'user_useful_votes', 'user_funny_votes', 'user_cool_votes', 'total_checkins']:
    mean = merged_df[feature].mean()
    merged_df[feature] = merged_df[feature].fillna(mean)

# Fit Model
# Use Random Forests to predict number of useful votes for the review
featuresList = ['stars_x', 'days_since_review', 'review_length',
       'review_readability', 'average_stars', 'review_count_x',
       'user_useful_votes', 'user_funny_votes', 'user_cool_votes',
       'latitude', 'longitude', 'stars_y']
X_df = merged_df[featuresList]
y_df = merged_df['review_useful_votes']
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_df, y_df, test_size=0.3, random_state=0)
print('Model Fitting...')
print('\tNo of Training Data:', len(X_train_df))
print('\tNo of Test Data:'    , len(X_test_df))

# Use `GridSearchCV` to find optimum parameters for `RandomForestRegressor()` model
param_grid_RF = {
    'max_features': ['sqrt', 'log2', None],
    'n_estimators': [10, 100, 500],
    'min_samples_leaf': [1, 5, 10, 20, 40]
}

param_grid_GBT = {
    'learning_rate': [0.01, 0.1, 1],
	'max_features': ['sqrt', 'log2', None],
    'n_estimators': [10, 100, 500],
    'min_samples_leaf': [1, 5, 10, 20, 40]
}

for name, model, param_grid in [['RF' , RandomForestRegressor()    , param_grid_RF ],
								['GBT', GradientBoostingRegressor(), param_grid_GBT],]:
	print('Fitting {0} model'.format(name))
	grid_search_model = GridSearchCV(estimator=model,
									 param_grid=param_grid,
									 refit=True)
	grid_search_model.fit(X_train_df, y_train_df)
	print(grid_search_model.best_estimator_)
	y_train_pred = grid_search_model.predict(X_train_df)
	y_pred       = grid_search_model.predict(X_test_df)
	print('Training Set:')
	print('\tThe mean absolute error is: {0:.3f}'.format(mean_absolute_error(list(y_train_pred), list(y_train_df))))
	print('\tThe root mean squared error is: {0:.3f}'.format(mean_squared_error(list(y_train_pred), list(y_train_df))**0.5))
	print('\tThe root mean squared logarithmic error is: {0:.3f}'.format(mean_squared_log_error(list(y_train_pred), list(y_train_df))**0.5))
	print('Test Set:')
	print('\tThe mean absolute error is: {0:.3f}'.format(mean_absolute_error(list(y_pred), list(y_test_df))))
	print('\tThe root mean squared error is: {0:.3f}'.format(mean_squared_error(list(y_pred), list(y_test_df))**0.5))
	print('\tThe root mean squared logarithmic error is: {0:.3f}'.format(mean_squared_log_error(list(y_pred), list(y_test_df))**0.5))

	features_importances = list(zip(X_df[featuresList].columns, grid_search_model.best_estimator_.feature_importances_))
	features_importances.sort(key=lambda x:x[1], reverse=True)
	print_df = pd.DataFrame(np.array(features_importances))
	print_df.to_csv(name+"_features_importances.csv", index=False, header=False)

	# Plot predictions vs actual data
	fig = plt.figure()
	plt.scatter(list(y_pred), list(y_test_df))
	plt.plot(list(y_test_df), list(y_test_df), c='r')
	plt.xlabel('Useful Votes - Prediction')
	plt.ylabel('Useful Votes - Actual')
	plt.savefig(name+'_PredictionVsActual.png')

	# Predict Kaggle Test Set
	review_test_df['clean_text']          = review_test_df.apply(extract_clean_text  , axis=1)
	review_test_df['review_length']       = review_test_df.apply(find_review_length  , axis=1)
	review_test_df['review_readability']  = review_test_df.apply(find_readability    , axis=1)
	checkin_test_df['total_checkins']     = checkin_test_df.apply(find_total_checkins, axis=1)

	today_date_str = '2013-03-12'
	today_date = datetime.datetime.strptime(today_date_str, "%Y-%m-%d")
	review_test_df['days_since_review']    = review_test_df.apply(find_days_since_review, args=(today_date,), axis=1)

	# Merge review data frame with user and business data frames
	user_test2_df = pd.concat([user_df, user_test_df])
	business_test2_df = pd.concat([business_df, business_test_df])
	checkin_test2_df = pd.concat([checkin_df, checkin_test_df])
	merged_test_df = pd.merge(review_test_df, user_test2_df, on='user_id', how='left')
	merged_test_df = pd.merge(merged_test_df, business_test2_df, on='business_id', how='left')
	merged_test_df = pd.merge(merged_test_df, checkin_test2_df, on='business_id', how='left')

	for feature in ['average_stars', 'review_count_x', 'user_useful_votes', 'user_funny_votes', 'user_cool_votes', 'total_checkins']:
		mean = merged_test_df[feature].mean()
		merged_test_df[feature] = merged_test_df[feature].fillna(mean)

	X_kaggle_test_df = merged_test_df[featuresList]
	y_kaggle_pred = grid_search_model.predict(X_kaggle_test_df)
	kaggle_pred_df = pd.DataFrame(list(zip(merged_test_df['review_id'], y_kaggle_pred)), columns=['Id', 'Votes'])
	kaggle_pred_df.to_csv(name+"_kaggle_predictions.csv", index=False, header=True)
