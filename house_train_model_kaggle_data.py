#TSB  11/26/2017
# Hari Santanam
#Use housing price dataset from kaggle (kings county, Washington) to predict house price 
#based on andrew geitgey's ML class on lynda.com
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

# Load the data set
df = pd.read_csv("/Users/hsantanam/Downloads/kc_house_data.csv")

# Remove the fields from the data set that we don't want to include in our model
del df['id']
del df['date']
del df['zipcode']
del df['lat']
del df['long']

# Replace categorical data with one-hot encoded data
#features_df = pd.get_dummies(df, columns=['garage_type', 'city'])

# Remove the sale price from the feature data
#create y first as we will delete the price column, which is needed for X!
y = df['price'].as_matrix()
del df['price']

# Create the X and y arrays
X = df.as_matrix()
#y = df['price'].as_matrix() - don't need this as already created before deletion in X

# Split the data set in a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fit regression model
model = ensemble.GradientBoostingRegressor(
    n_estimators=1400,        #how many decision trees to build
    learning_rate=0.01,        #how much each additional decision tree influences the overall prediction
    max_depth=6,              #how many layers deep each decision tree is
    min_samples_leaf=9,       # minimum samples must exhibit similar behavior for decision tree to make decision around it
    max_features=0.1,         #percentage of features to choose each time we create branch in decision tree
    loss='huber',             #how scikit calculates model's error rate or cost as it learns
    random_state=0            #used to determine random seed passed to random number generator
)
model.fit(X_train, y_train)   #tell the model to train using our training data

# Save the trained model to a file so we can use it in other programs
joblib.dump(model, '/Users/hsantanam/datascience-projects/kaggle_trained_house_classifier_model.pkl')

# Find the error rate on the training set
mse = mean_absolute_error(y_train, model.predict(X_train))

print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set
mse = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)
