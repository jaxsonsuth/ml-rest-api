from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib

# Train a RandomForest model on the Iris dataset and save it to disk

#Load Iris dataset from Scikit Learn
data = load_iris()
X, y = data.data, data.target

#Train RandomForest classifier
clf = RandomForestClassifier()
clf.fit(X, y)

#Save trained model to 'app/iris_model.pkl'
joblib.dump(clf, 'app/iris_model.pkl')
