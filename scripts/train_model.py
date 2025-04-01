from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib


data = load_iris()
X, y = data.data, data.target
clf = RandomForestClassifier()
clf.fit(X, y)

joblib.dump(clf, 'app/iris_model.pkl')
