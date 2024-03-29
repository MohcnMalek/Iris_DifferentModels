import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_decision_tree(X, y):
    # Train Decision Tree classifier
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X, y)
    return decision_tree

def train_knn(X, y):
    # Train K-Nearest Neighbors (KNN) classifier
    knn = KNeighborsClassifier()
    knn.fit(X, y)
    return knn

def train_random_forest(X, y):
    # Train Random Forest classifier
    random_forest = RandomForestClassifier()
    random_forest.fit(X, y)
    return random_forest

def train_svm(X, y):
    # Train Support Vector Machine (SVM) classifier
    svm = SVC()
    svm.fit(X, y)
    return svm

def save_model(model, filename):
    # Save the trained model using pickle
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

# Load the Iris dataset
df = pd.read_csv('iris.data')

X = np.array(df.iloc[:, 0:4])
y = np.array(df.iloc[:, 4])

le = LabelEncoder()
y = le.fit_transform(y.reshape(-1))
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and save Decision Tree model
decision_tree_model = train_decision_tree(X_train, y_train)
save_model(decision_tree_model, 'decision_tree_model.pkl')

# Train and save KNN model
knn_model = train_knn(X_train, y_train)
save_model(knn_model, 'knn_model.pkl')

# Train and save Random Forest model
random_forest_model = train_random_forest(X_train, y_train)
save_model(random_forest_model, 'random_forest_model.pkl')

# Train and save SVM model
svm_model = train_svm(X_train, y_train)
save_model(svm_model, 'svm_model.pkl')