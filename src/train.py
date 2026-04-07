import sklearn
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data # shape (150,4)
y=iris.target # shape(150,)
print(iris.feature_names, iris.target_names)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(random_state=42)

model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Predictions:",y_pred[:5])
print("True labels:",y_test[:5])

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report
print("classification_report")

sklearn.tree.plot_tree(model)

from sklearn.neighbors import KNeighborsClassifier
model_knn=KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train,y_train)
y_pred_knn=model_knn.predict(X_test)
print("K-NN accuracy:",accuracy_score(y_test,y_pred_knn))

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train,y_train)

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
output_dir="outputs"
cm=confusion_matrix(y_test,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(f"{output_dir}/confusion_matrix.png")

import joblib
joblib.dump(model, f"{output_dir}/decision_tree_model.pkl")
joblib.dump(model_knn, f"{output_dir}/knn_model.pkl")




