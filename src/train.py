import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

iris=load_iris()
X=iris.data # shape (150,4)
y=iris.target # shape(150,)
print(iris.feature_names, iris.target_names)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=DecisionTreeClassifier(random_state=42)

model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Predictions:",y_pred[:5])
print("True labels:",y_test[:5])

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

confusion_matrix(y_test,y_pred)

print("classification_report")

sklearn.tree.plot_tree(model)

model_knn=KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train,y_train)
y_pred_knn=model_knn.predict(X_test)
print("K-NN accuracy:",accuracy_score(y_test,y_pred_knn))

model=DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train,y_train)

matplotlib.use('Agg') 
output_dir="outputs"
cm=confusion_matrix(y_test,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(f"{output_dir}/confusion_matrix.png")

joblib.dump(model, f"{output_dir}/decision_tree_model.pkl")
joblib.dump(model_knn, f"{output_dir}/knn_model.pkl")