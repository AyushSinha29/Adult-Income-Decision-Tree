import pandas as pd
df = pd.read_csv("/content/AdultIncome.csv")
df.isnull().sum()
df  = pd.get_dummies(df, drop_first=True)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state= 0)
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
score = dtc.score(x_test, y_test)
df.groupby('wc').size()
import graphviz
from sklearn import tree
import matplotlib.pyplot as plt
full = list(df.columns)
feat = full.pop()
features = full
target = feat
print(target)
dot_data = tree.export_graphviz(dtc, out_file=None, 
                                feature_names=features,  
                                class_names=target,
                                filled=True)
# Draw graph
graph = graphviz.Source(dot_data, format="jpg") 
print(graph)
graph.render("decision_tree_graphivz")

