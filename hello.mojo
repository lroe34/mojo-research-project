fn main():
   from python import Python
   
   let pdplus = Python.import_module("pydotplus")
   let DecisionTreeClassifier = Python.import_module("sklearn.tree").DecisionTreeClassifier
   let iris = Python.import_module("sklearn.datasets").load_iris()
   let export_graphviz = Python.import_module("sklearn.tree").export_graphviz
   let graphviz = Python.import_module("graphviz")
   let tree = Python.import_module("sklearn.tree")

   let Image = Python.import_module("IPython.display").Image
   let display = Python.import_module("IPython.display").display
   
   let features = iris.data
   let target = iris.target
   let decisiontree = DecisionTreeClassifier(random_state=0)

   let model = decisiontree.fit(features, target)
   let dot_data = tree.export_graphviz(decisiontree,
                               out_file=None,
                               feature_names=iris.feature_names,
                               class_names=iris.target_names)

   #save the image to a file
   let graph = graphviz.Source(dot_data)
   graph.render("iris")
   display(Image(filename="iris.png"))
   
   # test the model
   let test_data = [[5.1, 3.5, 1.4, 0.2]]
   let prediction = model.predict(test_data)
   print("prediction: ", prediction)
   print(iris.target_names[prediction[0]])


   