from python import Python

fn main() raises:
   let tree = Python.import_module("sklearn.tree")
   let np = Python.import_module("numpy")
   let pd = Python.import_module("pandas")
   let train_test_split = Python.import_module("sklearn.model_selection").train_test_split
   let accuracy_score = Python.import_module("sklearn.metrics").accuracy_score
   let time = Python.import_module("time")
   var timers = np.array([])
   var accuracies = np.array([])

   let decisiontree = tree.DecisionTreeClassifier()
   
   let iris = pd.read_csv("iris/iris.csv")
   iris.columns = ['sepal_length', 'sepal_width', 'petal_width', 'petal_length', 'class']
   let X = iris[['sepal_length', 'sepal_width', 'petal_width', 'petal_length']]
   let y = iris[["class"]]

   let split = train_test_split(X, y)
   

   let test_X = split[0]
   let train_X = split[1]
   let test_y = split[2]
   let train_y = split[3]

   for i in range(100):
      let start_time = time.time()
      let model = decisiontree.fit(train_X, train_y)
      let end_time = time.time()
      let y_pred = model.predict(test_X)
      timers = np.append(timers, (end_time - start_time))
      accuracies = np.append(accuracies,accuracy_score(test_y, y_pred))

   print("Average accuracy:", np.mean(accuracies))
   print("Average seconds taken to train the classifier:", np.mean(timers), "seconds")

