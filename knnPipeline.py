from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class KNNSimplePipeline:
    def __init__(self, df, target, test_size=0.2, random_state=None):
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.accuracy = None

        self.run_knn_pipeline()

    def preprocess_data(self):
        """
        Prepare the data for KNN by splitting and scaling.
        """
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test


    def run_knn(self, X_train, y_train, X_test, y_test):
        """
        Train the KNN model, predict, and evaluate.
        """
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')

        self.accuracy = accuracy


    def run_knn_pipeline(self):
        """
        Run the KNN model and return accuracy.
        """
        X_train, X_test, y_train, y_test = self.preprocess_data()
        return self.run_knn(X_train, y_train, X_test, y_test)