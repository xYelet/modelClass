from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

class KNNPredict:
    def __init__(self, df, target, test_size=0.2, random_state=None):
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.accuracy = None
        self.best_params_ = None

        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess_data()
        self.best_knn_model = self.run_knn()
        self.evaluate_model()
        self.result = None


    def preprocess_data(self):
        """
        Prepare the data for KNN by splitting and scaling.
        """
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test


    def run_knn(self):
        """
        Perform grid search to find the best KNN model.
        """
        # Create a pipeline with scaling and KNN
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])

        # Define the parameter grid
        param_grid = {'knn__n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
        
        # Perform grid search
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        # Get the best parameters
        self.best_params_ = grid_search.best_params_
        best_knn = grid_search.best_params_['knn__n_neighbors']

        # Train the model with the best parameters
        best_knn_model = KNeighborsClassifier(n_neighbors=best_knn)
        best_knn_model.fit(self.X_train, self.y_train)

        return best_knn_model


    def evaluate_model(self):
        """
        Evaluate the trained KNN model and print results.
        """
        y_pred = self.best_knn_model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        print(classification_report(self.y_test, y_pred))
        print(f'Accuracy: {self.accuracy:.2f}')
        print("Best parameters found: ", self.best_params_)

        # Plot prediction results
        self.plot_prediction_results(y_pred)


    def plot_prediction_results(self, y_pred):
        """
        Plot a bar graph showing the number of correct and incorrect predictions for each class,
        with green for correct and red for incorrect predictions.
        """
        # Create DataFrame for counting correct and incorrect predictions
        results = pd.DataFrame({
            'True Label': self.y_test.values,
            'Predicted Label': y_pred,
            'Correct Prediction': self.y_test.values == y_pred
        })

        # Count the number of correct and incorrect predictions per class
        correct_counts = results[results['Correct Prediction']].groupby('True Label').size()
        incorrect_counts = results[~results['Correct Prediction']].groupby('True Label').size()

        # Combine counts into a DataFrame
        plot_data = pd.DataFrame({
            'Class': correct_counts.index,
            'Correct Predictions': correct_counts.values,
            'Incorrect Predictions': incorrect_counts.reindex(correct_counts.index, fill_value=0).values
        })

        # Set index to Class
        plot_data.set_index('Class', inplace=True)

        # Plot the bar plot with specific colors
        ax = plot_data.plot(kind='bar', stacked=True, figsize=(10, 6),
                            color=['green', 'red'])
        plt.title('Number of Correct and Incorrect Predictions per CODE_MISSPAY')
        plt.xlabel('CODE_MISSPAY')
        plt.ylabel('Count')
        plt.legend(title='Prediction', labels=['Correct Predictions', 'Incorrect Predictions'])
        plt.show()


    def print_prediction_results(self, y_pred):
        """
        Print DataFrame showing predictions and whether they are correct.
        """
        results = pd.DataFrame({
            'True Label': self.y_test.values,
            'Predicted Label': y_pred,
            'Correct Prediction': self.y_test.values == y_pred
        })
        self.result = results

