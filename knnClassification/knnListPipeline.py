import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


class KnnListSplit:
    def __init__(self, df_list, target, test_size=0.1, random_state=None):
        self.df_list = df_list
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.accuracy = None
        self.train_size = int((1 - self.test_size) * len(self.df_list))
        self.best_params_ = None
        self.best_knn_model = None
        self.y_test = None
        
        self.run_knn_pipeline()


    def preprocess_data(self):
        """
        Prepare the data for KNN by combining dataframes and scaling.
        """
        # Combine the first train_size dataframes for training
        train_df = pd.concat(self.df_list[:self.train_size], ignore_index=True)
        
        # Combine the remaining dataframes for testing
        test_df = pd.concat(self.df_list[self.train_size:], ignore_index=True)
        
        X_train = train_df.drop(columns=[self.target])
        y_train = train_df[self.target]
        
        X_test = test_df.drop(columns=[self.target])
        y_test = test_df[self.target]

        self.y_test = y_test  # Save y_test for plotting

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test


    def run_knn(self, X_train, y_train, X_test, y_test):
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
        grid_search.fit(X_train, y_train)
        
        # Get the best parameters
        self.best_params_ = grid_search.best_params_
        best_knn = grid_search.best_params_['knn__n_neighbors']

        # Train the model with the best parameters
        best_knn_model = KNeighborsClassifier(n_neighbors=best_knn)
        best_knn_model.fit(X_train, y_train)
        self.best_knn_model = best_knn_model

        y_pred = best_knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print(f'Accuracy: {accuracy:.2f}')
        print("Best parameters found: ", self.best_params_)

        self.accuracy = accuracy
        self.plot_prediction_results(y_pred)


    def plot_prediction_results(self, y_pred):
        """
        Plot a bar graph showing the number of correct and incorrect predictions for each class,
        with green for correct and red for incorrect predictions.
        """
        results = pd.DataFrame({
            'True Label': self.y_test.values,
            'Predicted Label': y_pred,
            'Correct Prediction': self.y_test.values == y_pred
        })
        correct_counts = results[results['Correct Prediction']].groupby('True Label').size()
        incorrect_counts = results[~results['Correct Prediction']].groupby('True Label').size()

        # Combine counts into a DataFrame
        plot_data = pd.DataFrame({
            'Class': correct_counts.index,
            'Correct Predictions': correct_counts.values,
            'Incorrect Predictions': incorrect_counts.reindex(correct_counts.index, fill_value=0).values
        })
        plot_data.set_index('Class', inplace=True)
        ax = plot_data.plot(kind='bar', stacked=True, figsize=(10, 6),
                            color=['green', 'red'])
        plt.title('Number of Correct and Incorrect Predictions per CODE_MISSPAY')
        plt.xlabel('CODE_MISSPAY')
        plt.ylabel('Count')
        plt.legend(title='Prediction', labels=['Correct Predictions', 'Incorrect Predictions'])
        plt.show()


    def run_knn_pipeline(self):
        """
        Run the KNN model and return accuracy.
        """
        X_train, X_test, y_train, y_test = self.preprocess_data()
        self.run_knn(X_train, y_train, X_test, y_test)