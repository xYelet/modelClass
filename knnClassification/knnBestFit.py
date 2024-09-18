import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np

class KnnBestFit:
    def __init__(self, df, target='CODE_MISSPAY', test_size = 0.3, random_state = 42):
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

        self.accuracy = None
        self.conf_matrix = None
        self.class_report = None
        self.best_features = None
        self.accuracies_df = None
        self.y_pred = None
        self.y_test = None

        self.run_bestfit_knn_pipeline()


    def preprocess_data(self):
        """
        Preparing the df before doing KNN
        """
        X = self.df[self.df.columns.difference([self.target])]
        y = self.df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test


    def evaluate_model(self, X, y, selected_features, cv):
        """
        Evaluate a KNN model with the given features using cross-validation.
        """
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])
        param_grid = {'knn__n_neighbors': np.arange(1, 5)}
        grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy')
        grid.fit(X[selected_features], y)

        return grid.best_score_


    def feature_selection(self):
        """
        Identifies the most relevant features from the df
        """
        names = self.df.columns.difference([self.target])
        selected = []
        accuracies = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=27)
        np.random.seed(27)
        
        for i in range(1, len(names) + 1):
            accs = []
            models = []
            for name in names:
                preds_new = selected + [name]
                acc = self.evaluate_model(self.df, self.df[self.target], preds_new, cv)
                accs.append(acc)
                models.append(preds_new)
            
            jstar = np.argmax(accs)
            accuracies.append({
                'size': i,
                'features': models[jstar],
                'accuracy': accs[jstar]
            })
            selected.append(names[jstar])
            names = names.drop(names[jstar])
        
        self.accuracies_df = pd.DataFrame(accuracies)
        self.best_features = self.accuracies_df.loc[self.accuracies_df['accuracy'].idxmax(), 'features']
        return self.best_features, self.accuracies_df


    def train_and_evaluate_model(self, X, y):
        """
        Train and evaluate the model
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        self.y_pred = y_pred
        self.y_test = y_test
        self.accuracy = accuracy_score(y_test, y_pred)
        self.conf_matrix = confusion_matrix(y_test, y_pred)
        self.class_report = classification_report(y_test, y_pred)
        
        return y_pred, y_test


    def plot_accuracies(self, width, height):
        """
        Plot the results
        """
        plt.figure(figsize=(width, height))
        sns.lineplot(data=self.accuracies_df, x='size', y='accuracy', marker='o')
        plt.xlabel('Number of Predictors', fontsize=16)
        plt.ylabel('Accuracy Estimate', fontsize=16)
        plt.title('Estimated Accuracy vs Number of Predictors', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()


    def run_bestfit_knn_pipeline(self):
        """
        Executes the entire KNN pipeline: preprocessing, feature selection,
        model training, evaluation, and plotting accuracy results.
        """
        # Step 1: Preprocess the data
        X_train, X_test, y_train, y_test = self.preprocess_data()
        
        # Step 2: Perform feature selection
        self.best_features, self.accuracies_df = self.feature_selection()
        
        # Step 3: Prepare the data with the selected features
        X = self.df[self.best_features]
        y = self.df[self.target]
        
        # Step 4: Train and evaluate the model with selected features
        self.y_pred, self.y_test = self.train_and_evaluate_model(X, y)

        # Output evaluation metrics
        self.accuracy
        self.conf_matrix
        self.class_report
        self.best_features
        self.accuracies_df
        self.y_pred
        self.y_test
