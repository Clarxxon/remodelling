import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import itertools
from joblib import Parallel, delayed

class ModelRecreator:
    def __init__(self, model, feature_names, num_samples=1000):
        self.model = model
        self.num_samples = num_samples
        self.distributions = {
            'uniform': lambda num_samples: np.random.rand(num_samples),
            'normal': lambda num_samples: np.random.randn(num_samples),
            'exponential': lambda num_samples: np.random.exponential(scale=1.0, size=num_samples),
            'gamma': lambda num_samples: np.random.gamma(shape=2.0, scale=1.0, size=num_samples),
            'beta': lambda num_samples: np.random.beta(a=2.0, b=5.0, size=num_samples)
        }
        self.best_mse = float('inf')
        self.best_r2 = float('-inf')
        self.best_distribution = None
        self.best_new_X = None
        self.best_new_y = None
        self.feature_names = feature_names

    def generate_correlated_data(self, base_data):
        """Создание коррелированных данных на основе базовых данных."""
        correlation_matrix = np.random.rand(base_data.shape[1], base_data.shape[1])
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1)  # Диагональные элементы - 1

        # Генерация случайных данных и коррекция с использованием матрицы корреляции
        random_data = np.random.multivariate_normal(mean=np.zeros(base_data.shape[1]), cov=correlation_matrix, size=self.num_samples)
        return pd.DataFrame(random_data, columns=base_data.columns)

    def evaluate_combination(self, combination, X_test_scaled, y_test):
        """Оценка модели на основе генерации данных."""
        random_data = np.zeros((self.num_samples, X_test_scaled.shape[1]))

        for col_idx, dist_name in enumerate(combination):
            random_data[:, col_idx] = self.distributions[dist_name](self.num_samples)

        # Создание коррелированных данных
        random_data_df = pd.DataFrame(random_data, columns=self.feature_names)
        correlated_data_df = self.generate_correlated_data(random_data_df)

        scaler = StandardScaler()
        correlated_data_scaled = scaler.fit_transform(correlated_data_df)

        generated_targets = self.model.predict(correlated_data_scaled)

        new_X = correlated_data_df
        new_y = pd.Series(generated_targets, name='Generated target')

        new_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        new_model.fit(new_X, new_y)

        y_pred_new_model = new_model.predict(X_test_scaled)

        mse_new_model = mean_squared_error(y_test, y_pred_new_model)
        r2_new_model = r2_score(y_test, y_pred_new_model)

        return combination, mse_new_model, r2_new_model, new_X, new_y

    def recreate_model(self, X_train, X_test_scaled, y_test):
        """Восстановление модели на основе сгенерированных данных."""
        distribution_names = list(self.distributions.keys())
        all_combinations = list(itertools.product(distribution_names, repeat=X_train.shape[1]))

        results = Parallel(n_jobs=-1)(delayed(self.evaluate_combination)(comb, X_test_scaled, y_test) for comb in all_combinations)

        for combination, mse_new_model, r2_new_model, new_X, new_y in results:
            if mse_new_model < self.best_mse:
                print(f"Best dataset metrics: mse {mse_new_model} r2 {r2_new_model}")
                self.best_mse = mse_new_model
                self.best_r2 = r2_new_model
                self.best_distribution = combination
                self.best_new_X = new_X
                self.best_new_y = new_y

        print(f"Best distribution combination: {self.best_distribution}, Best MSE: {self.best_mse:.2f}, Best R^2: {self.best_r2:.2f}")

        best_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        best_model.fit(self.best_new_X, self.best_new_y)

        return best_model