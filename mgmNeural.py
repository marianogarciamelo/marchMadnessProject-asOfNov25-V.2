import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from feature_builder import materialize_dataset, OUTPUT_PATH


class MarchMadnessNeuralNet:
    """Neural network model for March Madness predictions."""

    def __init__(self):
        self.df = None
        self.scaler = StandardScaler()
        self.model = None

    def load_data(self):
        """Load the parquet data from OUTPUT_PATH."""
        self.df = pd.read_parquet(OUTPUT_PATH)
        return self

    def display_champions(self):
        """Display championship teams (ROUND == 1) with YEAR, TEAM, SEED, and ROUND."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Filter for championship teams (ROUND == 1)
        champions = self.df[self.df.iloc[:, 4] == 1]

        # Select columns: 0 (YEAR), 1 (TEAM), 3 (SEED), 4 (ROUND) - skip 2 (TEAM_NO)
        champion_info = champions.iloc[:, [0, 1, 3, 4]]

        print(f"\n{'='*60}")
        print("CHAMPIONSHIP TEAMS (ROUND == 1)")
        print(f"{'='*60}")
        print(champion_info.to_string(index=False))
        print(f"{'='*60}")

        return self

    def prep_data(self):
        """Prepare data for training."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Display number of features
        num_features = self.df.shape[1]
        print(f"\n{'='*60}")
        print("DATA PREPARATION")
        print(f"{'='*60}")
        print(f"Total columns in dataset: {num_features}")
        print(f"{'='*60}")

        # Display champions right after
        self.display_champions()

        # Create binary target: 1 if ROUND == 1 (champion), 0 otherwise
        # Column index 4 is ROUND
        self.y = (self.df.iloc[:, 4] == 1).astype(int)

        # Feature columns: Include SEED (column 3) + statistical features (columns 5-14)
        # Exclude: YEAR (0), TEAM (1), TEAM_NO (2), ROUND (4), and columns 15+
        seed_col = self.df.iloc[:, [3]]  # SEED
        stats_cols = self.df.iloc[:, 5:16]  # Columns 5-15 (10 features)
        self.X = pd.concat([seed_col, stats_cols], axis=1).copy()

        print(f"\n{'='*60}")
        print(f"FEATURE SELECTION ({self.X.shape[1]} features)")
        print(f"{'='*60}")
        for i, col in enumerate(self.X.columns, 1):
            print(f"  {i:2d}. {col}")
        print(f"{'='*60}")

        # Check for and handle missing values
        missing_counts = self.X.isnull().sum()
        if missing_counts.any():
            print(f"\n{'='*60}")
            print("MISSING VALUES DETECTED")
            print(f"{'='*60}")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count} missing ({count/len(self.X)*100:.1f}%)")
            print(f"{'='*60}")

            # Fill NaN values with column median (robust to outliers)
            print(f"\nImputation: Filling with column medians...")
            self.X = self.X.fillna(self.X.median())
            print(f"Imputation complete.")

        print(f"\n{'='*60}")
        print("TARGET DISTRIBUTION")
        print(f"{'='*60}")
        print(f"  Champions (y=1):     {self.y.sum():4d}")
        print(f"  Non-champions (y=0): {(self.y == 0).sum():4d}")
        print(f"  Class imbalance:     {(self.y == 0).sum() / self.y.sum():.2f}:1")
        print(f"{'='*60}")

        return self

    def grid_search_for_neural_params(self, param_grid, random_state=42):
        """Perform grid search to find optimal neural network hyperparameters.

        Args:
            param_grid: Dictionary defining parameter grid for GridSearchCV
            random_state: Random seed for reproducibility
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not prepared. Call prep_data() first.")

        print(f"\n{'='*60}")
        print("GRID SEARCH FOR NEURAL NETWORK HYPERPARAMETERS")
        print(f"{'='*60}")
        print("Note: Using time-based folds to prevent data leakage")
        print("WARNING: Scaling happens INSIDE each fold via Pipeline")
        print(f"{'='*60}")

        # Get YEAR column from original dataframe
        years = self.df.iloc[:, 0]  # Column 0 is YEAR

        # Define EXPANDING WINDOW time-based folds
        # This is NOT data leakage because we always train on PAST, validate on FUTURE
        # Leakage would be: train on future, validate on past (we never do this!)
        time_folds = [
            {"name": "Fold 1", "train": range(2008, 2012), "val": range(2012, 2016)},
            {"name": "Fold 2", "train": range(2008, 2016), "val": range(2016, 2020)},
            {"name": "Fold 3", "train": range(2008, 2020), "val": range(2020, 2023)},
            {"name": "Fold 4", "train": range(2008, 2023), "val": range(2023, 2026)},
        ]

        # Create custom CV splits for GridSearchCV
        # GridSearchCV expects an iterable of (train_indices, val_indices) tuples
        custom_cv = []
        for fold in time_folds:
            train_idx = years.isin(fold['train'])
            val_idx = years.isin(fold['val'])
            # Convert boolean arrays to integer indices
            train_indices = np.where(train_idx)[0]
            val_indices = np.where(val_idx)[0]
            custom_cv.append((train_indices, val_indices))
            print(f"{fold['name']}: Train {min(fold['train'])}-{max(fold['train'])}, Val {min(fold['val'])}-{max(fold['val'])}")

        print(f"{'='*60}")

        # Create pipeline with scaler + MLP
        # This ensures scaling is fitted ONLY on training data in each fold
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(random_state=random_state))
        ])

        # Update param_grid keys to include 'mlp__' prefix for pipeline
        pipeline_param_grid = {}
        for key, value in param_grid.items():
            pipeline_param_grid[f'mlp__{key}'] = value

        # Setup GridSearchCV with time-based cross-validation
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=pipeline_param_grid,
            scoring='f1',
            n_jobs=-1,
            cv=custom_cv,  # Use custom time-based folds
            verbose=2
        )

        # Fit grid search on UNSCALED data
        # Pipeline will handle scaling inside each fold
        grid_search.fit(self.X, self.y)

        # Extract best params and remove 'mlp__' prefix
        best_params = {}
        for key, value in grid_search.best_params_.items():
            if key.startswith('mlp__'):
                best_params[key.replace('mlp__', '')] = value

        print(f"\nBest parameters found: {best_params}")
        print(f"Best F1 score: {grid_search.best_score_:.4f}")
        print(f"{'='*60}\n")

        return best_params
    
    def train(self, n_splits=5, hidden_layer_sizes=None, max_iter=None, random_state=42, **kwargs):
        """Train the neural network model using time-based k-fold cross-validation.

        Args:
            n_splits: Number of folds for k-fold cross-validation (ignored, using 5 time-based folds)
            hidden_layer_sizes: Tuple defining the architecture (number of neurons per hidden layer)
            max_iter: Maximum iterations for training
            random_state: Random seed for reproducibility
            **kwargs: Additional MLPClassifier parameters from grid search (e.g., alpha, learning_rate_init)
        """
        #set default parameters if not provided
        if hidden_layer_sizes is None:
            hidden_layer_sizes = (100, 50)
        if max_iter is None:
            max_iter = 500

        if self.X is None or self.y is None:
            raise ValueError("Data not prepared. Call prep_data() first.")

        print(f"\n{'='*60}")
        print(f"TIME-BASED K-FOLD CROSS-VALIDATION TRAINING")
        print(f"{'='*60}")
        print(f"Model architecture: {hidden_layer_sizes}")
        print(f"Max iterations: {max_iter}")
        print(f"Note: Using time-preserved folds to prevent data leakage")
        print(f"WARNING: Scaler fitted ONLY on training data per fold")
        print(f"{'='*60}\n")

        # Define EXPANDING WINDOW time-based folds manually
        # Get YEAR column from original dataframe
        years = self.df.iloc[:, 0]  # Column 0 is YEAR

        # Define year ranges for each fold (EXPANDING WINDOW - more training data over time)
        # This is NOT data leakage because we always train on PAST, validate on FUTURE
        time_folds = [
            {"name": "Fold 1", "train": range(2008, 2012), "val": range(2012, 2016)},  # Train: 2008-2011, Val: 2012-2015
            {"name": "Fold 2", "train": range(2008, 2016), "val": range(2016, 2020)},  # Train: 2008-2015, Val: 2016-2019
            {"name": "Fold 3", "train": range(2008, 2020), "val": range(2020, 2023)},  # Train: 2008-2019, Val: 2020-2022
            {"name": "Fold 4", "train": range(2008, 2023), "val": range(2023, 2026)},  # Train: 2008-2022, Val: 2023-2025
        ]

        # Store metrics for each fold
        fold_accuracies = []
        fold_f1_scores = []

        # Iterate through each time-based fold
        for fold_idx, fold in enumerate(time_folds, 1):
            print(f"{fold['name']}")
            print("-" * 60)
            print(f"  Train years:      {min(fold['train'])}-{max(fold['train'])}")
            print(f"  Validation years: {min(fold['val'])}-{max(fold['val'])}")

            # Create train/val indices based on years
            train_idx = years.isin(fold['train'])
            val_idx = years.isin(fold['val'])

            # Split data for this fold using boolean indexing
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            print(f"  Train samples:    {len(X_train):4d} teams ({y_train.sum()} champions)")
            print(f"  Val samples:      {len(X_val):4d} teams ({y_val.sum()} champions)")

            # Scale features (fit ONLY on train, transform both)
            # CRITICAL: Do NOT fit on validation data to prevent leakage
            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_val_scaled = fold_scaler.transform(X_val)

            # Compute sample weights to handle class imbalance
            # Champions get higher weight (inversely proportional to class frequency)
            sample_weights = compute_sample_weight('balanced', y_train)

            # Create and train model for this fold
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                verbose=False,
                **kwargs  # Pass through any additional params from grid search
            )

            # Fit with sample weights to penalize champion misclassification more
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

            # Evaluate on validation set
            y_pred = model.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, zero_division=0)

            fold_accuracies.append(accuracy)
            fold_f1_scores.append(f1)

            print(f"  Accuracy:         {accuracy:.4f}")
            print(f"  F1 Score:         {f1:.4f}")
            print(f"  Iterations:       {model.n_iter_}")
            print()

        # Print summary statistics
        print(f"{'='*60}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"  Number of folds: {len(time_folds)}")
        print(f"  Mean Accuracy:   {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
        print(f"  Mean F1 Score:   {np.mean(fold_f1_scores):.4f} (+/- {np.std(fold_f1_scores):.4f})")
        print(f"{'='*60}\n")

        # Train final model on ALL data
        print(f"{'='*60}")
        print("FINAL MODEL TRAINING")
        print(f"{'='*60}")
        print(f"Training on full dataset ({len(self.X)} samples)...")
        print(f"Note: This scaler is fitted on ALL data for future predictions only")

        # Fit scaler on all historical data (correct for making future predictions)
        self.scaler.fit(self.X)
        X_scaled = self.scaler.transform(self.X)

        # Compute sample weights for full dataset
        sample_weights = compute_sample_weight('balanced', self.y)

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=False,  # Disabled to prevent random validation split
            verbose=False,
            **kwargs  # Pass through any additional params from grid search
        )

        self.model.fit(X_scaled, self.y, sample_weight=sample_weights)
        print(f"  Iterations:       {self.model.n_iter_}")
        print(f"  Training accuracy: {self.model.score(X_scaled, self.y):.4f}")
        print(f"{'='*60}\n")

        # Display feature importance based on input layer weights
        print(f"{'='*60}")
        print("FEATURE IMPORTANCE (Input Layer Weights)")
        print(f"{'='*60}")
        print("Higher absolute values = stronger influence on predictions\n")

        # Get weights from input layer to first hidden layer
        input_weights = self.model.coefs_[0]  # Shape: (n_features, n_neurons_layer1)

        # Calculate feature importance as mean absolute weight across all neurons in first layer
        feature_importance = np.abs(input_weights).mean(axis=1)

        # Create dataframe for better display
        importance_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)

        # Format the output with aligned columns
        print(f"{'Rank':<6} {'Feature':<30} {'Importance':>12}")
        print("-" * 60)
        for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
            print(f"{rank:<6} {row['Feature']:<30} {row['Importance']:>12.6f}")
        print(f"{'='*60}\n")

        return self
        


    def predict(self, test_csv_path):
        """Make predictions on a test dataset.

        Args:
            test_csv_path: Path to the test CSV file

        Returns:
            DataFrame with predictions
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() first.")

        print(f"\n{'='*60}")
        print("LOADING TEST DATA")
        print(f"{'='*60}")
        print(f"Test file: {test_csv_path}")

        # Load test data
        test_df = pd.read_csv(test_csv_path)
        print(f"  Samples loaded:   {len(test_df)}")
        print(f"  Total columns:    {test_df.shape[1]}")

        # Extract same features as training (columns 3, 5-15)
        # Assuming test CSV has same structure as training data
        seed_col = test_df.iloc[:, [3]]  # SEED
        stats_cols = test_df.iloc[:, 5:16]  # Columns 5-15
        X_test = pd.concat([seed_col, stats_cols], axis=1).copy()

        print(f"  Features extracted: {X_test.shape[1]}")
        print(f"{'='*60}")

        # Handle missing values using similarity-based imputation
        missing_counts = X_test.isnull().sum()
        if missing_counts.any():
            print(f"\n{'='*60}")
            print("HANDLING MISSING VALUES")
            print(f"{'='*60}")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count} missing ({count/len(X_test)*100:.1f}%)")

            # Check if entire columns are missing (100% missing rate)
            completely_missing_cols = missing_counts[missing_counts == len(X_test)]

            if len(completely_missing_cols) > 0:
                print(f"\nSimilarity-based imputation (k=10 nearest neighbors)...")

                # Get columns that ARE available in both test and training data
                available_cols = [col for col in X_test.columns if col not in completely_missing_cols.index]

                # For each test team, find similar teams in training data
                for idx in X_test.index:
                    test_row = X_test.loc[idx, available_cols]

                    # Calculate similarity to all training teams using available features
                    # Using Manhattan distance on standardized features
                    similarities = []
                    for train_idx in self.X.index:
                        train_row = self.X.loc[train_idx, available_cols]
                        # Simple distance calculation (lower = more similar)
                        distance = np.abs(test_row - train_row).sum()
                        similarities.append((train_idx, distance))

                    # Sort by similarity and get top 10 most similar teams
                    similarities.sort(key=lambda x: x[1])
                    similar_team_indices = [s[0] for s in similarities[:10]]

                    # Fill missing values with median of similar teams
                    for missing_col in completely_missing_cols.index:
                        similar_values = self.X.loc[similar_team_indices, missing_col]
                        X_test.loc[idx, missing_col] = similar_values.median()

                print(f"  Imputed {len(completely_missing_cols)} columns")

            # For partially missing data, drop those rows
            remaining_missing = X_test.isnull().sum()
            if remaining_missing.any():
                rows_with_missing = X_test.isnull().any(axis=1)
                num_rows_before = len(X_test)

                # Drop rows with any remaining missing values
                valid_indices = ~rows_with_missing
                X_test = X_test[valid_indices].copy()
                test_df = test_df[valid_indices].copy()

                num_rows_after = len(X_test)
                print(f"\nDropped {num_rows_before - num_rows_after} rows with partial missing data")
                print(f"  Remaining samples: {num_rows_after}")

            print(f"{'='*60}")

        # Scale test data using fitted scaler from training
        X_test_scaled = self.scaler.transform(X_test)

        # Make predictions
        print(f"\n{'='*60}")
        print("MAKING PREDICTIONS")
        print(f"{'='*60}")

        probabilities = self.model.predict_proba(X_test_scaled)

        # Use custom threshold instead of default 0.5
        # Default predict() uses 0.5 threshold, but with class imbalance, lower threshold is needed
        threshold = 0.05  # 5% threshold instead of 50%
        predictions = (probabilities[:, 1] >= threshold).astype(int)

        print(f"  Decision threshold:   {threshold:.1%}")

        # Create results dataframe
        results = test_df.iloc[:, [0, 1, 3]].copy()  # YEAR, TEAM, SEED
        results['PREDICTED_CHAMPION'] = predictions
        results['CHAMPION_PROBABILITY'] = probabilities[:, 1]  # Probability of class 1 (champion)

        # Sort by probability (highest first)
        results = results.sort_values('CHAMPION_PROBABILITY', ascending=False)

        print(f"  Total predictions:    {len(predictions)}")
        print(f"  Champions predicted:  {predictions.sum()}")
        print(f"{'='*60}")

        # Print all teams predicted as champions (PREDICTED_CHAMPION == 1)
        if predictions.sum() > 0:
            print(f"\n{'='*60}")
            print("PREDICTED CHAMPIONS")
            print(f"{'='*60}")

            champions = results[results['PREDICTED_CHAMPION'] == 1].copy()
            for idx, (i, row) in enumerate(champions.iterrows(), 1):
                print(f"\n  Rank #{idx}")
                print(f"  Team:        {row.iloc[1]}")  # TEAM name (column index 1)
                print(f"  Seed:        {int(row.iloc[2])}")  # SEED (column index 2)
                print(f"  Probability: {row['CHAMPION_PROBABILITY']:.2%}")
                if idx < len(champions):
                    print(f"  {'-'*56}")

            print(f"\n{'='*60}")

        return results


def main():
    """Main execution function."""
    print(f"\n{'='*60}")
    print("MARCH MADNESS NEURAL NETWORK PREDICTOR")
    print(f"{'='*60}\n")

    # Materialize the dataset
    print("Building dataset from feature_builder...")
    materialize_dataset()
    print("Dataset materialized successfully.\n")

    # Create and run the neural network
    nn = MarchMadnessNeuralNet()
    nn.load_data().prep_data()

    # Define parameter grid for grid search
    param_grid = {
        'hidden_layer_sizes': [(50, 25), (100, 50), (150, 75)],
        'max_iter': [300, 500, 700]
    }

    # Run grid search to find best hyperparameters
    best_params = nn.grid_search_for_neural_params(param_grid)

    # Train with best parameters from grid search
    nn.train(**best_params)

    # Predict on 2026 test data
    test_results = nn.predict("march+madness+data/2026training.csv")
    test_results.to_csv("predictions_2026.csv", index=False)

    print(f"\n{'='*60}")
    print("RESULTS SAVED")
    print(f"{'='*60}")
    print(f"  File: predictions_2026.csv")
    print(f"  Rows: {len(test_results)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
