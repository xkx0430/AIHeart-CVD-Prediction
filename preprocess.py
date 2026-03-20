import os
import sys
import warnings
import numpy as np
import pandas as pd
from collections import OrderedDict


class PredictorProcessor:
    """Validate raw CVD data and build engineered predictors for model inference."""

    def __init__(self, model_type):
        """Initialize processor configuration and predictor schema.

        Args:
            model_type: Model variant name. Supported values are "simplified" and "full".
        """
        allowed_model_types = {'simplified', 'full'}
        model_type = model_type.lower()
        if model_type not in allowed_model_types:
            raise ValueError(f"Invalid model_type: {model_type}. Expected one of {sorted(allowed_model_types)}")
        self.model_flag = model_type == 'full'
        
        self.predictors = OrderedDict({
            'Sex': (0, 1),
            'Age': [40, 79], 
            'Estimated Glomerular Filtration Rate': [15, 140], 
            'Total Cholesterol': [2, 11], 
            'High-density Lipoprotein Cholesterol': [0.5, 4], 
            'Systolic Blood Pressure': [70, 200], 
            'Body Mass Index': [18.5, 39.9], 
            'Sleep Duration': [5, 10], 
            'County-level Area-Deprivation Index': [],
            'Antihypertensive Treatment': (0, 1), 
            'Lipid Lowering Treatment': (0, 1), 
            'Diabetes Mellitus': (0, 1), 
            'Current Smoker': (0, 1), 
            'Northern China Residence': (0, 1), 
            'Alcohol Consumption': (0, 1), 
            'Urban/Rural Residence': (0, 1),
        })
        if self.model_flag:
            self.predictors['Fasting Glucose'] = [3, 20]
            self.predictors['2-hour Postprandial Glucose'] = [3, 30]
            self.predictors['Waist Circumference'] = [50, 130]
            self.predictors['Urinary Albumin-to-Creatinine Ratio'] = [0, 25000]
            self.predictors['HbA1c'] = [4, 10]
            self.predictors['Family History of CVD'] = (0, 1)
            self.predictors['Glucose Lowering Treatment'] = (0, 1)
        self.labels = ['Event', 'Time']
        
        self.num_predictors = 0
        self.con_predictors = []
        self.dis_predictors = []

    def _read_file(self, file_path):
        """Read input data from a CSV or Excel file.

        Args:
            file_path: Path to the input dataset.

        Returns:
            A pandas DataFrame loaded from the given file.

        Raises:
            ValueError: If the file extension is not supported.
        """
        suffix = os.path.splitext(file_path)[1].lower()
        if suffix == '.csv':
            df = pd.read_csv(file_path)
            return df
        elif suffix in {'.xls', '.xlsx'}:
            df = pd.read_excel(file_path)
            return df
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _check_columns(self, df):
        """Validate required columns and basic data quality constraints.

        Checks predictor/label presence, missing values, and discrete-value validity.

        Args:
            df: Raw input DataFrame.

        Returns:
            A DataFrame restricted to required predictors and labels.

        Raises:
            ValueError: If required columns are missing or invalid values are found.
        """
        required_cols = list(self.predictors.keys())
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing predictor columns: {missing_cols}")

        missing_labels = [label for label in self.labels if label not in df.columns]
        if missing_labels:
            raise ValueError(f"Missing label columns: {missing_labels}")

        df = df[required_cols + self.labels]
        
        nan_rows = df[df.isna().any(axis=1)]
        if not nan_rows.empty:
            raise ValueError(f"NaN values found in rows:\n{nan_rows.index + 1}")
        
        abnormal_row_mask = pd.Series(False, index=df.index)
        for col, value_range in self.predictors.items():
            if isinstance(value_range, tuple):
                if len(value_range) != 2:
                    raise ValueError(f"Invalid range config for {col}: {value_range}")
                col_mask = ~df[col].isin(value_range)
            elif isinstance(value_range, list):
                col_mask = df[col].isna() # add codes for continuous data-value verification
            else:
                raise ValueError(f"Unsupported range config type for {col}: {type(value_range)}")
            abnormal_row_mask |= col_mask

        abnormal_rows = df[abnormal_row_mask]
        if not abnormal_rows.empty:
            raise ValueError(f"Invalid values found in rows:\n{abnormal_rows}")

        present_sex_values = set(df['Sex'].unique().tolist())
        if len(present_sex_values) != 2:
            warnings.warn(
                f"Dataset should include both male (Sex=1) and female (Sex=2). Missing sex values: {missing_sex_values}",
                UserWarning
            )

        return df

    def _process(self, df):
        """Generate transformed features used by downstream prediction models.

        Args:
            df: Validated input DataFrame.

        Returns:
            DataFrame with additional engineered feature columns.
        """
        df['(Age - 55) / 10'] = df['Age'].apply(lambda x: (x - 55) / 10) 
        df['(min(eGFR, 100) - 95) / 13'] = df['Estimated Glomerular Filtration Rate'].apply(lambda x: min(x, 100)).apply(lambda x: (x - 95) / 13)
        df['(max(eGFR, 60) - 95) / 13'] = df['Estimated Glomerular Filtration Rate'].apply(lambda x: max(x, 60)).apply(lambda x: (x - 95) / 13)
        df['(TC - HDL-C) - 3.5'] = (df['Total Cholesterol'] - df['High-density Lipoprotein Cholesterol']).apply(lambda x: x - 3.5)
        df['(HDL-C - 1.3) / 0.3'] = df['High-density Lipoprotein Cholesterol'].apply(lambda x: (x - 1.3) / 0.3)
        df['(min(SBP, 170) - 130) / 20'] = df['Systolic Blood Pressure'].apply(lambda x: min(x, 170)).apply(lambda x: (x - 130) / 20)
        df['(max(SBP, 100) - 130) / 20'] = df['Systolic Blood Pressure'].apply(lambda x: max(x, 100)).apply(lambda x: (x - 130) / 20)
        df['(min(BMI, 20) - 25) / 5'] = df['Body Mass Index'].apply(lambda x: min(x, 20)).apply(lambda x: (x - 25) / 5)
        df['(max(BMI, 20) - 25) / 5'] = df['Body Mass Index'].apply(lambda x: max(x, 20)).apply(lambda x: (x - 25) / 5)
        df['SDI'] = pd.qcut(df['County-level Area-Deprivation Index'], 10, labels=False, duplicates='drop') + 1
        if self.model_flag:
            df['ln(UACR)'] = df['Urinary Albumin-to-Creatinine Ratio'].apply(np.log)
            df['(HbA1c - 6.0)'] = df['HbA1c'].apply(lambda x: x - 6.0)
        return df

    def __call__(self, file_path):
        """Run the complete preprocessing pipeline for one dataset file.

        The pipeline reads raw data, validates schema and values, creates engineered
        features, and returns the model-ready table with labels.

        Args:
            file_path: Path to the source CSV/XLS/XLSX file.

        Returns:
            A processed DataFrame, or None if reading/validation fails.
        """
        try:
            df = self._read_file(file_path)
        except ValueError as e:
            print(f"Error reading file {file_path}: {e}")
            return None
        try:
            df = self._check_columns(df)
        except ValueError as e:
            print(f"Error checking columns in file {file_path}: {e}")
            return None
        
        df = self._process(df)
        
        X = pd.DataFrame()
        X['Sex'] = df['Sex']
        X['(Age - 55) / 10'] = df['(Age - 55) / 10']
        X['(TC - HDL-C) - 3.5'] = df['(TC - HDL-C) - 3.5']
        X['(HDL-C - 1.3) / 0.3'] = df['(HDL-C - 1.3) / 0.3']
        X['(min(SBP, 170) - 130) / 20'] = df['(min(SBP, 170) - 130) / 20']
        X['(max(SBP, 100) - 130) / 20'] = df['(max(SBP, 100) - 130) / 20']
        X['(min(BMI, 20) - 25) / 5'] = df['(min(BMI, 20) - 25) / 5']
        X['(max(BMI, 20) - 25) / 5'] = df['(max(BMI, 20) - 25) / 5']
        X['(min(eGFR, 100) - 95) / 13'] = df['(min(eGFR, 100) - 95) / 13']
        X['(max(eGFR, 60) - 95) / 13'] = df['(max(eGFR, 60) - 95) / 13']
        if self.model_flag:
            X['ln(UACR)'] = df['ln(UACR)']
            X['(HbA1c - 6.0)'] = df['(HbA1c - 6.0)']
            X['wc'] = df['Waist Circumference']
            X['Glu0'] = df['Fasting Glucose']
            X['Glu120'] = df['2-hour Postprandial Glucose']
        X['SDI'] = df['SDI']
        X['sleepr'] = df['Sleep Duration']
        if self.model_flag:
            X['diabetes_drug A1c'] = df['Glucose Lowering Treatment'] * df['HbA1c']
        X['Statin nonhdl'] = df['Lipid Lowering Treatment'] * (df['Total Cholesterol'] - df['High-density Lipoprotein Cholesterol'])
        X['Antihtn sbp'] = df['Antihypertensive Treatment'] * df['Systolic Blood Pressure']
        X['(Age - 55) / 10 (min(SBP, 170) - 130) / 20'] = df['(Age - 55) / 10'] * df['(min(SBP, 170) - 130) / 20']
        X['(Age - 55) / 10 (max(SBP, 100) - 130) / 20'] = df['(Age - 55) / 10'] * df['(max(SBP, 100) - 130) / 20']
        X['(Age - 55) / 10 Diabetes'] = df['(Age - 55) / 10'] * df['Diabetes Mellitus']
        X['(Age - 55) / 10 Cursmk'] = df['(Age - 55) / 10'] * df['Current Smoker']
        X['(Age - 55) / 10 (min(BMI, 20) - 25) / 5'] = df['(Age - 55) / 10'] * df['(min(BMI, 20) - 25) / 5']
        X['(Age - 55) / 10 (min(eGFR, 100) - 95) / 13'] = df['(Age - 55) / 10'] * df['(min(eGFR, 100) - 95) / 13']
        if self.model_flag:
            X['(Age - 55) / 10 ln(UACR)'] = df['(Age - 55) / 10'] * df['ln(UACR)']
            X['(Age - 55) / 10 Glu0'] = df['(Age - 55) / 10'] * df['Fasting Glucose']
            X['(Age - 55) / 10 Glu120'] = df['(Age - 55) / 10'] * df['2-hour Postprandial Glucose']
        X['(min(SBP, 170) - 130) / 20 (max(SBP, 100) - 130) / 20'] = df['(min(SBP, 170) - 130) / 20'] * df['(max(SBP, 100) - 130) / 20']
        X['(min(SBP, 170) - 130) / 20 (min(BMI, 20) - 25) / 5'] = df['(min(SBP, 170) - 130) / 20'] * df['(min(BMI, 20) - 25) / 5']
        if self.model_flag:
            X['(min(SBP, 170) - 130) / 20 ln(UACR)'] = df['(min(SBP, 170) - 130) / 20'] * df['ln(UACR)']
            X['(min(SBP, 170) - 130) / 20 Glu0'] = df['(min(SBP, 170) - 130) / 20'] * df['Fasting Glucose']
            X['(min(SBP, 170) - 130) / 20 Glu120'] = df['(min(SBP, 170) - 130) / 20'] * df['2-hour Postprandial Glucose']
        X['(max(SBP, 100) - 130) / 20 (min(BMI, 20) - 25) / 5'] = df['(max(SBP, 100) - 130) / 20'] * df['(min(BMI, 20) - 25) / 5']
        if self.model_flag:
            X['(max(SBP, 100) - 130) / 20 ln(UACR)'] = df['(max(SBP, 100) - 130) / 20'] * df['ln(UACR)']
            X['(max(SBP, 100) - 130) / 20 Glu0'] = df['(max(SBP, 100) - 130) / 20'] * df['Fasting Glucose']
            X['(max(SBP, 100) - 130) / 20 Glu120'] = df['(max(SBP, 100) - 130) / 20'] * df['2-hour Postprandial Glucose']
            X['Diabetes ln(UACR)'] = df['Diabetes Mellitus'] * df['ln(UACR)']
        X['(min(BMI, 20) - 25) / 5 (min(eGFR, 100) - 95) / 13'] = df['(min(BMI, 20) - 25) / 5'] * df['(min(eGFR, 100) - 95) / 13']
        X['(min(BMI, 20) - 25) / 5 (max(eGFR, 60) - 95) / 13'] = df['(min(BMI, 20) - 25) / 5'] * df['(max(eGFR, 60) - 95) / 13']
        if self.model_flag:
            X['(min(BMI, 20) - 25) / 5 ln(UACR)'] = df['(min(BMI, 20) - 25) / 5'] * df['ln(UACR)']
            X['(min(BMI, 20) - 25) / 5 Glu120'] = df['(min(BMI, 20) - 25) / 5'] * df['2-hour Postprandial Glucose']
            X['(min(eGFR, 100) - 95) / 13 ln(UACR)'] = df['(min(eGFR, 100) - 95) / 13'] * df['ln(UACR)']
            X['(min(eGFR, 100) - 95) / 13 Glu0'] = df['(min(eGFR, 100) - 95) / 13'] * df['Fasting Glucose']
            X['(min(eGFR, 100) - 95) / 13 Glu120'] = df['(min(eGFR, 100) - 95) / 13'] * df['2-hour Postprandial Glucose']
            X['(max(eGFR, 60) - 95) / 13 ln(UACR)'] = df['(max(eGFR, 60) - 95) / 13'] * df['ln(UACR)']
            X['(max(eGFR, 60) - 95) / 13 Glu0'] = df['(max(eGFR, 60) - 95) / 13'] * df['Fasting Glucose']
            X['(max(eGFR, 60) - 95) / 13 Glu120'] = df['(max(eGFR, 60) - 95) / 13'] * df['2-hour Postprandial Glucose']
            X['ln(UACR) Glu0'] = df['ln(UACR)'] * df['Fasting Glucose']
            X['ln(UACR) Glu120'] = df['ln(UACR)'] * df['2-hour Postprandial Glucose']
        X['Diabetes'] = df['Diabetes Mellitus']
        self.dis_predictors.append('Diabetes')
        X['Cursmk'] = df['Current Smoker']
        self.dis_predictors.append('Cursmk')
        X['Antihtn'] = df['Antihypertensive Treatment']
        self.dis_predictors.append('Antihtn')
        X['Statin'] = df['Lipid Lowering Treatment']
        self.dis_predictors.append('Statin')
        if self.model_flag:
            X['diabetes_drug'] = df['Glucose Lowering Treatment']
            self.dis_predictors.append('diabetes_drug')
            X['cvd_f'] = df['Family History of CVD']
            self.dis_predictors.append('cvd_f')
        X['north'] = df['Northern China Residence']
        self.dis_predictors.append('north')
        X['drk_g'] = df['Alcohol Consumption']
        self.dis_predictors.append('drk_g')
        X['region'] = df['Urban/Rural Residence']
        self.dis_predictors.append('region')
        
        X[self.labels[0]] = df[self.labels[0]]
        X[self.labels[1]] = df[self.labels[1]]
        
        for col in X.keys():
            if col not in self.dis_predictors and col not in self.labels and col != 'Sex':
                self.con_predictors.append(col)
        self.num_predictors = len(self.con_predictors) + len(self.dis_predictors)
        return X


if __name__ == "__main__":
    predictor_processor = PredictorProcessor(model_type='full')
    df = predictor_processor('datasets/example_data.csv')
    print(df)
    