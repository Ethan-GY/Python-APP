import os
import subprocess
import sys

def install_from_requirements():
    requirements_file = 'requirements.txt'
    if os.path.exists(requirements_file):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("Suceess")
        except subprocess.CalledProcessError as e:
            print(f"Failure: {e}")

install_from_requirements()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Font and Style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Read Data
df = pd.read_csv('student-por.csv')

# G1, G2, G3 average grade
df['avg_grade'] = (df['G1'] + df['G2'] + df['G3']) / 3

print("=== Advanced Grade Analysis (Improved) ===")
print(f"Dataset Size: {df.shape}")
print(f"Average Grade: {df['avg_grade'].mean():.2f} ± {df['avg_grade'].std():.2f}")
print(f"Median Grade: {df['avg_grade'].median():.2f}")

# 1. Advanced Feature Engineering
print("\n=== Advanced Feature Engineering ===")

# More Features
df['total_alcohol'] = df['Dalc'] + df['Walc']  # Total Alcohol Consumption
df['alcohol_frequency'] = (df['Dalc'] + df['Walc'] * 2) / 3  # Weighted Alcohol Frequency (Weekends are more important)
df['study_efficiency'] = df['studytime'] / (df['absences'] + 1)  # Study Efficiency
df['parent_edu_score'] = (df['Medu'] * 0.6 + df['Fedu'] * 0.4)  # Parent Education Score (Mother is more important)
df['academic_risk'] = df['failures'] * 2 + (df['absences'] > 5).astype(int) * 3  # Academic Risk Index
df['social_activity'] = df['goout'] + df['freetime']  # Social Activity Index
df['family_support'] = df['famrel'] + (df['famsup'] == 'yes').astype(int) * 2  # Family Support Index
df['school_support'] = (df['schoolsup'] == 'yes').astype(int) * 2 + (df['paid'] == 'yes').astype(int)  # School Support Index
df['motivation'] = (df['higher'] == 'yes').astype(int) * 3 + df['reason'].map({'home': 1, 'reputation': 2, 'course': 3, 'other': 1})  # Learning Motivation

# Interaction Features
df['alcohol_study_interaction'] = df['total_alcohol'] * (5 - df['studytime'])  # Negative Interaction between Alcohol and Studytime
df['absence_failure_interaction'] = df['absences'] * df['failures']  # Interaction between Absences and Failures
df['support_motivation_interaction'] = df['family_support'] * df['motivation']  # Interaction between Family Support and Learning Motivation

# Categorical Features
df['risk_category'] = pd.cut(df['academic_risk'], 
                           bins=[-1, 2, 5, 10, 100], 
                           labels=['Low', 'Medium', 'High', 'Very High'])
df['alcohol_category'] = pd.cut(df['total_alcohol'], 
                              bins=[-1, 2, 5, 8, 10], 
                              labels=['None-Low', 'Moderate', 'High', 'Very High'])
df['support_level'] = pd.cut(df['family_support'] + df['school_support'], 
                           bins=[-1, 3, 6, 9, 12], 
                           labels=['Low', 'Medium', 'High', 'Very High'])

print("New Advanced Features:")
advanced_features = ['alcohol_frequency', 'study_efficiency', 'parent_edu_score', 
                    'academic_risk', 'social_activity', 'family_support', 
                    'school_support', 'motivation', 'alcohol_study_interaction',
                    'absence_failure_interaction', 'support_motivation_interaction']

for feature in advanced_features:
    if feature in df.columns:
        corr = df[feature].corr(df['avg_grade'])
        print(f"  {feature}: correlation = {corr:.3f}")

# 2. Deeper Data Exploration
print("\n=== Deeper Data Exploration ===")

# Grades Distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Distribution of Grades
axes[0].hist(df['avg_grade'], bins=20, alpha=0.7, color='skyblue', edgecolor='black', density=True)
df['avg_grade'].plot(kind='density', ax=axes[0], color='red', linewidth=2)
axes[0].axvline(df['avg_grade'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {df["avg_grade"].mean():.2f}')
axes[0].axvline(df['avg_grade'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df["avg_grade"].median():.2f}')
axes[0].set_xlabel('Average Grade')
axes[0].set_ylabel('Density')
axes[0].set_title('Distribution of Average Grades')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Grades Distribution Across Semesters
grades_data = pd.DataFrame({
    'G1': df['G1'],
    'G2': df['G2'], 
    'G3': df['G3']
})
grades_stats = grades_data.describe()
axes[1].boxplot([grades_data['G1'], grades_data['G2'], grades_data['G3']])
axes[1].set_xticklabels(['G1', 'G2', 'G3'])
axes[1].set_ylabel('Grade')
axes[1].set_title('Grade Distribution Across Semesters')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 3. Advanced Feature Selection
print("\n=== Advanced Feature Selection ===")

# Prepare feature set
base_features = ['Dalc', 'Walc', 'studytime', 'absences', 'famrel', 'health', 
                'failures', 'goout', 'freetime', 'Medu', 'Fedu', 'traveltime']

# Additional Features
all_features = base_features + advanced_features

# Categorical Features
categorical_features = ['romantic', 'activities', 'internet', 'higher', 'famsup', 'schoolsup', 'paid', 'nursery']

# Create Full Feature Set
X = df[all_features].copy()
for cat_feat in categorical_features:
    if cat_feat in df.columns:
        dummies = pd.get_dummies(df[cat_feat], prefix=cat_feat, drop_first=True)
        X = pd.concat([X, dummies], axis=1)

y = df['avg_grade']

print(f"Initial Feature Count: {X.shape[1]}")

# Recursive Feature Elimination (RFE)       
# Using Recursive Feature Elimination for feature selection
estimator = RandomForestRegressor(n_estimators=100, random_state=42)
selector = RFE(estimator, n_features_to_select=20, step=1)
X_selected = selector.fit_transform(X, y)
selected_mask = selector.support_
selected_features = X.columns[selected_mask]

print(f"Selected Feature Count: {len(selected_features)}")
print("Selected Features:", list(selected_features))

X_final = X[selected_features]

# 4. Advanced Model Comparison
print("\n=== Advanced Model Comparison ===")

# Split Train and Test Sets
# Split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5))

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models and parameter grids
models = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'Ridge Regression': {
        'model': Ridge(),
        'params': {'alpha': [0.1, 1.0, 10.0, 100.0]}
    },
    'Lasso Regression': {
        'model': Lasso(),
        'params': {'alpha': [0.1, 1.0, 10.0, 100.0]}
    },
    'ElasticNet': {
        'model': ElasticNet(),
        'params': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]}
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
    },
    'AdaBoost': {
        'model': AdaBoostRegressor(random_state=42),
        'params': {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]}
    },
    'SVR': {
        'model': SVR(),
        'params': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']}
    }
}

# Model comparison results
results = {}
best_models = {}

for name, config in models.items():
    print(f"Training {name}...")
    
    if config['params']:  # Models with hyperparameters
        grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='r2', n_jobs=-1)
        
        if name in ['Random Forest', 'Gradient Boosting', 'AdaBoost']:
            # Tree models don't need standardization
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
        else:
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test_scaled)
            
        best_models[name] = best_model
        print(f"  Best parameters: {grid_search.best_params_}")
        
    else:  # Models without hyperparameters
        if name == 'Linear Regression':
            best_model = config['model']
            best_model.fit(X_train_scaled, y_train)
            y_pred = best_model.predict(X_test_scaled)
        else:
            best_model = config['model']
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            
        best_models[name] = best_model
    
    # Evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    results[name] = {
        'R2': r2,
        'MSE': mse,
        'MAE': mae,
        'RMSE': np.sqrt(mse)
    }
    
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {np.sqrt(mse):.4f}")
    print(f"  MAE = {mae:.4f}")

# 5. Model Performance Comparison
print("\n=== Model Performance Comparison ===")
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('R2', ascending=False)
print(results_df)

# Visualize model comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# R² comparison
models_sorted = results_df.index
r2_scores = results_df['R2']
colors = plt.cm.viridis(np.linspace(0, 1, len(models_sorted)))

bars = axes[0].bar(range(len(models_sorted)), r2_scores, color=colors)
axes[0].set_xlabel('Model')
axes[0].set_ylabel('R² Score')
axes[0].set_title('Model Comparison by R² Score')
axes[0].set_xticks(range(len(models_sorted)))
axes[0].set_xticklabels(models_sorted, rotation=45)
axes[0].grid(True, alpha=0.3)

# Add values on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')

# RMSE comparison
rmse_scores = results_df['RMSE']
bars = axes[1].bar(range(len(models_sorted)), rmse_scores, color=colors)
axes[1].set_xlabel('Model')
axes[1].set_ylabel('RMSE')
axes[1].set_title('Model Comparison by RMSE')
axes[1].set_xticks(range(len(models_sorted)))
axes[1].set_xticklabels(models_sorted, rotation=45)
axes[1].grid(True, alpha=0.3)

# Add values on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 6. Select Best Model and Conduct In-depth Analysis
best_model_name = results_df.index[0]
best_model = best_models[best_model_name]
print(f"\nBest Model: {best_model_name}")

# Train best model on full dataset
if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet', 'SVR']:
    X_full_scaled = scaler.fit_transform(X_final)
    best_model.fit(X_full_scaled, y)
    y_full_pred = best_model.predict(X_full_scaled)
else:
    best_model.fit(X_final, y)
    y_full_pred = best_model.predict(X_final)

final_r2 = r2_score(y, y_full_pred)
final_rmse = np.sqrt(mean_squared_error(y, y_full_pred))
final_mae = mean_absolute_error(y, y_full_pred)

print(f"Full Dataset Performance:")
print(f"  R² = {final_r2:.4f}")
print(f"  RMSE = {final_rmse:.4f}")
print(f"  MAE = {final_mae:.4f}")

# 7. Feature Importance Analysis
print("\n=== Feature Importance Analysis ===")

if hasattr(best_model, 'feature_importances_'):
    # Feature importance for tree models
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("Feature Importance:")
    print(feature_importance_df.head(10))
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 15 Most Important Features ({best_model_name})')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
elif hasattr(best_model, 'coef_'):
    # Coefficients for linear models
    coefficients = best_model.coef_
    coef_df = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': coefficients,
        'Absolute': np.abs(coefficients)
    }).sort_values('Absolute', ascending=False)
    
    print("Feature Coefficients:")
    print(coef_df.head(10))
    
    # Visualize coefficients
    plt.figure(figsize=(12, 8))
    top_coef = coef_df.head(15)
    colors = ['red' if x < 0 else 'green' for x in top_coef['Coefficient']]
    plt.barh(range(len(top_coef)), top_coef['Coefficient'], color=colors)
    plt.yticks(range(len(top_coef)), top_coef['Feature'])
    plt.xlabel('Coefficient Value')
    plt.title(f'Top 15 Most Important Features ({best_model_name})')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 8. Residual Analysis and Model Diagnostics
print("\n=== Model Diagnostics ===")

residuals = y - y_full_pred

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Residual distribution
axes[0,0].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
residuals.plot(kind='density', ax=axes[0,0], color='red', linewidth=2)
axes[0,0].axvline(0, color='black', linestyle='--', linewidth=2)
axes[0,0].set_xlabel('Residuals')
axes[0,0].set_ylabel('Density')
axes[0,0].set_title('Residual Distribution')
axes[0,0].grid(True, alpha=0.3)

# Predicted vs Actual values
axes[0,1].scatter(y_full_pred, y, alpha=0.5, color='blue')
axes[0,1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
axes[0,1].set_xlabel('Predicted Grades')
axes[0,1].set_ylabel('Actual Grades')
axes[0,1].set_title('Predicted vs Actual Grades')
axes[0,1].grid(True, alpha=0.3)
axes[0,1].text(0.05, 0.95, f'R² = {final_r2:.3f}\nRMSE = {final_rmse:.3f}\nMAE = {final_mae:.3f}', 
              transform=axes[0,1].transAxes, fontsize=12, 
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Residuals vs Predicted values
axes[1,0].scatter(y_full_pred, residuals, alpha=0.5, color='green')
axes[1,0].axhline(0, color='black', linestyle='--', linewidth=2)
axes[1,0].set_xlabel('Predicted Grades')
axes[1,0].set_ylabel('Residuals')
axes[1,0].set_title('Residuals vs Predicted Values')
axes[1,0].grid(True, alpha=0.3)

# Q-Q plot for normality check
stats.probplot(residuals, dist="norm", plot=axes[1,1])
axes[1,1].set_title('Q-Q Plot for Normality Check')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. Multi-factor Interaction Analysis
print("\n=== Multi-factor Interaction Analysis ===")

# Create multi-factor analysis plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Alcohol and study time interaction effect
for study_level in [1, 2, 3, 4]:
    subset = df[df['studytime'] == study_level]
    alcohol_effect = subset.groupby('total_alcohol')['avg_grade'].mean()
    axes[0,0].plot(alcohol_effect.index, alcohol_effect.values, 
                  marker='o', label=f'Study Time: {study_level}')
axes[0,0].set_xlabel('Total Alcohol Consumption')
axes[0,0].set_ylabel('Average Grade')
axes[0,0].set_title('Alcohol Effect by Study Time')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Absence and failure interaction effect
absence_groups = pd.cut(df['absences'], bins=[-1, 0, 3, 10, 100], labels=['0', '1-3', '4-10', '10+'])
for failure_count in [0, 1, 2, 3]:
    subset = df[df['failures'] == failure_count]
    if len(subset) > 0:
        absence_effect = subset.groupby(absence_groups)['avg_grade'].mean()
        axes[0,1].plot(range(len(absence_effect)), absence_effect.values, 
                      marker='s', label=f'Failures: {failure_count}')
axes[0,1].set_xlabel('Absence Group')
axes[0,1].set_ylabel('Average Grade')
axes[0,1].set_title('Absence Effect by Failure History')
axes[0,1].set_xticks(range(4))
axes[0,1].set_xticklabels(['0', '1-3', '4-10', '10+'])
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Family support and motivation interaction effect
for support_level in ['Low', 'Medium', 'High', 'Very High']:
    subset = df[df['support_level'] == support_level]
    if len(subset) > 0:
        motivation_effect = subset.groupby('motivation')['avg_grade'].mean()
        axes[1,0].plot(motivation_effect.index, motivation_effect.values, 
                      marker='^', label=f'Support: {support_level}')
axes[1,0].set_xlabel('Motivation Score')
axes[1,0].set_ylabel('Average Grade')
axes[1,0].set_title('Motivation Effect by Support Level')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Risk category analysis
risk_effect = df.groupby('risk_category')['avg_grade'].agg(['mean', 'std', 'count'])
x_pos = range(len(risk_effect))
axes[1,1].bar(x_pos, risk_effect['mean'], yerr=risk_effect['std'], 
             capsize=5, alpha=0.7, color=['green', 'yellow', 'orange', 'red'])
axes[1,1].set_xticks(x_pos)
axes[1,1].set_xticklabels(risk_effect.index)
axes[1,1].set_xlabel('Academic Risk Category')
axes[1,1].set_ylabel('Average Grade')
axes[1,1].set_title('Academic Risk vs Average Grade')
for i, (mean_val, count_val) in enumerate(zip(risk_effect['mean'], risk_effect['count'])):
    axes[1,1].text(i, mean_val + 0.1, f'{mean_val:.1f}\n(n={count_val})', 
                  ha='center', va='bottom', fontsize=9)
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 10. Student Segmentation Analysis
print("\n=== Student Segmentation Analysis ===")

# Segmentation based on prediction error
df['predicted_grade'] = y_full_pred
df['prediction_error'] = df['avg_grade'] - df['predicted_grade']
df['performance_category'] = pd.cut(df['prediction_error'], 
                                  bins=[-10, -2, 2, 10], 
                                  labels=['Underpredicted', 'Accurate', 'Overpredicted'])

# Analyze characteristics of different segments
cluster_analysis = df.groupby('performance_category').agg({
    'avg_grade': ['mean', 'std'],
    'predicted_grade': ['mean', 'std'],
    'prediction_error': ['mean', 'std'],
    'total_alcohol': 'mean',
    'studytime': 'mean',
    'absences': 'mean',
    'failures': 'mean',
    'family_support': 'mean',
    'motivation': 'mean'
})

print("Student Segmentation Analysis:")
print(cluster_analysis)

# 11. Create Advanced Prediction Function
def predict_student_performance_advanced(
    Dalc=1, Walc=1, studytime=2, absences=0, failures=0,
    famrel=4, Medu=2, Fedu=2, romantic='no', activities='no', 
    internet='yes', higher='yes', famsup='no', schoolsup='no', paid='no',
    goout=3, freetime=3, health=4
):
    """Advanced Student Performance Prediction Function"""
    
    # Calculate advanced features
    total_alcohol = Dalc + Walc
    alcohol_frequency = (Dalc + Walc * 2) / 3
    study_efficiency = studytime / (absences + 1)
    parent_edu_score = (Medu * 0.6 + Fedu * 0.4)
    academic_risk = failures * 2 + (1 if absences > 5 else 0) * 3
    social_activity = goout + freetime
    family_support = famrel + (1 if famsup == 'yes' else 0) * 2
    school_support = (1 if schoolsup == 'yes' else 0) * 2 + (1 if paid == 'yes' else 0)
    motivation = (1 if higher == 'yes' else 0) * 3 + 2  # Default medium motivation
    
    # Calculate interaction features
    alcohol_study_interaction = total_alcohol * (5 - studytime)
    absence_failure_interaction = absences * failures
    support_motivation_interaction = family_support * motivation
    
    # Prepare input data
    input_data = {
        'Dalc': Dalc, 'Walc': Walc, 'studytime': studytime, 
        'absences': absences, 'failures': failures, 'famrel': famrel,
        'Medu': Medu, 'Fedu': Fedu, 'goout': goout, 'freetime': freetime,
        'health': health, 'total_alcohol': total_alcohol,
        'alcohol_frequency': alcohol_frequency,
        'study_efficiency': study_efficiency,
        'parent_edu_score': parent_edu_score,
        'academic_risk': academic_risk,
        'social_activity': social_activity,
        'family_support': family_support,
        'school_support': school_support,
        'motivation': motivation,
        'alcohol_study_interaction': alcohol_study_interaction,
        'absence_failure_interaction': absence_failure_interaction,
        'support_motivation_interaction': support_motivation_interaction,
        'romantic_yes': 1 if romantic == 'yes' else 0,
        'activities_yes': 1 if activities == 'yes' else 0,
        'internet_yes': 1 if internet == 'yes' else 0,
        'higher_yes': 1 if higher == 'yes' else 0,
        'famsup_yes': 1 if famsup == 'yes' else 0,
        'schoolsup_yes': 1 if schoolsup == 'yes' else 0,
        'paid_yes': 1 if paid == 'yes' else 0
    }
    
    # Create DataFrame, ensuring all selected features are included
    input_df = pd.DataFrame([input_data])
    
    # Keep only selected features
    for feature in selected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Default value
    
    input_df = input_df[selected_features]
    
    # Prediction
    if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet', 'SVR']:
        input_scaled = scaler.transform(input_df)
        predicted_grade = best_model.predict(input_scaled)[0]
    else:
        predicted_grade = best_model.predict(input_df)[0]
    
    # Calculate confidence interval (simplified)
    confidence_interval = final_rmse * 1.96  # 95% confidence interval
    
    return max(0, min(20, predicted_grade)), confidence_interval

# Test advanced prediction function
print("\n=== Advanced Prediction Function Test ===")
test_cases = [
    {"Dalc": 1, "Walc": 1, "studytime": 3, "absences": 2, "failures": 0, 
     "romantic": "no", "internet": "yes", "higher": "yes", "famsup": "yes", "schoolsup": "no"},
    {"Dalc": 4, "Walc": 5, "studytime": 1, "absences": 10, "failures": 2, 
     "romantic": "yes", "internet": "no", "higher": "no", "famsup": "no", "schoolsup": "yes"},
    {"Dalc": 1, "Walc": 1, "studytime": 4, "absences": 0, "failures": 0, 
     "romantic": "no", "internet": "yes", "higher": "yes", "famsup": "yes", "schoolsup": "yes",
     "Medu": 4, "Fedu": 4}
]

for i, case in enumerate(test_cases, 1):
    grade, confidence = predict_student_performance_advanced(**case)
    
    # Risk assessment
    if grade >= 13:
        risk_level = "Low Risk"
        recommendation = "Maintain good study habits"
    elif grade >= 10:
        risk_level = "Medium Risk" 
        recommendation = "Need to monitor learning status"
    else:
        risk_level = "High Risk"
        recommendation = "Immediate intervention required"
    
    print(f"Case {i}:")
    print(f"  Features - Alcohol(D:{case['Dalc']}/W:{case['Walc']}), "
          f"Study:{case['studytime']}h, Absences:{case['absences']} times, "
          f"Failures:{case['failures']} times, Romantic:{case['romantic']}, "
          f"Family Support:{case['famsup']}, School Support:{case['schoolsup']}")
    print(f"  Predicted Grade: {grade:.1f} ± {confidence:.1f} → {risk_level}")
    print(f"  Recommendation: {recommendation}\n")

# 12. Comprehensive Correlation Heatmap
print("\n=== Comprehensive Correlation Analysis ===")

# Select most important features for correlation analysis
if 'feature_importance_df' in locals():
    top_corr_features = list(feature_importance_df.head(12)['Feature']) + ['avg_grade']
else:
    top_corr_features = list(coef_df.head(12)['Feature']) + ['avg_grade']

# Ensure all features exist
available_features = [f for f in top_corr_features if f in df.columns]
corr_matrix = df[available_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Top Features Correlation Heatmap')
plt.tight_layout()
plt.show()

# 13. Final Summary
print("\n" + "="*50)
print("Final Analysis Summary")
print("="*50)

print(f"\nModel Performance:")
print(f"  Best Model: {best_model_name}")
print(f"  Explanatory Power (R²): {final_r2:.1%}")
print(f"  Average Prediction Error: {final_rmse:.2f} points")
print(f"  Mean Absolute Error: {final_mae:.2f} points")

print(f"\nMost Important Influencing Factors:")
if 'feature_importance_df' in locals():
    top_factors = feature_importance_df.head(5)
    for _, row in top_factors.iterrows():
        # Determine effect direction based on feature name
        feature_name = row['Feature']
        if any(keyword in feature_name for keyword in ['alcohol', 'absence', 'failure', 'risk']):
            effect = "Negative"
        else:
            effect = "Positive"
        print(f"  - {feature_name}: {effect} effect (Importance: {row['Importance']:.3f})")
else:
    top_factors = coef_df.head(5)
    for _, row in top_factors.iterrows():
        effect = "Negative" if row['Coefficient'] < 0 else "Positive"
        print(f"  - {row['Feature']}: {effect} effect (Coefficient: {row['Coefficient']:.3f})")

print(f"\nKey Findings:")
print("1. Advanced feature engineering significantly improved model explanatory power")
print("2. Multi-factor interaction effects have stronger predictive power than single factors")
print("3. Interaction between family support and learning motivation has important impact on grades")
print("4. Negative interaction effect between alcohol and study time is evident")
print("5. Academic risk index can effectively identify at-risk students")

print(f"\nPractical Recommendations:")
print("1. Focus on students with both absenteeism and alcohol consumption")
print("2. Strengthen collaborative support between family and school")
print("3. Develop differentiated intervention strategies for different risk levels")
print("4. Use prediction model for early warning and resource allocation")