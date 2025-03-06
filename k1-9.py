# Importaciones corregidas y organizadas
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import torch
import shap
from itertools import combinations
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import roc_curve, auc, make_scorer
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
)
from joblib import parallel_backend

from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from statsmodels.graphics.tsaplots import plot_acf


# Función para optimizar el uso de memoria
def optimize_dtypes(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype('category')
        elif df_copy[col].dtype == 'float64':
            df_copy[col] = df_copy[col].astype('float32')
        elif df_copy[col].dtype == 'int64':
            df_copy[col] = df_copy[col].astype('int32')
    return df_copy


# Función para reemplazar nulos con valores por defecto
def replace_nulls_with_default(df, float_default=-1.0, object_default="Missing"):
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=['float', 'int']).columns:
        df_copy.loc[:, col] = df_copy[col].replace("", float_default).fillna(float_default)
    
    for col in df_copy.select_dtypes(include=['object', 'category']).columns:
        if df_copy[col].dtype.name == 'category':
            if object_default not in df_copy[col].cat.categories:
                df_copy[col] = df_copy[col].cat.add_categories(object_default)
        df_copy.loc[:, col] = df_copy[col].replace("", object_default).fillna(object_default)
    
    return df_copy


# Función para calcular la probabilidad de supervivencia
def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], event_observed=df[event_col])
    survival_probabilities = kmf.survival_function_at_times(df[time_col]).values.flatten()
    return survival_probabilities

# Función para calcular el índice C estratificado (corregida)
def stratified_c_index(y_true, y_pred, groups):
    unique_groups = np.unique(groups)
    c_indices = []

    for group in unique_groups:
        mask = groups == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        if len(y_true_group) < 2:
            continue
        
        pairs = np.array(list(combinations(range(len(y_true_group)), 2)))
        y_true_pairs = y_true_group[pairs]
        y_pred_pairs = y_pred_group[pairs]
        
        true_diff = y_true_pairs[:, 0] - y_true_pairs[:, 1]
        pred_diff = y_pred_pairs[:, 0] - y_pred_pairs[:, 1]
        
        permissible = true_diff != 0
        concordant = (true_diff * pred_diff) > 0
        
        c_index = np.sum(concordant & permissible) / np.sum(permissible)
        c_indices.append(c_index)

    return np.mean(c_indices) - np.std(c_indices)

def custom_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return stratified_c_index(y.values, y_pred, race_groups.values[X.index])

# Optimizar el uso de memoria
df_tr = optimize_dtypes(df_tr)
df_ts = optimize_dtypes(df_ts)


# Reemplazar nulos con valores por defecto
df_tr = replace_nulls_with_default(df_tr)
df_ts = replace_nulls_with_default(df_ts)

# Calcular y añadir la columna objetivo
df_tr["target"] = np.sqrt(transform_survival_probability(df_tr, time_col='efs_time', event_col='efs'))

# Eliminar columnas irrelevantes
drop_cols = ["ID", 'efs', 'efs_time']
df_tr = df_tr.drop(columns=[col for col in drop_cols if col in df_tr.columns])
df_ts = df_ts.drop(columns=[col for col in drop_cols if col in df_ts.columns])

# Preparar los datos para el modelo
X = df_tr.drop(columns=['target'], axis=1)
y = df_tr['target']
race_groups = df_tr['race_group']

# Identificar columnas categóricas
cat_features = list(X.select_dtypes(include=['object', 'category']).columns)

# Convertir columnas categóricas a tipo string
for col in cat_features:
    X[col] = X[col].astype(str)
    df_ts[col] = df_ts[col].astype(str)

# Aplicar PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X.select_dtypes(include=['float32', 'int32']))

# Aplicar t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X.select_dtypes(include=['float32', 'int32']))

# Añadir las nuevas características al DataFrame original
X_new = pd.concat([X, 
                   pd.DataFrame(X_pca, columns=[f'PCA_{i}' for i in range(10)], index=X.index),
                   pd.DataFrame(X_tsne, columns=['TSNE_1', 'TSNE_2'], index=X.index)], 
                   axis=1)

# Actualizar la lista de características categóricas
cat_features = list(X_new.select_dtypes(include=['object', 'category']).columns)


# Configurar validación cruzada con KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Función objetivo para la búsqueda amplia (más agresiva)
def objective_wide(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 10000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1.0, log=True),
        'depth': trial.suggest_int('depth', 2, 16),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-10, 100.0, log=True),
        'border_count': trial.suggest_categorical('border_count', [32, 64, 128, 255, 512]),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 2),
        'random_strength': trial.suggest_float('random_strength', 1e-10, 100.0, log=True),
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'task_type': 'GPU',
        'devices': '0,1',
        'verbose': False,
        'cat_features': cat_features,
        'used_ram_limit': '10gb',
        'allow_writing_files': False
    }

    model = CatBoostRegressor(**params)

    try:
        score = cross_val_score(model, X_new, y, cv=kf, scoring=custom_scorer, error_score="raise").mean()
        print(f"Trial completed with score: {score}")
        return score
    except Exception as e:
        print(f"Exception encountered: {e}")
        return float('-inf')

# Crear el estudio de Optuna y optimizar (búsqueda amplia)
study_wide = optuna.create_study(direction='maximize')

# Usar parallel_backend aquí
with parallel_backend('loky', n_jobs=-1):
    study_wide.optimize(objective_wide, n_trials=1000, timeout=14400)  # 4 horas de tiempo límite

# Mejores parámetros y puntuación de la búsqueda amplia
best_params_wide = study_wide.best_params
print('Mejores parámetros (búsqueda amplia):', best_params_wide)
print('Mejor puntuación (búsqueda amplia):', study_wide.best_value)

# Guardar los mejores parámetros de la búsqueda amplia
np.save('best_params_wide.npy', best_params_wide)


# Mejores parámetros y puntuación de la búsqueda amplia
best_params_wide = study_wide.best_params
print('Mejores parámetros (búsqueda amplia):', best_params_wide)
print('Mejor puntuación (búsqueda amplia):', study_wide.best_value)

# Definir el espacio de búsqueda fina (más agresiva)
def objective_fine(trial):
    params = {
        'iterations': trial.suggest_int('iterations', max(100, int(best_params_wide['iterations']*0.5)), min(15000, int(best_params_wide['iterations']*1.5))),
        'learning_rate': trial.suggest_float('learning_rate', max(1e-6, best_params_wide['learning_rate']*0.5), min(1.0, best_params_wide['learning_rate']*1.5), log=True),
        'depth': trial.suggest_int('depth', max(1, best_params_wide['depth']-3), min(20, best_params_wide['depth']+3)),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', max(1e-11, best_params_wide['l2_leaf_reg']*0.5), min(200.0, best_params_wide['l2_leaf_reg']*1.5), log=True),
        'border_count': trial.suggest_categorical('border_count', [32, 64, 128, 255, 512]),
        'bagging_temperature': trial.suggest_float('bagging_temperature', max(0, best_params_wide['bagging_temperature']*0.5), min(3, best_params_wide['bagging_temperature']*1.5)),
        'random_strength': trial.suggest_float('random_strength', max(1e-11, best_params_wide['random_strength']*0.5), min(200.0, best_params_wide['random_strength']*1.5), log=True),
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'task_type': 'GPU',
        'devices': '0,1',
        'verbose': False,
        'cat_features': cat_features,
        'used_ram_limit': '10gb',
        'allow_writing_files': False
    }

    model = CatBoostRegressor(**params)

    try:
        score = cross_val_score(model, X_new, y, cv=kf, scoring=custom_scorer, error_score="raise").mean()
        print(f"Fine-tuning trial completed with score: {score}")
        return score
    except Exception as e:
        print(f"Exception encountered in fine-tuning: {e}")
        return float('-inf')

# Crear el estudio de Optuna y optimizar (búsqueda fina)
study_fine = optuna.create_study(direction='maximize')

# Usar parallel_backend aquí
with parallel_backend('loky', n_jobs=-1):
    study_fine.optimize(objective_fine, n_trials=400, timeout=10400)  # 3 hora de tiempo límite

# Mejores parámetros y puntuación de la búsqueda fina
best_params_fine = study_fine.best_params
print('Mejores parámetros (búsqueda fina):', best_params_fine)
print('Mejor puntuación (búsqueda fina):', study_fine.best_value)

# Comparar resultados
print('Mejora en la puntuación:', study_fine.best_value - study_wide.best_value)

# Comparar resultados y seleccionar los mejores parámetros
if study_fine.best_value > study_wide.best_value:
    print("La búsqueda fina mejoró los resultados.")
    best_params = best_params_fine
else:
    print("La búsqueda amplia dio mejores resultados.")
    best_params = best_params_wide

# Guardar los mejores parámetros finales
np.save('best_params.npy', best_params)

print('Mejores parámetros finales:', best_params)
print('Mejor puntuación final:', max(study_fine.best_value, study_wide.best_value))

# Intentar cargar los mejores parámetros, si el archivo existe
if os.path.exists('best_params.npy'):
    best_params = np.load('best_params.npy', allow_pickle=True).item()
else:
    print("Advertencia: No se encontró el archivo 'best_params.npy'. Usando los parámetros en memoria.")
 

# Initialize variables for predictions and scores
final_predictions = np.zeros(len(X))
fold_scores = []


for fold, (train_idx, val_idx) in enumerate(kf.split(X, race_groups)):
    print(f"Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    race_val = race_groups.iloc[val_idx]

    # Set up CatBoost model with best parameters
    model = CatBoostRegressor(
        **best_params,
        random_seed=42,
        task_type='GPU',
        devices='0,1',  # Usa GPUs en serie
        verbose=False,
        cat_features=cat_features,
        gpu_ram_part=0.8,
        allow_writing_files=False
    )

    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )

    # Predict and compute metric
    y_val_pred = model.predict(X_val)
    fold_score = stratified_c_index(y_val.values, y_val_pred, race_val.values)
    fold_scores.append(fold_score)
    print(f"Stratified C-Index for Fold {fold + 1}: {fold_score}")

    final_predictions[val_idx] = y_val_pred

# Overall Stratified Concordance Index
overall_score = stratified_c_index(y.values, final_predictions, race_groups.values)
print(f"Overall Stratified C-Index: {overall_score}")

# Calculate standard C-index
c_index = concordance_index(y, final_predictions)
print(f"Standard C-Index: {c_index}")

# Plot the scores per fold
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(fold_scores) + 1), fold_scores, marker='o', linestyle='-', color='b')
plt.title('Stratified C-Index by Fold')
plt.xlabel('Fold')
plt.ylabel('Stratified C-Index')
plt.xticks(range(1, len(fold_scores) + 1))
plt.grid()
plt.show()

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y, final_predictions, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.tight_layout()
plt.show()

 ##########
 
 # SHAP Analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualize a specific prediction
shap.force_plot(
    explainer.expected_value, 
    shap_values[0, :], 
    X.iloc[0, :],
    matplotlib=True,
    show=False
)
plt.savefig('shap_force_plot.png')
plt.close()

# SHAP summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig('shap_summary_plot.png')
plt.close()

# SHAP bar plot for feature importance
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_feature_importance.png')
plt.close()

# Final predictions on the test set
pred = model.predict(df_ts)


# SHAP Analysis
# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualize a specific prediction
shap.force_plot(
    explainer.expected_value, 
    shap_values[0, :], 
    X.iloc[0, :],
    matplotlib=True
)

# SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X, show=True)

# SHAP bar plot for feature importance
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=True)

# Final predictions on the test set 
pred = model.predict(df_ts)



# Importancia de características
feature_importance = model.get_feature_importance()
feature_names = X.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 10))  # Aumentamos el tamaño de la figura
sns.barplot(x='importance', y='feature', data=importance_df)
plt.title('Importancia de características')
plt.xticks(rotation=45, ha='right')  # Rotamos las etiquetas del eje x
plt.tight_layout()  # Ajustamos el diseño para evitar que se corten las etiquetas
plt.show()

# Distribución de predicciones por race_group
plt.figure(figsize=(14, 6))
sns.boxplot(x='race_group', y='target', data=df_tr)
plt.title('Distribución de target por race_group')
plt.xticks(rotation=45, ha='right')  # Rotamos las etiquetas del eje x
plt.tight_layout()  # Ajustamos el diseño para evitar que se corten las etiquetas
plt.show()

# Distribución de predicciones por ethnicity
plt.figure(figsize=(14, 6))
sns.boxplot(x='ethnicity', y='target', data=df_tr)
plt.title('Distribución de target por ethnicity')
plt.xticks(rotation=45, ha='right')  # Rotamos las etiquetas del eje x
plt.tight_layout()  # Ajustamos el diseño para evitar que se corten las etiquetas
plt.show()


# Gráfico de violín para race_group
plt.figure(figsize=(14, 6))
sns.violinplot(x='race_group', y='target', data=df_tr)
plt.title('Distribución de target por race_group')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Gráfico de violín para ethnicity
plt.figure(figsize=(14, 6))
sns.violinplot(x='ethnicity', y='target', data=df_tr)
plt.title('Distribución de target por ethnicity')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ANOVA
race_groups = df_tr['race_group'].unique()
race_data = [df_tr[df_tr['race_group'] == group]['target'] for group in race_groups]
f_value, p_value = stats.f_oneway(*race_data)
print(f"ANOVA para race_group: F-value = {f_value}, p-value = {p_value}")

# Gráfico de dispersión
plt.figure(figsize=(10, 6))
scatter = plt.scatter(y, final_predictions, c=df_tr['race_group'].astype('category').cat.codes, cmap='viridis')
plt.colorbar(scatter)
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs. Valores reales (coloreado por race_group)')
plt.show()

# Heatmap de predicciones promedio por race_group y ethnicity
pivot_table = df_tr.pivot_table(values='target', index='race_group', columns='ethnicity', aggfunc='mean')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.2f')
plt.title('Predicciones promedio por race_group y ethnicity')
plt.tight_layout()
plt.show()



# Calcular la moda y el decil más alto de sobrevida
moda_sobrevida = df_tr['target'].mode().values[0]
decil_alto_sobrevida = df_tr['target'].quantile(0.9)

# Obtener los índices de las muestras cercanas a la moda y al decil alto
indices_moda = np.abs(df_tr['target'] - moda_sobrevida).argsort()[:100]
indices_decil_alto = np.abs(df_tr['target'] - decil_alto_sobrevida).argsort()[:100]

# Calcular los valores SHAP para estas muestras
explainer = shap.TreeExplainer(model)
shap_values_moda = explainer.shap_values(X.iloc[indices_moda])
shap_values_decil_alto = explainer.shap_values(X.iloc[indices_decil_alto])

# Crear las gráficas
plt.figure(figsize=(20, 10))

# Gráfica para la moda
plt.subplot(1, 2, 1)
shap.summary_plot(shap_values_moda, X.iloc[indices_moda], plot_type="bar", show=False)
plt.title("Variables con mayor impacto en torno a la moda de sobrevida")
plt.tight_layout()

# Gráfica para el decil alto
plt.subplot(1, 2, 2)
shap.summary_plot(shap_values_decil_alto, X.iloc[indices_decil_alto], plot_type="bar", show=False)
plt.title("Variables con mayor impacto en torno al decil más alto de sobrevida")
plt.tight_layout()
plt.show()




def survival_roc_curve(y_true, y_pred, num_thresholds=100):
    thresholds = np.linspace(y_pred.min(), y_pred.max(), num_thresholds)
    tpr = []
    fpr = []
    
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        tp = np.sum((y_true >= y_true.mean()) & (y_pred_binary == 1))
        fp = np.sum((y_true < y_true.mean()) & (y_pred_binary == 1))
        tn = np.sum((y_true < y_true.mean()) & (y_pred_binary == 0))
        fn = np.sum((y_true >= y_true.mean()) & (y_pred_binary == 0))
        
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
    
    return np.array(fpr), np.array(tpr), thresholds

# Calcular la curva ROC adaptada
fpr, tpr, thresholds = survival_roc_curve(y, final_predictions)

# Calcular el AUC
roc_auc = roc_auc_score(y >= y.mean(), final_predictions)

# Graficar la curva ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC para Predicciones de Supervivencia')
plt.legend(loc="lower right")
plt.show()


# Análisis de residuos
residuals = y - final_predictions
plt.figure(figsize=(10,6))
plt.scatter(final_predictions, residuals)
plt.xlabel('Predicciones de Supervivencia')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos para Predicciones de Supervivencia')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()



#  Distribución de residuos
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribución de Residuos')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.show()

#  QQ plot de residuos

plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q plot de Residuos")
plt.show()

#  Residuos vs variables predictoras
for column in X.columns[:5]:  # Limitamos a las primeras 5 variables para no saturar
    plt.figure(figsize=(10, 6))
    plt.scatter(X[column], residuals)
    plt.xlabel(column)
    plt.ylabel('Residuos')
    plt.title(f'Residuos vs {column}')
    plt.show()

#  Autocorrelación de residuos

plt.figure(figsize=(10, 6))
plot_acf(residuals)
plt.title('Autocorrelación de Residuos')
plt.show()

#  Gráfico de dispersión de predicciones vs valores reales con línea de identidad
plt.figure(figsize=(10, 6))
plt.scatter(y, final_predictions, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')
plt.show()

#  Distribución de las predicciones vs distribución de los valores reales
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(y, kde=True)
plt.title('Distribución de Valores Reales')
plt.subplot(1, 2, 2)
sns.histplot(final_predictions, kde=True)
plt.title('Distribución de Predicciones')
plt.tight_layout()
plt.show()

#  Gráfico de barras de error
sorted_indices = y.argsort()
plt.figure(figsize=(12, 6))
plt.errorbar(range(len(y)), y.iloc[sorted_indices], 
             yerr=np.abs(residuals.iloc[sorted_indices]), 
             fmt='o', alpha=0.5)
plt.xlabel('Índice de muestra (ordenado)')
plt.ylabel('Valor')
plt.title('Gráfico de Barras de Error')
plt.show()

# Cargar los datos desde los archivos CSV
submission = pd.read_csv('submission.csv')
print(submission)

######### 

paciente_hipotetico = {
    'dri_score': 'Intermediate',
    'psych_disturb': 'No',
    'cyto_score': 'Intermediate',
    'diabetes': 'No',
    'hla_match_c_high': 2,
    'hla_high_res_8': 8,
    'tbi_status': 'No TBI',
    'arrhythmia': 'No',
    'hla_low_res_6': 6,
    'graft_type': 'Peripheral blood',
    'vent_hist': 'No',
    'renal_issue': 'No',
    'pulm_severe': 'No',
    'prim_disease_hct': 'AML',
    'hla_high_res_6': 6,
    'cmv_status': '+/+',
    'hla_high_res_10': 10,
    'hla_match_dqb1_high': 2,
    'tce_imm_match': 'P/P',
    'hla_nmdp_6': 6,
    'hla_match_c_low': 2,
    'rituximab': 'No',
    'hla_match_drb1_low': 2,
    'hla_match_dqb1_low': 2,
    'prod_type': 'PB',
    'cyto_score_detail': 'Intermediate',
    'conditioning_intensity': 'MAC',
    'ethnicity': 'Not Hispanic or Latino',
    'year_hct': 2016,
    'obesity': 'No',
    'mrd_hct': 'No',
    'in_vivo_tcd': 'Yes',
    'tce_match': 'P/P',
    'hla_match_a_high': 2,
    'hepatic_severe': 'No',
    'donor_age': 30,
    'prior_tumor': 'No',
    'hla_match_b_low': 2,
    'peptic_ulcer': 'No',
    'age_at_hct': 50,
    'hla_match_a_low': 2,
    'gvhd_proph': 'FKalone',
    'rheum_issue': 'No',
    'sex_match': 'M-F',
    'hla_match_b_high': 2,
    'race_group': 'White',
    'comorbidity_score': 0,
    'karnofsky_score': 90,
    'hepatic_mild': 'No',
    'tce_div_match': 'P/P',
    'donor_related': 'Unrelated',
    'melphalan_dose': 'N/A, Mel not given',
    'hla_low_res_8': 8,
    'cardiac': 'No',
    'hla_match_drb1_high': 2,
    'pulm_moderate': 'No',
    'hla_low_res_10': 10
}

# Crear el DataFrame
paciente_hipotetico_df = pd.DataFrame([paciente_hipotetico])

print(paciente_hipotetico_df)



# Crear el DataFrame del paciente hipotético
paciente_hipotetico_df = pd.DataFrame([paciente_hipotetico])

# Asegúrate de que tienes todas las columnas necesarias en paciente_hipotetico_df
columnas_necesarias = X.columns
for col in columnas_necesarias:
    if col not in paciente_hipotetico_df.columns:
        paciente_hipotetico_df[col] = 0  # o cualquier otro valor por defecto

# Hacer la predicción para el paciente hipotético
prediccion = model.predict(paciente_hipotetico_df[columnas_necesarias])

print(f"Predicción para el paciente hipotético: {prediccion[0]:.4f}")

# Calcular estadísticas descriptivas de las predicciones en el conjunto de entrenamiento
predicciones_train = model.predict(X)
print("\nEstadísticas descriptivas de las predicciones en el conjunto de entrenamiento:")
print(pd.Series(predicciones_train).describe())

# Calcular percentiles para la predicción del paciente hipotético
percentil = stats.percentileofscore(predicciones_train, prediccion[0])
print(f"\nLa predicción del paciente hipotético está en el percentil {percentil:.2f} del conjunto de entrenamiento.")

# Visualizar la distribución de las predicciones y la posición del paciente hipotético
plt.figure(figsize=(10, 6))
sns.histplot(predicciones_train, kde=True)
plt.axvline(prediccion[0], color='r', linestyle='--', label='Paciente hipotético')
plt.title('Distribución de predicciones y posición del paciente hipotético')
plt.xlabel('Predicción')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

##########
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(paciente_hipotetico_df[columnas_necesarias])

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, paciente_hipotetico_df[columnas_necesarias], plot_type="bar", show=False)
plt.title("Importance of characteristics for the prediction of the hypothetical patient")
plt.tight_layout()
plt.show()

# SHAP force plot
plt.figure(figsize=(20, 3))
shap.force_plot(explainer.expected_value, shap_values[0], paciente_hipotetico_df[columnas_necesarias].iloc[0], matplotlib=True, show=False)
plt.title("SHAP force plot")
plt.tight_layout()
plt.show()

 ####### 

def stratified_c_index(y_true, y_pred, groups):
    unique_groups = np.unique(groups)
    c_indices = []

    for group in unique_groups:
        mask = groups == group
        y_true_group = y_true[mask].values
        y_pred_group = y_pred[mask].values
        
        if len(y_true_group) < 2:
            continue
        
        pairs = list(combinations(range(len(y_true_group)), 2))
        y_true_pairs = np.array([y_true_group[i] - y_true_group[j] for i, j in pairs])
        y_pred_pairs = np.array([y_pred_group[i] - y_pred_group[j] for i, j in pairs])
        
        permissible = y_true_pairs != 0
        concordant = (y_true_pairs * y_pred_pairs) > 0
        
        c_index = np.sum(concordant & permissible) / np.sum(permissible)
        c_indices.append(c_index)

    return np.mean(c_indices) - np.std(c_indices)

# Make sure y, final_predictions, and race_groups have the same length as df_tr
y = df_tr['target']  # Using 'target' as the target variable
race_groups = df_tr['race_group']

# If final_predictions is not defined or not the same length as df_tr, you may need to recalculate it
# For example:
# final_predictions = your_model.predict(df_tr[feature_columns])

# Now use the function
for grupo in df_tr['race_group'].unique():
    mask = df_tr['race_group'] == grupo
    score = stratified_c_index(y[mask], pd.Series(final_predictions)[mask], race_groups[mask])
    print(f"Índice C estratificado para {grupo}: {score}")


###

top_features = importance_df['feature'].head(5).tolist()
for i in range(len(top_features)):
    for j in range(i+1, len(top_features)):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_tr, x=top_features[i], y=top_features[j], hue='target')
        plt.title(f'Interacción entre {top_features[i]} y {top_features[j]}')
        plt.show()
        
###

# Save predictions to the submission file
df_s['prediction'] = pred
df_s.to_csv('submission.csv', index=False)

print("Predictions saved to 'submission.csv'.")

R: 0.74 

