# %% [markdown]
## Modelo de prediccion de abandono de clientes para empresa Interconnect
# ---
# Al operador de telecomunicaciones Interconnect le gustaría poder pronosticar su tasa de cancelación de clientes. Si se descubre que un usuario o usuaria planea irse, se le ofrecerán códigos promocionales y opciones de planes especiales. El equipo de marketing de Interconnect ha recopilado algunos de los datos personales de sus clientes, incluyendo información sobre sus planes y contratos.
# 
# Nuestra meta es construir un modelo que nos ayude a predecir si un cliente esta en riesgo de cancelar el servicio, Interconnect nos pide obtener una metrica igual o mayor a 0.88 en la curva AUC-ROC.

# %% [markdown]
## Índice

# - [Inicialización](#Inicializacion)
#    - [Corrección de datos para dataset de contratos](#Correccion-de-datos-para-dataset-de-contratos)
#    - [Corrección de datos para dataset de internet](#Correccion-de-datos-para-dataset-de-internet)
#    - [Corrección de datos para dataset de info personal](#Correccion-de-datos-para-dataset-de-info-personal)
#    - [Corrección de datos para dataset de phone](#Correccion-de-datos-para-dataset-de-phone)
# - [Creación de dataset mayor](#Creacion-de-dataset-mayor)
#    - [Llenado de valores ausentes](#Llenado-de-valores-ausentes)
# - [Feature Engineering](#Feature-Engineering)
#    - [Codificación](#Codificacion)
#    - [Datasets para entrenamiento, validacion y prueba](#Datasets-para-entrenamiento,-validacion-y-prueba)
# - [Entrenamiento de modelos](#Entrenamiento-de-modelos)
#    - [Modelo 1 - Regresión logística](#Modelo-1-Regresion-logistica)
#    - [Modelo 2 - Bosque aleatorio](#Modelo-2-Bosque-aleatorio)
#    - [Modelo 3 - LightGBM](#Modelo-3-LightGBM)
#    - [Modelo 4 - XGBoost](#Modelo-4-XGBoost)
#    - [Modelo 5 - CatBoost](#Modelo-5-CatBoost)
# - [Análisis de resultados](#Analisis-de-resultados)
# - [Selección del mejor modelo y validación con dataset de pruebas](#Seleccion-del-mejor-modelo-y-validacion-con-dataset-de-pruebas)

# %% [markdown]
##  Inicializacion

# %% 
import warnings
# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb

# %%
df_contracts = pd.read_csv(r"C:\Users\luisP\OneDrive\Documentos\Proyecto Final\contract.csv")
df_internet = pd.read_csv(r"C:\Users\luisP\OneDrive\Documentos\Proyecto Final\internet.csv")
df_personal = pd.read_csv(r"C:\Users\luisP\OneDrive\Documentos\Proyecto Final\personal.csv")
df_phone = pd.read_csv(r"C:\Users\luisP\OneDrive\Documentos\Proyecto Final\phone.csv")

# %% [markdown]
## Correccion de datos para dataset de contratos

# %%
df_contracts.info()

# %%
df_contracts.head()

# %%
df_contracts.duplicated().sum()

# %%
# df_contracts['TotalCharges'] = df_contracts['TotalCharges'].astype('float64')

# %%
df_contracts[df_contracts['TotalCharges'] == ' '] 

# %%
df_contracts['TotalCharges'] = df_contracts['TotalCharges'].where(df_contracts['TotalCharges'] != ' ', df_contracts['MonthlyCharges'])

# %%
df_contracts[df_contracts['TotalCharges'] == ' ']

# %%
df_contracts['TotalCharges'] = df_contracts['TotalCharges'].astype('float64')

# %%
df_contracts[df_contracts['customerID'] == '4472-LVYGI']

# %% [markdown]
# Descubrimos que la columna 'TotalCharges' estaba catalogada como objeto, ya que tenia strings vacios. Para corregir lo anterior utilizamos el metodo where, el cual nos ayuda a ubicar los strings y sustituirlos por el valor que escojamos, en este caso fue por el valor de la columna 'MonthlyCharges'. Un vez realizado lo anterior validamos que se hayan realizado los cambios mencionados, buscamos el ID del cliente y verificamos que el cambio fue realizado con exito. 

# %%
df_contracts['BeginDate'] = pd.to_datetime(df_contracts['BeginDate'])

df_contracts['Month'] = df_contracts['BeginDate'].dt.month
df_contracts['Year'] = df_contracts['BeginDate'].dt.year

df_contracts = df_contracts.drop('BeginDate', axis=1)

# %% [markdown]
# Aqui hicimos un simple cambio de tipo de datos en la columna 'BeginDate', de objeto a datetime, esto con la idea de crear 2 nuevas columnas 'Month' y 'Year'. 

# %% [markdown]
## Correccion de datos para dataset de internet

# %%
df_internet.info()

# %%
df_internet.head()

# %%
df_internet.duplicated().sum()

# %% [markdown]
## Correccion de datos para dataset de info personal

# %%
df_personal.info()

# %%
df_personal.head()

# %%
df_personal.duplicated().sum()

# %% [markdown]
## Correccion de datos para dataset de phone

# %%
df_phone.info()

# %%
df_phone.head()

# %%
df_phone.duplicated().sum()

# %% [markdown]
## Creacion de dataset mayor
# ---
# Tenemos diferentes datasets, cada uno cuenta con la columna customerID la cual contiene un código único asignado a cada cliente, este nos servira para juntar todo los dataset y crear uno con el cual vamos a trabajar para mas adelante construir nuestro modelo de prediccion.

# %%
df_merge = df_contracts.merge(df_personal, on='customerID')
df_merge = df_merge.merge(df_phone, on='customerID', how='outer')
df = df_merge.merge(df_internet, on='customerID', how='outer')

df

# %%
df.duplicated().sum()

# %% [markdown]
## Llenado de valores ausentes
#---
# Ya que unimos los datasets de manera externa quedaron demasiadas filas con valores nulos. A continuacion veremos la cantidad extacta y como seran rellenados.

# %%
df.isna().sum()

# %%
df[df['MultipleLines'].isna()]

# %%
df[df['InternetService'].isna()]

# %%
df['InternetService'].fillna('No service', inplace=True)

df.fillna('No', inplace=True)

# %% [markdown]
# Rellenamos los valores ausententes siguiendo la logica de que si hay un dato faltante es porque no ha sido solocitado el servicio, hay que recordar que los datasets originales de internet y phone contaban con 5,517 y 6,361 clientes registrados respectivamente.

# %%
df.sample(10)

# %% [markdown]
# Tenemos nuestro dataset listo, podemos continuar.

# %% [markdown]
## Feature Engineering
# ---
# Vamos a utilizar diferentes tecnicas para que nuestro dataset tenga informacion de calidad y pueda ser utilizado para entrenar un modelo de aprendizaje automatico.

# %%
df['EndDate'] = df['EndDate'].where(df['EndDate'] == 'No', 1)
df['EndDate'].replace('No', 0, inplace=True)

# %%
df.head()

# %%
target = df['EndDate']
features = df.drop(['EndDate', 'customerID'], axis=1)

# %% [markdown]
## Codificacion
# ---
# En este dataset existen columnas categoricas que cuentan con 2 valores (Yes y No), por lo que no considero necesario utilizar el método One-Hot para codificar los valores en ellas, en su lugar vamos a sustuir por 1 los valores afirmativos y 0 los negativos.

# %%
yesno_col = ['PaperlessBilling', 'Partner', 'Dependents', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for col in yesno_col:
    features[col].replace('Yes', '1', inplace=True)
    features[col].replace('No', '0', inplace=True)
    features[col] = features[col].astype('int')

# %%
cat_col = ['PaymentMethod', 'Type', 'InternetService', 'gender']

dummies = pd.get_dummies(features[cat_col], drop_first=True)
features = pd.concat([features, dummies], axis=1)
features.drop(cat_col, axis=1, inplace=True)

# %%
features.head()

# %% [markdown]
## Datasets para entrenamiento, validacion y prueba

# %%
features_train, sub_features, target_train, sub_target = train_test_split(
    features, target, test_size=0.40, random_state=12345)

features_valid, features_test, target_valid, target_test = train_test_split(
    sub_features, sub_target, test_size=0.50, random_state=12345)

features_train.shape, features_valid.shape, features_test.shape

# %% [markdown]
# Entrenamiento de modelos
# ---
# A continuación, utilizaremos diferentes algoritmos de poderosas librerías como Scikit-learn, XGBoost, LightGBM, CatBoost, y usaremos diferentes técnicas para optimizarlos con el fin de encontrar el mejor modelo y alcanzar nuestra meta.

# %%
def metrics(model, features, target):
    predictions = model.predict(features)

    probabilities = model.predict_proba(features)
    probabilities_one = probabilities[:, 1]

    auc_roc = roc_auc_score(target, probabilities_one)

    f1 = f1_score(target, predictions)

    final_score = cross_val_score(model, features, target, cv=5).mean()

    confusion_matriz = confusion_matrix(target.tolist(), predictions.tolist())

    return auc_roc, f1, final_score, confusion_matriz

# %% [markdown]
## Modelo 1 Regresion logistica

# %%
%%time
model_1 = LogisticRegression(random_state=54321, solver='liblinear').fit(features_train, target_train)

auc_roc, f1, final_score, confusion_matriz = metrics(model=model_1, features=features_valid, target=target_valid)

print('Área Bajo la Curva ROC:', auc_roc)
print('F1 score:', f1)
print('Puntuación media de la evaluación del modelo:', final_score)
print('Matriz de confusion:')
print(confusion_matriz)

# %%
coef = model_1.coef_.tolist()

(pd.Series(coef[0], index=model_1.feature_names_in_)
 .sort_values(key=abs)
 .plot.barh()
)

# %%
rfe = RFE(model_1, n_features_to_select=5)

rfe.fit(features_train, target_train)

# %%
pd.DataFrame({'features':features_train.columns,
              'support': rfe.support_,
              'ranking':rfe.ranking_}).sort_values('ranking')

# %%
%%time
best_5_features = ['InternetService_Fiber optic', 'Type_Two year', 'Type_One year', 'OnlineSecurity', 'InternetService_No service']

model_1_1 = LogisticRegression(random_state=54321, solver='liblinear').fit(features_train[best_5_features], target_train)

auc_roc, f1, final_score, confusion_matriz = metrics(model=model_1_1, features=features_valid[best_5_features], target=target_valid)

print('Área Bajo la Curva ROC:', auc_roc)
print('F1 score:', f1)
print('Puntuación media de la evaluación del modelo:', final_score)
print('Matriz de confusion:')
print(confusion_matriz)

# %% [markdown]
# El primer modelo arroja resultados decentes, me parece interesante que estos no mejoran reduciendo la dimensionalidad. La regresion logistica es un buen algoritmo, pero existen otros aun mas potentes, considero que este modelo nos servira como base para evaluar modelos mas poderosos.

# %% [markdown]
## Modelo 2 Bosque aleatorio

# %%
%%time
model_2 = RandomForestClassifier(random_state=54321, class_weight='balanced', max_depth=9).fit(features_train, target_train)

auc_roc, f1, final_score, confusion_matriz = metrics(model=model_2, features=features_valid, target=target_valid)

print('Área Bajo la Curva ROC:', auc_roc)
print('F1 score:', f1)
print('Puntuación media de la evaluación del modelo:', final_score)
print('Matriz de confusion:')
print(confusion_matriz)

# %%
rfe = RFE(model_2, n_features_to_select=5)

rfe.fit(features_train, target_train)

# %%
pd.DataFrame({'features':features_train.columns,
              'support': rfe.support_,
              'ranking':rfe.ranking_}).sort_values('ranking')

# %%
%%time
best_5_features = ['MonthlyCharges', 'TotalCharges', 'Month', 'Year', 'Type_Two year']

model_2_1 = RandomForestClassifier(random_state=54321, class_weight='balanced', max_depth=9).fit(features_train[best_5_features], target_train)

auc_roc, f1, final_score, confusion_matriz = metrics(model=model_2_1, features=features_valid[best_5_features], target=target_valid)

print('Área Bajo la Curva ROC:', auc_roc)
print('F1 score:', f1)
print('Puntuación media de la evaluación del modelo:', final_score)
print('Matriz de confusion:')
print(confusion_matriz)

# %% [markdown]
# Podemos ver que el algoritmo de Bosque Aleatorio nos brinda mejores resultados, y estos incrementan incluso cuando se disminuye la dimensionalidad.

# %% [markdown]
## Modelo 3 LightGBM

# %%
%%time
model_3 = LGBMClassifier()

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [-1, 3, 5, 7],
    'num_leaves': [31, 50, 70],
}

# Perform grid search
grid_search = GridSearchCV(model_3, param_grid, cv=5, verbose=1, n_jobs=-1) #scoring='accuracy')
grid_search.fit(features_train, target_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# %%
%%time
model_3 = LGBMClassifier(learning_rate=0.2, max_depth=7, n_estimators=200, num_leaves=31).fit(features_train, target_train)

auc_roc, f1, final_score, confusion_matriz = metrics(model=model_3, features=features_valid, target=target_valid)

print('Área Bajo la Curva ROC:', auc_roc)
print('F1 score:', f1)
print('Puntuación media de la evaluación del modelo:', final_score)
print('Matriz de confusion:')
print(confusion_matriz)

# %%
importance = model_3.feature_importances_.tolist()

(pd.Series(importance, index=model_3.feature_name_)
 .sort_values(key=abs)
 .plot.barh()
)

# %%
rfe = RFE(model_3, n_features_to_select=5)

rfe.fit(features_train, target_train)

pd.DataFrame({'features':features_train.columns,
              'support': rfe.support_,
              'ranking':rfe.ranking_}).sort_values('ranking')

# %%
%%time
best_5_features = ['gender_Male', 'MonthlyCharges', 'Month', 'Year', 'TotalCharges']

model_3_1 = LGBMClassifier(learning_rate=0.2, max_depth=7, n_estimators=200, num_leaves=31).fit(features_train[best_5_features], target_train)

auc_roc, f1, final_score, confusion_matriz = metrics(model=model_3_1, features=features_valid[best_5_features], target=target_valid)

print('Área Bajo la Curva ROC:', auc_roc)
print('F1 score:', f1)
print('Puntuación media de la evaluación del modelo:', final_score)
print('Matriz de confusion:')
print(confusion_matriz)

# %% [markdown]
## Modelo 4 XGBoost

# %%
%%time
model_4 = xgb.XGBClassifier(eval_metric='logloss', random_state=12345)

param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [1, 3, 5, 7, 9]
}

# Perform grid search
grid_search = GridSearchCV(model_4, param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(features_train, target_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# %%
%%time
model_4 = xgb.XGBClassifier(eval_metric='logloss', random_state=12345, learning_rate= 0.2, max_depth=3, n_estimators=500)
model_4.fit(features_train, target_train)

auc_roc, f1, final_score, confusion_matriz = metrics(model=model_4, features=features_valid, target=target_valid)

print('Área Bajo la Curva ROC:', auc_roc)
print('F1 score:', f1)
print('Puntuación media de la evaluación del modelo:', final_score)
print('Matriz de confusion:')
print(confusion_matriz)

# %%
pd.Series(model_4.feature_importances_, index=features_valid.columns).sort_values().tail(25).plot.barh()

# %%
top_5_features = ['Type_Two year','PaymentMethod_Electronic check','InternetService_Fiber optic','Type_One year','Year']

model_4_1 = xgb.XGBClassifier(eval_metric='logloss', random_state=12345, learning_rate= 0.2, max_depth=3, n_estimators=500)
model_4_1.fit(features_train[top_5_features], target_train)

auc_roc, f1, final_score, confusion_matriz = metrics(model=model_4_1, features=features_valid[top_5_features], target=target_valid)

print('Área Bajo la Curva ROC:', auc_roc)
print('F1 score:', f1)
print('Puntuación media de la evaluación del modelo:', final_score)
print('Matriz de confusion:')
print(confusion_matriz)

# %% [markdown]
# Aunque la reduccion de dimensionalidad no mejoro las metricas, nuestro modelo original nos brindo resultados excelentes, hasta ahora los mejores.

# %% [markdown]
## Modelo 5 CatBoost    
# %%
# model_5 = CatBoostClassifier(random_state=12345, verbose=20, loss_function='Logloss', early_stopping_rounds=15)

# param_grid = {
#     'iterations': [100, 200, 300],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'depth': [4, 6, 8],
#     'l2_leaf_reg': [1, 3, 5, 7, 9],
#     'bagging_temperature': [0.1, 0.5, 1.0],
#     'rsm': [0.6, 0.8, 1.0]
# }

# # Perform grid search
# grid_search = GridSearchCV(model_5, param_grid, cv=5, verbose=1, n_jobs=-1).fit(features_train, target_train)

# # Best parameters and score
# print("Best Parameters:", grid_search.best_params_)
# print("Best Score:", grid_search.best_score_)

# %%
%%time
model_5 = CatBoostClassifier(random_state=12345, verbose=20, loss_function='Logloss', early_stopping_rounds=20, depth=6, iterations=500,
l2_leaf_reg=9, learning_rate=0.1, rsm=0.6).fit(features_train, target_train)

auc_roc, f1, final_score, confusion_matriz = metrics(model=model_5, features=features_valid, target=target_valid)

print('Área Bajo la Curva ROC:', auc_roc)
print('F1 score:', f1)
print('Puntuación media de la evaluación del modelo:', final_score)
print('Matriz de confusion:')
print(confusion_matriz)

# %% [markdown]
# Los resultados fueron maravillos, pero no superaron a nuestro algoritmo previo.

# %% [markdown]
## Analisis de resultados
# ---
# Después de haber entrenado diferentes modelos utilizando una variedad de algoritmos y técnicas para simplificarlos y evaluarlos obtuvimos las siguientes conclusiones:

# - Los mejores resultados arrojados por la regresión logística fueron 0.86 para la curva AUC-ROC, 0.62 en valor F, y 0.81 en accuracy usando validación cruzada. Estos resultados fueron decentes y nos sirvieron como base para evaluar otros algoritmos. 

# - Al momento de reducir la dimensionalidad en el algoritmo de bosque aleatorio, nuestros resultados mejoraron un poco, obtuvimos un 0.88 para la curva AUC-ROC, 0.68 en el valor F, y 0.83 en accuracy. Con este modelo se pudo apreciar una clara mejora en las predicciones. 

# - Nuestros resultados dieron un gran salto al utilizar la librería LightGBM, este modelo nos brindó un 0.92 en la curva AUC-ROC, 0.77 en el valor F, y 0.84 en accuracy. 

# - La poderosa librería de XGBoost nos arrojó los mejores resultados utilizando nuestro dataset de validación, 0.93 en la curva AUC-ROC, casi 0.78 en el valor F, y 0.85 en accuracy. Se puede apreciar la potencia que tienen los modelos basados en boosting. 

# - Por último, entrenamos un modelo utilizando la potente librería CatBoost. Los resultados también estuvieron a la altura, aunque no superaron a los ofrecidos por el último modelo entrenado, estos fueron 0.92 en la curva AUC-ROC, 0.76 en el valor F, y 0.85 en accuracy.

# %% [markdown]
## Seleccion del mejor modelo y validacion con dataset de pruebas

# %%
%%time
auc_roc, f1, final_score, confusion_matriz = metrics(model=model_4, features=features_test, target=target_test)

print('Área Bajo la Curva ROC:', auc_roc)
print('F1 score:', f1)
print('Puntuación media de la evaluación del modelo:', final_score)
print('Matriz de confusion:')
print(confusion_matriz)

# %% [markdown]
# Tenemos resultados excelentes con el conjunto de datos para pruebas, en mi opinion este modelo podria ser lanzado a produccion.
