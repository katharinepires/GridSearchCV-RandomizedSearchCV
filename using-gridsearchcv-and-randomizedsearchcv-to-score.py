#Using the XGB Regressor like a example:
from sklearn.model_selection import GridSearchCV
xgb = xgboost.XGBRegressor()
#Here you can put all the parameters and values you want
par = {'nthread':[2, 3, 4, 5],
       'learning_rate': [0.01, 0.05, 0.07, 0.1], 
       'max_depth': [3, 4, 5, 6],
       'min_child_weight': [2, 3, 4, 5],
       'subsample': [0.5, 0.6, 0.7, 1],
       'colsample_bytree': [0.5, 0.6, 0.7, 1],
       'n_estimators': [200, 250, 300, 350, 400]}

xgb_grid = GridSearchCV(xgb,par,cv = 3,n_jobs = 3,verbose = True)
xgb_grid.fit(X_train, Y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

#Using the Gradient Boosting Regressor like a example:
from sklearn.model_selection import RandomizedSearchCV
gb = GradientBoostingRegressor()
#Here you can put all the parameters and values you want
par = {"learning_rate": np.linspace(0.05, 0.15,5),
       "max_depth": range(3, 5),
       "min_samples_leaf": range(3, 5)}

rand = RandomizedSearchCV(gb, par, cv= 3, n_iter = 10, random_state = 42)
rand.fit(X_train, Y_train)

print(rand.best_score_)
print(rand.best_params_)