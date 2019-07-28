###########################################
CS229 Final Project
Authors:
Liubov Nikolenko (liubov@stanford.edu)
Hoormazd Rezaei (hoormazd@stanford.edu)
Pouya Rezazadeh Kalehbasti (pouyar@stanford.edu)
###########################################

In order to run the code make sure you pre-instal all the dependecies such as
TextBlob and sklearn

+ INITIAL DATA PREPROCESSING:
1. Generate a fine with review sentiment: python sentiment_analysis.py
2. Clean the data: python data_cleanup.py
3. Normalize and split the data: data_preprocessing_reviews.py

+ GENERATE THE FEATURE SELECTION .NPY FILE:
1. For P-value feature selection: python feature_selection.py
2. For Lasso CV: python cv.py

+ TRAIN AND RUN THE MODELS:
python run_models.py
Note that by commenting/uncommenting certain lines of code you will be able to
run different configurations of the models.
1. To run the models with Lasso CV feature selection comment out line 240
(coeffs = np.load('../Data/selected_coefs_pvals.npy') and uncomment line 241
(coeffs = np.load('../Data/selected_coefs.npy').
2. To run the models with p-value feature selection uncomment line 240
(coeffs = np.load('../Data/selected_coefs_pvals.npy') and comment out line 241
(coeffs = np.load('../Data/selected_coefs.npy').
3. To run the baseline uncomment the lines 277, 278
    (
    print("--------------------Linear Regression--------------------")
    LinearModel(X_concat, y_concat, X_test, y_test)
    )
    and comment out everything below these lines. Also, comment out the lines 268,
    269 and 270 (
    X_train = X_train[list(col_set)]
    X_val = X_val[list(col_set)]
    X_test = X_test[list(col_set)]
    
    )

Warning: certain models take a while to train and run!
