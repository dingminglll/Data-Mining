import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv("Assignment1/dataset/dataset_mood_smartphone.csv",na_values='?')
dataset.dropna()
print(dataset.dtypes)

def get_user_ids(data):
    user_ids = []
    for user_id in data['id'].unique():
        user_ids.append(user_id)
    return user_ids

user_ids = get_user_ids(dataset)

# =============== Data visualization & analysis ==============================
# ========= Visualization =========
    
# Ensuring the 'time' column is a pandas datetime object
dataset['time'] = pd.to_datetime(dataset['time'])

# Checking whether the values are within the correct range    
activity_data = dataset[dataset['variable'] == 'activity']
mood_data = dataset[dataset['variable'] == 'mood']

# Function for creating a dictionary with individual user's data
def get_user_dfs(df):
    # Dictionary to hold DataFrames for each user
    user_dfs = {}
    
    # Iterate over each unique user
    for user_id in df['id'].unique():
        # Filter the data for the current user
        df_user = df[df['id'] == user_id]
        
        # Determine the first and last recorded dates for the current user
        user_start_date = df_user['time'].min()
        user_end_date = df_user['time'].max()

        # Filter the user DataFrame for the determined date range
        df_user_filtered = df_user[(df_user['time'] >= user_start_date) & (df_user['time'] <= user_end_date)]
        df_user_filtered.sort_values(by='time')
        
        # Store the user-specific DataFrame in the dictionary
        user_dfs[user_id] = df_user_filtered

    return user_dfs

user_dfs = get_user_dfs(dataset)

# Visualizing the mood changes for every user over time
def plot_all_mood_changes(datasets):
    plt.figure(figsize=(15, 5))
    # Generate a different color for each user
    colors = plt.cm.jet(np.linspace(0, 1, len(datasets)))
    
    for idx, (user, data) in enumerate(datasets.items()):
        mood_data = data[data['variable'] == 'mood']
        plt.plot(mood_data['time'], mood_data['value'], linestyle='-', label=user, color=colors[idx])

    # Enhancing the plot with titles and labels
    plt.title('Mood Changes Over Time')
    plt.xlabel('Time')
    plt.ylabel('Mood Value')
    plt.grid(True)  # Adding a grid for better readability
    plt.legend(title='User')  # Adding a legend with the title 'User'
    
    # Rotate and align the tick labels so they look better
    plt.gcf().autofmt_xdate()

    plt.show()

plot_all_mood_changes(user_dfs)

# Visualizing the mood changes over time for a selected user and specific
# timeperiod
def plot_user_mood_changes(user,user_id, start_date, end_date):
    
    user_filtered = user[(user['time'] >= start_date) & (user['time'] <= end_date)]
    plt.figure(figsize=(15, 5))
    mood_data = user_filtered[user_filtered['variable'] == 'mood']
    plt.scatter(mood_data['time'], mood_data['value'], linestyle='-', label=user)

    # Enhancing the plot with titles and labels
    plt.title('Mood Changes Over Time For User ' + user_id)
    plt.xlabel('Time')
    plt.ylabel('Mood Value')
    plt.grid(True)  # Adding a grid for better readability
    
    # Rotate and align the tick labels so they look better
    plt.gcf().autofmt_xdate()

    plt.show()

# Define your start and end dates for the period you're interested in
start_date = '2014-04-01'
start_date = pd.to_datetime(start_date)
end_date = '2014-04-02'
end_date = pd.to_datetime(end_date)

plot_user_mood_changes(user_dfs['AS14.15'],'AS14.15', start_date, end_date)

# ========= Analysis =========

# Reshaping the whole data to analyze the correlations between variables
def reshape_whole_dataset(data):
    df = pd.DataFrame(data)
    df_pivoted = df.pivot_table(index=['id','time'], columns='variable', values='value').reset_index()
    return df_pivoted

reshaped_data = reshape_whole_dataset(dataset)

def aggregate_data(df, mean_columns, sum_columns):
    # Convert the 'time' column to just dates without time components
    df['time'] = pd.to_datetime(df['time']).dt.date

    aggregation_dict = {col: 'mean' for col in mean_columns}
    # Add the sum calculations for the rest of the specified columns
    aggregation_dict.update({col: 'sum' for col in sum_columns})
    
    # Group by 'id' and 'time', then aggregate
    grouped_df = df.groupby(['id', 'time']).agg(aggregation_dict).reset_index()
    
    return grouped_df

# Specify which columns to average and which to sum
mean_columns = ['mood', 'activity', 'circumplex.valence', 'circumplex.arousal']
sum_columns = ['screen', 'call', 'sms', 'appCat.builtin', 'appCat.communication',
               'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office',
               'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown',
               'appCat.utilities', 'appCat.weather']

aggregated_data = aggregate_data(reshaped_data, mean_columns, sum_columns)

def get_daily_user_dfs(df):
    user_dfs = {}
    
    # Iterate over each unique user
    for user_id in df['id'].unique():
        # Filter the data for the current user
        df_user = df[df['id'] == user_id]
        
        # Determine the first and last recorded dates for the current user
        user_start_date = df_user['time'].min()
        user_end_date = df_user['time'].max()

        # Filter the user DataFrame for the determined date range
        df_user_filtered = df_user[(df_user['time'] >= user_start_date) & (df_user['time'] <= user_end_date)]
        df_user_filtered.sort_values(by='time')
        
        # Store the user-specific DataFrame in the dictionary
        user_dfs[user_id] = df_user_filtered

    return user_dfs

daily_user_dfs = get_daily_user_dfs(aggregated_data)

# heatmap for correlation matrix
def plot_heatmap(data, continuous_variables):
    correlation_matrix = data[continuous_variables].corr()
    plt.figure(figsize=(15, 15))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

#Heatmap for the whole dataset
plot_heatmap(aggregated_data, aggregated_data.columns)
#Heatmap for individual
plot_heatmap(daily_user_dfs['AS14.03'], daily_user_dfs['AS14.03'].columns)


# ============= IMPUTING MISSING VALUES ========================

# 1. Imputing the missing values using Interpolation

def interpolation(dfs):
    interpolated_dfs = {}
    for user_id, user_df in dfs.items():
        user_interpolated = user_df.interpolate(method='linear')
        # After interpolating applying forward fill and backward fill
        user_interpolated = user_interpolated.ffill().bfill()
    
        # Store the user-specific DataFrame in the dictionary
        interpolated_dfs[user_id] = user_interpolated
        
        # Concatenate all DataFrames into one, using the same index if desired
        merged_df = pd.concat(interpolated_dfs.values(), ignore_index=True)
        
    return merged_df,interpolated_dfs

interpolated_whole_dataset, interpolated_users_data = interpolation(daily_user_dfs)
correlation_matrix_interpolation = interpolated_users_data['AS14.12'].corr()

# 2. Imputing the missing values using K-nn 

def knn(data):
    dfs = {}
    
    for user_id, user_df in data.items():
        user_df = user_df.set_index('time')
        ids = user_df.pop('id') 
        
        # Initialize the KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        # Apply the imputer to the dataset
        # This will fit the imputer and transform the data in one step
        imputed_data = imputer.fit_transform(user_df)
        # Convert the imputed data back to a DataFrame
        # and include the time index from the original DataFrame
        imputed_data = pd.DataFrame(imputed_data, columns=user_df.columns, index=user_df.index)
        imputed_data = imputed_data.reset_index()
        imputed_data.insert(0, 'id', ids)
        
        dfs[user_id] = imputed_data
    
    return dfs

knn_users_data = knn(daily_user_dfs)
correlation_matrix_knn = knn_users_data['AS14.12'].corr()

# Visually comparing Interpolation to KNN 
def plot_correlation_matrices(corr1, corr2, title1='Interpolation', title2='K-NN'):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(corr1, annot=True, cmap='coolwarm')
    plt.title(title1)

    plt.subplot(1, 2, 2)
    sns.heatmap(corr2, annot=True, cmap='coolwarm')
    plt.title(title2)
    plt.show()

plot_correlation_matrices(correlation_matrix_interpolation, correlation_matrix_knn)

# Statistical comparison of the two methods
def model_performance(data):
    columns_to_drop = ['mood', 'id', 'time']
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns_to_drop, axis=1),
                                                        data['mood'], 
                                                        test_size=0.2, 
                                                        random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return mean_squared_error(y_test, predictions)

mse_interpolation = model_performance(interpolated_users_data['AS14.13'])
mse_knn = model_performance(knn_users_data['AS14.13'])

print("MSE Interpolation:", mse_interpolation)
print("MSE K-NN:", mse_knn)

# Comparing the methods based on the average of all the users

def all_users_MSE():
    interpolation_mses = 0
    knn_mses = 0
    for id in user_ids:
        interpolation_mses += model_performance(interpolated_users_data[id])
        knn_mses += model_performance(knn_users_data[id])
    interpolation_avg = interpolation_mses / 27
    knn_avg = knn_mses / 27
    
    print("MSE Interpolation average for the whole dataset:", interpolation_avg)
    print("MSE K-NN average for the whole dataset:", knn_avg)

all_users_MSE()
'''
Interpolation has a slightly better MSE score when considering the whole
dataset. Hence, I am choosing this method. 
'''

# ============= FEATURE ENGINEERING ========================
# ============= Part 1 ========================

# Combining feautes that make sense from an intuitive sense to categorize together
def combine_features(df):
    df['socialApps'] = df['appCat.communication'] + df['appCat.social']
    df['basicApps'] = df[['appCat.builtin', 'appCat.other', 'appCat.finance', 'appCat.travel', 
                                        'appCat.unknown', 'appCat.utilities', 'appCat.weather']].sum(axis=1)
    df['leisureApps'] = df['appCat.entertainment'] + df['appCat.game']
    df['workApps'] = df['appCat.office']
        
    drop_cols = ['appCat.communication', 'appCat.social', 'appCat.builtin', 'appCat.other', 
                     'appCat.finance', 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 
                     'appCat.weather', 'appCat.entertainment', 'appCat.game', 'appCat.office']
    df.drop(columns=drop_cols, inplace=True)
    return df

combined_features = combine_features(interpolated_whole_dataset)
plot_heatmap(combined_features, combined_features.columns)

def combine_features_per_user(dfs):
    for user_id, user_df in dfs.items():
        user_df['socialApps'] = user_df['appCat.communication'] + user_df['appCat.social']
        user_df['basicApps'] = user_df[['appCat.builtin', 'appCat.other', 'appCat.finance', 'appCat.travel', 
                                        'appCat.unknown', 'appCat.utilities', 'appCat.weather']].sum(axis=1)
        user_df['leisureApps'] = user_df['appCat.entertainment'] + user_df['appCat.game']
        user_df['workApps'] = user_df['appCat.office']
        
        # Drop the original columns since we have created combined features
        drop_cols = ['appCat.communication', 'appCat.social', 'appCat.builtin', 'appCat.other', 
                     'appCat.finance', 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 
                     'appCat.weather', 'appCat.entertainment', 'appCat.game', 'appCat.office']
        user_df.drop(columns=drop_cols, inplace=True)
    return dfs

combined_features_per_user = combine_features_per_user(interpolated_users_data)

# Looking at the correlation with the new features for a single user
correlation_matrix = combined_features_per_user['AS14.12'].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

# ============= Part 2 ========================

# Predicting values based on the mood from the past 5 days and other attributes

'''
Looking at the distribution of the mood values to see where
would be the best place to seperate the values into classes
to form it as a classification problem.
'''

# We need to filter the data for the first 5 days and day 6 separately
# Here you will need to adjust 'time_column' and 'user_id_column' to your actual column names

def prepare_data_for_numerical_prediction(dfs):
    user_dfs = {}
    for user_id, user_df in dfs.items():
        # Generate features for the first five days
        for i in range(1, 6):
            user_df[f'day_{i}'] = user_df['mood'].shift(-i+1)
        
        # Define the target for the 6th day
        user_df['target_mood'] = user_df['mood'].shift(-5)
        
        # Shift the 'date' column to get the date for the target mood.
        user_df['target_mood_date'] = user_df['time'].shift(-5)
        
        # Drop rows without full data for the 6-day window
        user_df.dropna(subset=[f'day_{i}' for i in range(1, 6)] + ['target_mood'], inplace=True)
        
        # For better readability
        user_df = user_df.drop(columns=['mood'])
        user_df.rename(columns={'time': 'initial_mood_date'}, inplace=True)
            
        # Store in a dictionary
        user_dfs[user_id] = user_df

    return user_dfs

prepared_data = prepare_data_for_numerical_prediction(combined_features_per_user)

# ============= NUMERICAL PREDICTION ==========================================

def train_test_single_user(df):
    # Linear Regression
    lin_model = LinearRegression()
    
    # RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    train_samples = int(len(df) * 0.8)  # 80% for training
    
    #Preparing training and testing data
    columns_to_drop = ['target_mood', 'id', 'initial_mood_date','target_mood_date']
    X_train = df.iloc[:train_samples][df.columns.drop(columns_to_drop)]
    y_train = df.iloc[:train_samples]['target_mood']
    X_test = df.iloc[train_samples:][df.columns.drop(columns_to_drop)]
    y_test = df.iloc[train_samples:]['target_mood']
    
    # For plotting
    X_label = df.iloc[train_samples:]['target_mood_date']
    
    #Fitting the models
    model.fit(X_train, y_train)
    lin_model.fit(X_train, y_train)
    pred_rf = model.predict(X_test)
    pred_lr = lin_model.predict(X_test)
    
    # Set the overall figure size
    plt.figure(figsize=(24,8))  # Width, Height for the entire figure containing both plots
    
    # First subplot for Random Forest
    plt.subplot(1, 2, 1)  # (nrows, ncols, index)
    plt.plot(X_label, pred_rf, label="Random Forest Predictions")
    plt.plot(X_label, y_test, label="Actual Mood")
    plt.legend(loc="upper left")
    plt.title("Random Forest Predictions vs Actual Mood")
    
    # Second subplot for Linear Regression
    plt.subplot(1, 2, 2)  # (nrows, ncols, index)
    plt.plot(X_label, pred_lr, label="Linear Regression Predictions")
    plt.plot(X_label, y_test, label="Actual Mood")
    plt.legend(loc="upper left")
    plt.title("Linear Regression Predictions vs Actual Mood")

    # Show the complete figure with both subplots
    plt.show()
    
# Function to assess which model performed better

train_test_single_user(prepared_data['AS14.12'])

def train_test_user_models(user_dfs):
    total_mse_rf = 0
    total_mse_lr = 0
    length = 0 # For calculating the mean mse later
    feature_importance_sum = None  # To accumulate feature importances
    
    for user, df in user_dfs.items():
        if len(df) > 5:  
            # Linear Regression
            lin_model = LinearRegression()
            
            # RandomForest model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            train_samples = int(len(df) * 0.8)  # 80% for training
            
            #Preparing training and testing data
            columns_to_drop = ['target_mood', 'id', 
                               'initial_mood_date', 'target_mood_date',
                               'workApps','screen','sms','call']
            X_train = df.iloc[:train_samples][df.columns.drop(columns_to_drop)]
            y_train = df.iloc[:train_samples]['target_mood']
            X_test = df.iloc[train_samples:][df.columns.drop(columns_to_drop)]
            y_test = df.iloc[train_samples:]['target_mood']
            
            #Fitting the models
            model.fit(X_train, y_train)
            lin_model.fit(X_train, y_train)
            pred_rf = model.predict(X_test)
            pred_lr = lin_model.predict(X_test)

            total_mse_rf += mean_squared_error(y_test, pred_rf)
            total_mse_lr += mean_squared_error(y_test, pred_lr)
            
            if feature_importance_sum is None:
                feature_importance_sum = model.feature_importances_
            else:
                feature_importance_sum += model.feature_importances_
                
            length += 1
            
    # Calculating average feature importance
    avg_feature_importance = feature_importance_sum / length
               
    # Get feature names
    feature_names = X_train.columns.tolist()
               
    # Plotting feature importance
    fig, ax = plt.subplots()
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, avg_feature_importance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Average Feature Importance')
    ax.set_title('Feature Importance Across All Users')
    plt.show()

    return total_mse_rf, total_mse_lr, length

# After running the train_test_user_models function to get the results
mse_rf, mse_lr, x = train_test_user_models(prepared_data)

# Average MSE for each model across all users
avg_mse_rf = mse_rf / x
avg_mse_lr = mse_lr / x
print(f"Average MSE for RandomForest: {avg_mse_rf}")
print(f"Average MSE for LinearRegression: {avg_mse_lr}")

# Determine which model performed better on average
if avg_mse_rf < avg_mse_lr:
    print("RandomForest performed better on average.")
elif avg_mse_rf > avg_mse_lr:
    print("LinearRegression performed better on average.")
else:
    print("Both models performed equally on average.")

# ============= CLASSIFICATION ===============================================

def mood_distribution(df, user_id):
    user_df = df[user_id]
    # Set the style of seaborn for better aesthetics
    sns.set_style('whitegrid')

    # Create the KDE plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(user_df['mood'], shade=True)
    plt.title('Kernel Density Estimate of Mood Scores')
    plt.xlabel('Mood Score')
    plt.ylabel('Density')
    plt.show()

mood_distribution(combined_features_per_user, 'AS14.14')

def prepare_data_for_classification(dfs):
    processed_dfs = {}
    for user_id, user_df in dfs.items():
        user_df = user_df.copy()  # Make a copy to avoid modifying the original data

        # Checking the column exists to prevent errors
        if 'target_mood' not in user_df.columns:
            raise KeyError(f"DataFrame for user {user_id} does not contain the column 'target_mood'")
        
        # Applying categorical transformations
        conditions = [
            (user_df['target_mood'] >= 1) & (user_df['target_mood'] <= 2),
            (user_df['target_mood'] > 2) & (user_df['target_mood'] < 5),
            (user_df['target_mood'] > 5) & (user_df['target_mood'] <= 6),
            (user_df['target_mood'] > 6) & (user_df['target_mood'] <= 7.5),
            (user_df['target_mood'] > 7.5) & (user_df['target_mood'] <= 10)
        ]
        categories = ['very low','low','average','high', 'very high']
        user_df['target_mood'] = np.select(conditions, categories)
        
        # For better readability
        user_df = user_df.drop(columns=['mood'])
        user_df.rename(columns={'time': 'initial_mood_date'}, inplace=True)

        # Store the processed DataFrame
        processed_dfs[user_id] = user_df

    return processed_dfs

classification_data = prepare_data_for_classification(combined_features_per_user)

# ========= To be worked on - confusion matrix doesn't work! =================
def classification_single_user(df, user_id):
    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    
    # RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    train_samples = int(len(df) * 0.8)  # 80% for training
    
    #Preparing training and testing data
    columns_to_drop = ['target_mood', 'id', 'initial_mood_date','target_mood_date']
    X_train = df.iloc[:train_samples][df.columns.drop(columns_to_drop)]
    y_train = df.iloc[:train_samples]['target_mood']
    X_test = df.iloc[train_samples:][df.columns.drop(columns_to_drop)]
    y_test = df.iloc[train_samples:]['target_mood']
    
    #Fitting the models
    model.fit(X_train, y_train)
    log_model.fit(X_train, y_train)
    pred_rf = model.predict(X_test)
    pred_lr = log_model.predict(X_test)
    
    # Set the overall figure size
    plt.figure(figsize=(24,8))  # Width, Height for the entire figure containing both plots
            
    # First confusion matrix for Random Forest
    cm_rf = confusion_matrix(y_test, pred_rf, labels=model.classes_)
    plt.subplot(1, 2, 1)  # (nrows, ncols, index)
    sns.heatmap(cm_rf, annot=True, fmt='g', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {user_id} with Random Forest')
    
    # Second confusion matrix for Linear Regression
    cm_lr = confusion_matrix(y_test, pred_lr, labels=log_model.classes_)
    plt.subplot(1, 2, 2)  
    sns.heatmap(cm_lr, annot=True, fmt='g', xticklabels=log_model.classes_, yticklabels=log_model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {user_id} with Logistic Regression')

    # Show the complete figure with both subplots
    plt.show()
    
classification_single_user(classification_data['AS14.12'],'AS14.12')

def train_test_user_models_classification(user_dfs):
    total_accuracy_lr = 0
    total_accuracy_rf = 0
    length = 0 # For calculating the mean accuracy later
    feature_importance_sum = None  # To accumulate feature importances

    for user, df in user_dfs.items():
        if len(df) > 5:
            # Logistic Regression model
            log_model = LogisticRegression(max_iter=1000)
            
            # RandomForest model
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            train_samples = int(len(df) * 0.8)  # 80% for training
            
            # Preparing training and testing data
            columns_to_drop = ['target_mood', 'id', 
                               'initial_mood_date', 'target_mood_date',
                               'workApps','screen','sms','call']
            X_train = df.iloc[:train_samples][df.columns.drop(columns_to_drop)]
            y_train = df.iloc[:train_samples]['target_mood']
            X_test = df.iloc[train_samples:][df.columns.drop(columns_to_drop)]
            y_test = df.iloc[train_samples:]['target_mood']
            
            # Fitting the models
            rf_model.fit(X_train, y_train)
            log_model.fit(X_train, y_train)
            pred_rf = rf_model.predict(X_test)
            pred_lr = log_model.predict(X_test)

            # Calculate accuracy
            accuracy_rf = accuracy_score(y_test, pred_rf)
            accuracy_lr = accuracy_score(y_test, pred_lr)
            total_accuracy_rf += accuracy_rf
            total_accuracy_lr += accuracy_lr
            
            if feature_importance_sum is None:
                feature_importance_sum = rf_model.feature_importances_
            else:
                feature_importance_sum += rf_model.feature_importances_
                
            length += 1
            
    # Calculating average feature importance
    avg_feature_importance = feature_importance_sum / length
               
    # Get feature names
    feature_names = X_train.columns.tolist()
               
    # Plotting feature importance
    fig, ax = plt.subplots()
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, avg_feature_importance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Average Feature Importance')
    ax.set_title('Feature Importance Across All Users')
    plt.show()

    return total_accuracy_rf / length, total_accuracy_lr / length

# Example usage:
accuracy_rf, accuracy_lr = train_test_user_models_classification(classification_data)
print(f"Average Accuracy for RandomForest: {accuracy_rf}")
print(f"Average Accuracy for LogisticRegression: {accuracy_lr}")

# Determine which model performed better on average
if accuracy_rf < accuracy_lr:
    print("RandomForest performed better on average.")
elif accuracy_rf > accuracy_lr:
    print("LogisticRegression performed better on average.")
else:
    print("Both models performed equally on average.")