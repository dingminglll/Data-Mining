import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



file_path = 'dataset/dataset_mood_smartphone.csv'
save_path = 'dataset/dataset_mood_smartphone_remade.csv'

continuous_variables = [
    'circumplex.arousal', 
    'circumplex.valence', 
    'activity', 
    'screen',
    'appCat.builtin',
    'appCat.communication',
    'appCat.entertainment',
    'appCat.finance',
    'appCat.game',
    'appCat.office',
    'appCat.other',
    'appCat.social',
    'appCat.travel',
    'appCat.unknown',
    'appCat.utilities',
    'appCat.weather'
]


def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def time_to_datetime(data):
    data['time'] = pd.to_datetime(data['time']).dt.date
    return data

def reshape_data_byid(data):
    data = time_to_datetime(data)
    df = pd.DataFrame(data)
    # reshape the data
    df_pivoted = df.pivot_table(index=['id', 'time'], columns='variable', values='value').reset_index()
    return df_pivoted

def reshape_data(data):
    data = time_to_datetime(data)
    df = pd.DataFrame(data)
    # reshape the data
    df_pivoted = df.pivot_table(index='time', columns='variable', values='value').reset_index()
    return df_pivoted

def calculate_daily_mood_mean(data):
    df = pd.DataFrame(data)
    daily_mood_mean = df.groupby(['id','time','variable'])['value'].mean().reset_index()
    return daily_mood_mean

def save_to_csv(data, file_path):
    data.to_csv(file_path, index=False)


# Histogram for mood distribution
def plot_histogram(data, column, bins):
    sns.histplot(data=data, x=column, bins=bins)
    plt.title('Mood Distribution')
    plt.xlabel('Mood Score')
    plt.ylabel('Frequency')
    plt.show()

# Time Series Plot for mood over time
# You might need to ensure that the time column is in datetime format
def plot_time_series(data, x, y):
    plt.figure(figsize=(15,5))
    sns.lineplot(data=data, x=x, y=y)
    plt.title('Mood Over Time')
    plt.xlabel('Time')
    plt.ylabel('Mood Score')
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    plt.show()

# Bar chart for mood vs. call
def plot_bar_chart(data, x, y):
    sns.barplot(data=data, x=x, y=y)
    plt.title('Mood and Call Made')
    plt.xlabel('Call Made (0 = No, 1 = Yes)')
    plt.ylabel('Average Mood Score')
    plt.show()

# boxplot for classification variables
def plot_boxplot(data):
    categorical_variables = ['call', 'sms']
    for var in categorical_variables:
        sns.boxplot(x=data[var], y=data['mood'])
        plt.title(f'Mood by {var}')
        plt.xlabel(var)
        plt.ylabel('Mood')
        plt.show()

# scatter plot for continuous variables
# def plot_scatter(data):
#    for var in continuous_variables:
#     sns.scatterplot(x=data[var], y=data['mood'])
#     plt.title(f'Mood vs {var}')
#     plt.xlabel(var)
#     plt.ylabel('Mood')
#     plt.show()

def plot_scatter(data):
    for var in continuous_variables:
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('time')
        ax1.set_ylabel('mood', color=color)
        ax1.plot(data['time'], data['mood'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel(var, color=color)  
        ax2.scatter(data['time'], data[var], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  
        plt.show()

# heatmap for correlation matrix
def plot_heatmap(data):
    correlation_matrix = data[continuous_variables + ['mood']].corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.title('Correlation Matrix')
    plt.show()



if __name__ == '__main__':
    data = read_csv(file_path)
    # print(data.head())
    data = time_to_datetime(data)
    data = calculate_daily_mood_mean(data)
    data_without_id = reshape_data(data)
    data_with_id = reshape_data_byid(data)
    save_to_csv(data_with_id, save_path)
    # plot_histogram(data, 'mood', 10)
    # plot_time_series(data, 'time', 'mood')
    # plot_boxplot(data)
    plot_scatter(data_without_id)
    plot_heatmap(data_without_id)
