import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_data(path='har70plus', patients_ids=(501,519), lagged =False, timesteps=5, exclude_ids=False):
    
    """
        
        The function allows to load the data
        
        Parameters:
        path (str) --> path where the data is stored
        patients_ids (str or tuple) --> range of ids to load
        lagged (bool) --> indicates whether to compute lagged features
        timesteps (int) --> number of timesteps for lagged features
        exclude_ids (bool) --> indicates whether to exclude samples with labels 3,4 and 5
        
    
    """
    
    df = pd.DataFrame()
    if type(patients_ids) is tuple:
        patients_ids = range(patients_ids[0], patients_ids[1])
    else:
        patients_ids = range(patients_ids)
        
    for i in patients_ids:

        # Read the CSV file for the current subject
        subject_df = pd.read_csv(f'{path}/{i}.csv')

        # Add a 'subject_id' column with the subject ID (i)
        subject_df['subject_id'] = i
        
        
        if lagged:
            # List of columns to lag
            columns_to_lag = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']

            # Number of timesteps to lag
            timesteps = 5

            # Lag the specified columns with the desired number of timesteps
            for col in columns_to_lag:
                for t in range(1, timesteps + 1):
                    subject_df[f'{col}_lag_{t}'] = subject_df[col].shift(t)
        if exclude_ids:
            subject_df = subject_df[~subject_df['label'].isin([3, 4, 5])]

        # Concatenate the current subject's DataFrame with the main DataFrame (df)
        df = pd.concat([df, subject_df], ignore_index=True)

    return df
   
    

def timeseries_visualization(df, subject_id, title, sensor_type='back' ):
    
    """
    
        This function allows to visualize the activity from an entire recording session of a specific patient
        
        Parameters:
        df (pd.DataFrame) --> dataframe of the entire dataset
        subject_id (int) --> id of patient
        sensor_type (str) --> type of sensor to visualize
        title (str) --> title for the plot
        
        Credit:
        - ChatGPT was asked to handle visualization of labels and the coloring of the background
        
        
    """
    
    df_subject = df[df.subject_id == subject_id]
    timesteps = df_subject.index - df_subject.index[0]
    # Plot the back sensor data
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(timesteps, df_subject[f'{sensor_type}_x'], label='x', color='purple')
    plt.plot(timesteps, df_subject[f'{sensor_type}_y'], label='y', color='cyan')
    plt.plot(timesteps, df_subject[f'{sensor_type}_z'], label='z', color='lawngreen')

    # Define colors for each label
    colors = {
        1: 'paleturquoise',
        3: 'palegreen',
        4: 'lightcoral',
        5: 'lavender',
        6: 'navajowhite',
        7: 'lightpink',
        8: 'plum'
    }

    # Find segments where the label changes
    label_changes = df_subject['label'] != df_subject['label'].shift(1)
    start_indices = df_subject.index[label_changes].tolist() 
    end_indices = start_indices[1:] + [df_subject.index[-1]]

    # Create a set to track encountered labels
    encountered_labels = set()

    # Fill the background based on label and create unique legend entries
    for label_value, color in colors.items():
        label_indices = df_subject['label'] == label_value
        for start, end in zip(start_indices, end_indices):
            if label_indices[start] and label_value not in encountered_labels:
                ax.axvspan(
                    start - df_subject.index[0],
                    end - df_subject.index[0],
                    alpha=0.6,  # Adjust transparency as needed
                    color=color,
                    label=f'Label {label_value}'
                )
                encountered_labels.add(label_value)
            elif label_indices[start]:
                 ax.axvspan(
                    start - df_subject.index[0],
                    end - df_subject.index[0],
                    alpha=0.6,  # Adjust transparency as needed
                    color=color
                )


    # Adjust the legend with unique entries
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Timesteps')
    plt.ylabel('Activity')
    plt.title(title)
    plt.grid()
    plt.show()
    
    
def visualize_hist_and_boxplot(df, subject_id):
    
    """
        This function plots histograms and boxplots of the attributes in the dataset.
        Helpful in detecting outliers
        
        Parameters:
        - df (pd.Dataframe) --> dataframe where the dataset is stored
        - subject_id (int) --> id of the subject
    
    """
    df_subject = df[df.subject_id == subject_id]
    dict_features = {0: 'back_x', 1: 'back_y', 2: 'back_z', 3: 'thigh_x', 4:'thigh_y', 5:'thigh_z'}
    
    # Create a figure with 4 rows and 6 of columns
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(30, 12))

    for i in range(6):

        # Computing indexes for rows and columns
        col_index = i % 6
        row_index = i // 6

        if i >= 6:
            # Add one to the row index to plot the attributes from 6 to 12 in row 3 and 4
            row_index += 1

        # Boxplots for the attributes
        axes[row_index, col_index].boxplot(df_subject[dict_features[i]])
        axes[row_index, col_index].set_title(f'Box Plot for Attribute {dict_features[i]}')
        axes[row_index, col_index].set_xlabel('Attribute')
        axes[row_index, col_index].set_ylabel('Value')

        # Histograms for the attributes
        axes[row_index+1, col_index].hist(df_subject[dict_features[i]], bins=20, color='lawngreen')
        axes[row_index+1, col_index].set_title(f'Histogram for Attribute {dict_features[i]}')
        axes[row_index+1, col_index].set_xlabel('Value')
        axes[row_index+1, col_index].set_ylabel('Frequency')

    # Title of the entire plot
    fig.suptitle(f'Box Plots and Histograms for Subject {subject_id}', fontsize = 20)

    # Show the plot
    plt.tight_layout()
    plt.show()

def visualize_pca(df, subject_id):

    """

        Visualizes the 2 principal components of the data

        Parameters:
        - df (pd.Dataframe) --> dataframe where the dataset is stored
        - subject_id (int) --> id of the subject

    """

    df_subject = df[df.subject_id == subject_id]
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df_subject.drop(['subject_id', 'timestamp', 'label'], axis=1))
    plt.figure(figsize=(8, 6))
    # Plot data points with original labels using distinct colors
    colors = {
            1: 'paleturquoise',
            3: 'palegreen',
            4: 'lightcoral',
            5: 'lavender',
            6: 'navajowhite',
            7: 'lightpink',
            8: 'plum'
        }

    labels = [1,3,4,5,6,7,8]
    for  label in labels:
        mask = df_subject['label'] == label
        plt.scatter(
            reduced_data[mask, 0],
            reduced_data[mask, 1],
            label=f'Label {label}',
            c=colors[label]  # Cycles through the class_colors list
        )

    plt.legend()
    plt.grid()
    plt.title("PCA applied on Accelerometer Data")
    plt.xlabel('First PC')
    plt.ylabel('Second PC')
    plt.show()

        
    
