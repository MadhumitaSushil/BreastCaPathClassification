import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data from the CSV file
data = pd.read_csv('../output/results.csv')

# Set the seaborn style to whitegrid
sns.set_style('whitegrid')

# Get the unique task names
tasks = data['Task'].unique()

# Set the number of models you have
num_models = len(data['Model'].unique())

# Calculate the number of rows and columns for subplots
num_rows = 4  # You can adjust the number of rows and columns as needed
num_cols = 4

# Create a figure with subplots
fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 10))

# Loop through tasks and create subplots
for i, task in enumerate(tasks):
    # Filter data for the current task
    task_data = data[data['Task'] == task]

    # Sort data by Macro F1 score in descending order
    task_data = task_data.sort_values(by='Macro F1', ascending=False)

    # Calculate the subplot coordinates
    row = i // num_cols
    col = i % num_cols

    # Create the horizontal bar plot for the current task
    sns.barplot(x='Macro F1', y='Model', data=task_data, ax=axs[row, col], palette='viridis')

    # Set the title for the subplot
    axs[row, col].set_title(f'{task}')
    axs[row, col].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axs[row, col].set_xlim([0, 1])
    axs[row, col].set_ylabel(None)

fig.delaxes(axs[3, 1])
fig.delaxes(axs[3, 2])
fig.delaxes(axs[3, 3])

# Adjust spacing between subplots
plt.tight_layout()

# Display or save the plot
plt.savefig('../output/results.png', dpi=600)
