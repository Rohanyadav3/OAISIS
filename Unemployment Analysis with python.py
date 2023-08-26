import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

# Load the dataset
url = "https://raw.githubusercontent.com/gokul-raj-km/unemployment_in_india/main/unemployment.csv"
data = pd.read_csv(url)

# Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Data Exploration
print("Data Overview:")
print(data.head())
print("\nData Info:")
print(data.info())

# Time Series Analysis
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x=data.index, y='Unemployment Rate', hue='Period')
plt.title('Unemployment Rate Trends')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate')
plt.xticks(rotation=45)
plt.legend(title='Period')
plt.tight_layout()
plt.show()

# Geographical Analysis
india_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
merged_data = india_map.set_index('name').join(data.groupby('State')['Unemployment Rate'].last())
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged_data.plot(column='Unemployment Rate', ax=ax, legend=True, cmap='coolwarm',
                 legend_kwds={'label': "Unemployment Rate", 'orientation': "horizontal"})
plt.title('Unemployment Rate by State')
plt.axis('off')
plt.show()

# Age Group Analysis
age_group_data = data.groupby('Age Group')['Unemployment Rate'].mean()
age_group_data.plot(kind='bar', figsize=(8, 6), color='skyblue')
plt.title('Average Unemployment Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Unemployment Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
