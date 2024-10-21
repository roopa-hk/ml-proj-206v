import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
'''
# Load the data
df = pd.read_excel("DataNew.xlsx")
print("Initial DataFrame:")
print(df.head())  # Print the first 6 lines
print(df.shape)
df.replace('.', 0, inplace=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df.head().T
'''
df = pd.read_csv('Original ATAC.txt', sep='\t')
# df.drop(['peakName', 'K562_ATACseq', 'FGRstart', 'FGRend', 'FGR'], axis=1, inplace=True)
df.replace('.', 0, inplace=True)
df = df[df.FGRstrand.isin(['+', '-'])]
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#print(df.dtypes)

unique_regions = df['region'].unique()
print(unique_regions)


# Convert to numeric, forcing non-convertible values to NaN
df['K562_POLR2A_ENCSR000FAJ'] = pd.to_numeric(df['K562_POLR2A_ENCSR000FAJ'], errors='coerce')

filter_column_names = ['chr','peakStart','peakEnd','peakScore','FGRstrand','region','K562_DPF2_ENCSR219BXP', 'K562_E2F1_ENCSR720HUL', 'K562_H3K4me3_ENCSR000DWD','K562_POLR2A_ENCSR000FAJ']
# Using .loc to filter
# filtered_df = df.loc[df['FRG region']
# Rename columns to remove spaces
# df.columns = df.columns.str.replace(' ', '_')
selected_columns = df.filter(items=filter_column_names)

filtered_rows = selected_columns[
    (selected_columns['peakScore'] < 2000 ) &
    (selected_columns['peakScore'] > 10 ) ]

df = filtered_rows.copy()
unique_regions = df['region'].unique()
print(unique_regions)


#df['ratio'] = df['K562_DPF2_ENCSR219BXP'] / df['K562_E2F1_ENCSR720HUL']



# First boxplot (without log transformation)
plt.figure(figsize=(8, 6))
sns.boxplot(x='region', y='K562_POLR2A_ENCSR000FAJ', data=df, hue='region', palette=["None",], legend=False)
plt.title('Boxplot of Pol2 by Distinct Genome Region')
plt.show()

# Second boxplot (with log2 transformation)
plt.figure(figsize=(8, 6))
sns.boxplot(x='region', y=np.log2(df['K562_DPF2_ENCSR219BXP']), data=df, hue='region', palette=None, legend=False)
plt.title('Boxplot of log2(Pol2) by Distinct Genome Region')
plt.show()


# extra
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_excel('DataNew.xlsx')
# Display the first few rows
# df.drop(['peakStart', 'peakEnd'], axis=1, inplace=True)

df = pd.read_excel('DataNew.xlsx')
# df.drop(['peakName', 'K562_ATACseq', 'FGRstart', 'FGRend', 'FGR'], axis=1, inplace=True)

df.replace('.', 0, inplace=True)
df = df[df.FGRstrand.isin(['+', '-'])]
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df.dtypes)
# Convert to numeric, forcing non-convertible values to NaN
df['K562_POLR2A_ENCSR000FAJ'] = pd.to_numeric(df['K562_POLR2A_ENCSR000FAJ'], errors='coerce')

filter_column_names = ['chr','peakStart','peakEnd','peakScore','FGRstrand','region','K562_DPF2_ENCSR219BXP', 'K562_E2F1_ENCSR720HUL',  'K562_H3K4me3_ENCSR000DWD','K562_POLR2A_ENCSR000FAJ']
# Using .loc to filter
# filtered_df = df.loc[df['FRG region']
# Rename columns to remove spaces
# df.columns = df.columns.str.replace(' ', '_')
selected_columns = df.filter(items=filter_column_names)
filtered_rows = selected_columns[
    (selected_columns['peakScore'] < 2000 ) &
    (selected_columns['peakScore'] > 100 ) &
    (selected_columns['K562_DPF2_ENCSR219BXP'] != 0 ) |
    (selected_columns['K562_E2F1_ENCSR720HUL'] != 0 ) |
    (selected_columns['K562_H3K4me3_ENCSR000DWD'] != 0 ) |
    (selected_columns['K562_POLR2A_ENCSR000FAJ'] != 0 ) &
    (selected_columns['chr'] == 'chr1') &
    (selected_columns['region'] == 'promoter')]


selected_columns.head(10)

'''
data = pd.read_excel('DataNew.xlsx')
x = data.iloc[;, 0],values.rehape(-1,1)
y = data.iloc[;,1],values.reshape(-1,1)
linear_regression = linearRegression()
linear_regression.fit(x,y)
y_predict = linear_regression.predict(x)
plt.scatter(x,y)
plt.plot(x,y='predict', color='red')
plt.show()
'''
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# First boxplot (without log transformation)
selected_columns = df.filter(items=filter_column_names)
plt.figure(figsize=(10, 8))
sns.boxplot(x='region', y='K562_POLR2A_ENCSR000DLY', data=df, hue='region', palette=["blue", "red", "grey"], legend=False)
plt.title('Boxplot of Pol2 by Distinct Genomic Region')
plt.show()

# Second boxplot (with log2 transformation)
selected_columns = df.filter(items=filter_column_names)
plt.figure(figsize=(10, 8))
sns.boxplot(x='region', y=np.log2(df['K562_POLR2A_ENCSR000DLY']), data=df, hue='region', palette=["Red", "blue", "grey"], legend=False)
plt.title('Boxplot of log2(Pol2) by Distinct genome region')
plt.show()
,,,
