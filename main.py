import pandas as pd

df = pd.read_csv('data/covid_data.csv')

print(df.head(5), '\n')
print(df.info(),'\n')
print('Colums covid_data:\n')
for i, item in enumerate(df):
    print(item, end="\t")

    if (i + 1) % 10 == 0:
        print()

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_columns:
    df[col].fillna(df[col].mean())

print('\n', df.duplicated().sum())#0