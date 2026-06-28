import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from matplotlib.pyplot import bar

#-----------------------------СТАРТ--------------------------------------

df = pd.read_csv('data/covid_data.csv')

# print(df.head(5), '\n')
# print(df.info(),'\n')
# print('Colums covid_data:\n')
# for i, item in enumerate(df):
#     print(item, end="\t")
#
#     if (i + 1) % 10 == 0:
#         print()


numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_columns:
    df[col].fillna(df[col].mean())


# print('\n', df.duplicated().sum())#0


# print(df['total_cases_per_million'].mean())
# print(df['total_cases_per_million'].median())
# print(df['total_cases_per_million'].max())


# df_brazil = df[df['location'] == 'Brazil']
#
# x = df_brazil['date']
# y = df_brazil['total_deaths']
#
# plt.plot(x, y, color='black')
# plt.title("Загальна кількість випадків та смертей по датам в Бразилії")
# plt.xlabel("Дати")
# plt.ylabel("Загальна кількість")
# plt.show()
#
#
# df_south_america = df[df['continent'] == 'South America']
# df_europe = df[df['continent'] == 'Europe']
#
# latest_south_america = df_south_america['date'].max()
# latest_europe = df_europe['date'].max()
#
# south_america_date = df_south_america[df_south_america['date'] == latest_south_america]
# df_europe_date = df_europe[df_europe['date'] == latest_europe]
#
# bar(df_europe_date['date'], df_europe_date['total_deaths'].sum(), color='red')
# bar(south_america_date['date'], south_america_date['total_deaths'].sum(), color='blue')
#
# plt.title("Порівняння кількості смертей між континентами за останню дату від COVID-19 - Південної Америки та Європи")
# plt.xlabel("Останні дати континентів")
# plt.ylabel("Загальна кількість смертей")
# plt.show()

#------------------------------------Аналіз даних--------------------------------

print('Категорії до обробки:')
print(df[['new_cases', 'new_deaths', 'total_cases']].isnull().sum())
df[['new_cases', 'new_deaths', 'total_cases']] = df[['new_cases', 'new_deaths', 'total_cases']].fillna(df[['new_cases', 'new_deaths', 'total_cases']].mean())
print('\n Після обробки:')
print(df[['new_cases', 'new_deaths', 'total_cases']].isnull().sum())#0, 0, 0

df = df.drop_duplicates()

encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df[['location']])
print('Перетворений категоріальний стовпець location')
print(encoded_data)

df['date'] = pd.to_datetime(df['date'])

df = df.sort_values(['location', 'date'])

df['growth_rate_new_cases'] = (
    df.groupby('location')['new_cases'].pct_change()
)

df['growth_rate_new_deaths'] = (
    df.groupby('location')['new_deaths'].pct_change()
)

df['growth_rate_new_cases'] = df['growth_rate_new_cases'].fillna(0)
df['growth_rate_new_deaths'] = df['growth_rate_new_deaths'].fillna(0)

df.to_csv('data/cleaned_covid_data.csv', index=False)#Очищені дані збережено в cleaned_covid_data.csv