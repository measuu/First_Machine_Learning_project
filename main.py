import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from matplotlib.pyplot import bar
import plotly.express as px

#-----------------------------СТАРТ--------------------------------------
df = pd.read_csv('data/covid_data.csv')

# print(df.head(5), '\n')
# print(df.info(),'\n')
# print('Стовпці в covid_data:\n')
# for i, item in enumerate(df):
#     print(item, end="\t")
#
#     if (i + 1) % 10 == 0:
#         print()
#
#
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_columns:
    df[col].fillna(df[col].mean())
#
# print('\n','Кількість дублікатів:', df.duplicated().sum())#0
#
# print('Среднє значення кількості випадків на міліон')
# print(df['total_cases_per_million'].mean())
# print('Медіана кількості випадків на міліон')
# print(df['total_cases_per_million'].median())
# print('Максимальне значення кількості випадків на міліон')
# print(df['total_cases_per_million'].max())
#
#
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


cols = ['new_cases', 'new_deaths', 'total_cases', 'population', 'gdp_per_capita']
cor = df[cols].corr()
print(cor)

plt.figure(figsize=(7, 5))
sns.heatmap(cor, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Теплова карта')
plt.tight_layout()
plt.show()


#Гістограми
plt.figure(figsize=(10, 6))

sns.histplot(data=df, x='total_cases', bins=30, kde=True)

plt.title('Гістограми для всіх випадків')
plt.xlabel('Кількість випадків')
plt.ylabel('Частота')

plt.show()


plt.figure(figsize=(10, 6))

sns.histplot(data=df, x='total_deaths', bins=30, kde=True)

plt.title('Гістограми для всіх смертей')
plt.xlabel('Кількість смертей')
plt.ylabel('Частота')

plt.show()


#boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x='continent', y='total_deaths_per_million', data=df)
plt.xticks(rotation=45)
plt.title('Розподіл смертей про міліон по континентам')
plt.tight_layout()

#pairplot
sns.pairplot(df[['total_cases', 'total_deaths', 'total_vaccinations', 'population']].dropna())
plt.show()

#Аналіз трендів
countries = ['United States', 'Ukraine', 'Germany']
select = df[df['location'].isin(countries)]

sns.lineplot(x='date', y='new_cases', hue='location', data=select)
plt.title('Нові випадки в США, Україні та Німеччині')
plt.show()


plt.figure(figsize=(14,6))

sns.lineplot(x='date', y='new_deaths', hue='location', data=select)
plt.title('Нові смерті в США, Україні та Німеччині')
plt.show()


df_countries = df.copy()
df_countries['date'] = pd.to_datetime(df_countries['date'], errors='coerce')
latest_country = df_countries.sort_values('date').groupby('location').tail(1)

top = latest_country.nlargest(10, 'total_cases_per_million')

plt.figure(figsize=(12,6))
sns.barplot(data=top, x='location', y='total_cases_per_million', palette='viridis')
plt.title('Топ 10 країн за кількості всього випадків на міліон')
plt.xlabel('Країни')
plt.ylabel('Кількість випадків')

plt.show()

for country in countries:
    country_data = df[df['location'] == country]

    maxc = country_data.loc[country_data['new_cases'].idxmax()]
    minc = country_data.loc[country_data['new_cases'].idxmin()]

    maxd = country_data.loc[country_data['new_deaths'].idxmax()]
    mind = country_data.loc[country_data['new_deaths'].idxmin()]

    print(f'\n------{country}------')

    print(f"Найбільше нових випадків: {maxc['new_cases']} ({maxc['date'].date()})")
    print(f"Найменше нових випадків: {minc['new_cases']} ({minc['date'].date()})")

    print(f"Найбільше нових смертей: {maxd['new_deaths']} ({maxd['date'].date()})")
    print(f"Найменше нових смертей: {mind['new_deaths']} ({mind['date'].date()})")