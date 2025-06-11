import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import textwrap

import numpy as np

sns.set(style='dark')

column_bike_rent = ["casual_day", "registered_day", "cnt_day"]
column_bike_rent_hour = ["casual_hour", "registered_hour", "cnt_hour"]
rename_bike_rent = ["Casual", "Registered", "All Types"]
season_mapping = {1:"springer", 2:"summer", 3:"fall", 4:"winter"}
weather_mapping = {
    1: "Clear, Few clouds, Partly cloudy, Partly cloudy",
    2: "Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist",
    3: "Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds",
    4: "Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog"
}
day_mapping = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"}
day_type_mapping = {0: "weekend or holiday", 1: "working day"}

def create_day_based_df(df):
    day_based_df = df[['instant_day', 'dteday', 'season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit_day', 'temp_celsius_day', 'atemp_celsius_day', 'hum_day', 'windspeed_day', 'casual_day', 'registered_day', 'cnt_day']].copy()
    day_based_df.drop_duplicates(inplace=True)
    return day_based_df

def create_rent_based_weather_df(df):
    rent_based_weather_df = df.groupby(by="weathersit_hour").agg({
        "casual_hour": "sum",
        "registered_hour": "sum",
        "cnt_hour": "sum",
    })
    rent_based_weather_df = rent_based_weather_df.reset_index()
    rent_based_weather_df['weather_description'] = rent_based_weather_df['weathersit_hour'].map(weather_mapping)
    return rent_based_weather_df

def create_weather_cnt_month_year_df(df):
    weather_cnt_month_year_df = df.groupby(by=[
        df["dteday"].dt.year.rename("year"),
        df["dteday"].dt.month.rename("month"),
        "weathersit_hour"
    ])[["cnt_hour", "casual_hour", "registered_hour"]].sum().reset_index()
    weather_cnt_month_year_df["month_year"] = pd.to_datetime(
        weather_cnt_month_year_df["year"].astype(int).astype(str) + "-" + weather_cnt_month_year_df["month"].astype(int).astype(str),
        format="%Y-%m",
    )
    weather_cnt_month_year_df['weather_description'] = weather_cnt_month_year_df['weathersit_hour'].map(weather_mapping)
    weather_cnt_month_year_df = weather_cnt_month_year_df.drop(columns=['year', 'month'])
    return weather_cnt_month_year_df

def create_weather_cnt_season_year_df(df):
    weather_cnt_season_year_df = df.groupby(by=[
        df["dteday"].dt.year.rename("year"),
        "season",
        "weathersit_hour",
    ])[["cnt_hour", "casual_hour", "registered_hour"]].sum().reset_index()
    weather_cnt_season_year_df['weather_description'] = weather_cnt_season_year_df['weathersit_hour'].map(weather_mapping)
    weather_cnt_season_year_df['season_description'] = weather_cnt_season_year_df['season'].map(season_mapping)
    weather_cnt_season_year_df["season_year"] = (
        weather_cnt_season_year_df["year"].astype(int).astype(str) + "-" + weather_cnt_season_year_df["season_description"]
    )
    weather_cnt_season_year_df = weather_cnt_season_year_df.drop(columns=['year', 'season', 'season_description'])
    return weather_cnt_season_year_df

def create_rent_based_year_df(df):
    rent_based_year_df = df.groupby(by=df['dteday'].dt.year.rename('year')).agg({
        "casual_day": "sum",
        "registered_day": "sum",
        "cnt_day": "sum",
    })
    rent_based_year_df = rent_based_year_df.reset_index()
    return rent_based_year_df

def create_rent_based_season_year_df(df):
    rent_based_season_year_df = df.groupby([df['dteday'].dt.year.rename('year'), "season"]).agg({
        "casual_day": "sum",
        "registered_day": "sum",
        "cnt_day": "sum",
    })
    rent_based_season_year_df = rent_based_season_year_df.reset_index()
    rent_based_season_year_df['season_description'] = rent_based_season_year_df['season'].map(season_mapping)
    rent_based_season_year_df["season_year"] = (
        rent_based_season_year_df["year"].astype(int).astype(str) + "-" + rent_based_season_year_df["season_description"]
    )
    rent_based_season_year_df = rent_based_season_year_df.drop(columns=['year', 'season', 'season_description'])
    return rent_based_season_year_df

def create_rent_based_month_year_df(df):
    rent_based_month_year_df = df.groupby([df['dteday'].dt.year.rename('year'), day_based_df['dteday'].dt.month.rename('month')]).agg({
        "casual_day": "sum",
        "registered_day": "sum",
        "cnt_day": "sum",
    })
    rent_based_month_year_df = rent_based_month_year_df.reset_index()
    rent_based_month_year_df["month_year"] = pd.to_datetime(
        rent_based_month_year_df["year"].astype(int).astype(str) + "-" + rent_based_month_year_df["month"].astype(int).astype(str),
        format="%Y-%m",
    ).dt.strftime('%Y-%m')
    rent_based_month_year_df = rent_based_month_year_df.drop(columns=['year', 'month'])
    return rent_based_month_year_df

def create_rent_based_weekday_df(df):
    rent_based_weekday_df = df.groupby(by="weekday").agg({
        "casual_day": "sum",
        "registered_day": "sum",
        "cnt_day": "sum",
    })
    rent_based_weekday_df = rent_based_weekday_df.reset_index()
    rent_based_weekday_df['weekday_description'] = rent_based_weekday_df['weekday'].map(day_mapping)
    rent_based_weekday_df = rent_based_weekday_df.drop(columns=['weekday'])
    return rent_based_weekday_df

def create_rent_based_workingday_df(df):
    rent_based_workingday_df = df.groupby(by="workingday").agg({
        "casual_day": "mean",
        "registered_day": "mean",
        "cnt_day": "mean",
    })
    rent_based_workingday_df = rent_based_workingday_df.reset_index()
    rent_based_workingday_df['workingday_description'] = rent_based_workingday_df['workingday'].map(day_type_mapping)
    rent_based_workingday_df = rent_based_workingday_df.drop(columns=['workingday'])
    return rent_based_workingday_df

def create_rent_based_hour_df(df):
    rent_based_hour_df = df.groupby(by="hr").agg({
        "casual_hour": "sum",
        "registered_hour": "sum",
        "cnt_hour": "sum",
    })
    rent_based_hour_df = rent_based_hour_df.reset_index()
    rent_based_hour_df['hour'] = rent_based_hour_df['hr'].astype(str)
    rent_based_hour_df = rent_based_hour_df.drop(columns=['hr'])
    return rent_based_hour_df


main_data_df = pd.read_csv("dashboard/main_data.csv")

datetime_columns = ["dteday"]
for column in datetime_columns:
    main_data_df[column] = pd.to_datetime(main_data_df[column])

# komponen filter data
min_date = main_data_df["dteday"].min()
max_date = main_data_df["dteday"].max()

with st.sidebar:
    start_date, end_date = st.date_input(
        label='Rentang Waktu',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = main_data_df[(main_data_df["dteday"] >= str(start_date)) &
                (main_data_df["dteday"] <= str(end_date))]

# menyiapkan dataframe
day_based_df = create_day_based_df(main_df)
rent_based_weather_df = create_rent_based_weather_df(main_df)
weather_cnt_month_year_df = create_weather_cnt_month_year_df(main_df)
weather_cnt_season_year_df = create_weather_cnt_season_year_df(main_df)
rent_based_year_df = create_rent_based_year_df(day_based_df)
rent_based_season_year_df = create_rent_based_season_year_df(day_based_df)
rent_based_month_year_df = create_rent_based_month_year_df(day_based_df)
rent_based_weekday_df = create_rent_based_weekday_df(day_based_df)
rent_based_workingday_df = create_rent_based_workingday_df(day_based_df)
rent_based_hour_df = create_rent_based_hour_df(main_df)

# visualisasi data
st.header('Proyek Akhir Analisis Data: Bike Sharing Dataset')

# Pertanyaan 1: Bagaimana pengaruh cuaca terhadap jumlah peminjaman sepeda?
st.subheader("Jumlah peminjaman sepeda yang dikelompokan berdasarkan tipe cuaca")
bar_width = 0.25

x_labels = rent_based_weather_df["weather_description"].values
x_labels = [textwrap.fill(label, 25) for label in x_labels]

r1 = np.arange(len(rent_based_weather_df))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# **Fix: Assign figure to 'fig'**
fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    r1,
    rent_based_weather_df["casual_hour"],
    color="skyblue",
    width=bar_width,
    edgecolor="grey",
    label="Casual",
)
ax.bar(
    r2,
    rent_based_weather_df["registered_hour"],
    color="orange",
    width=bar_width,
    edgecolor="grey",
    label="Registered",
)
ax.bar(
    r3,
    rent_based_weather_df["cnt_hour"],
    color="green",
    width=bar_width,
    edgecolor="grey",
    label="All Types",
)

ax.ticklabel_format(style="plain")
ax.set_xlabel("Weather Situation")
ax.set_ylabel("Total Count")
ax.set_title("Bike Sharing by Weather Situation")
ax.set_xticks(r1 + bar_width)
ax.set_xticklabels(x_labels, ha="center", fontsize=8)
ax.legend()
fig.tight_layout()

st.pyplot(fig)

st.subheader('Jumlah peminjaman sepeda yang dikelompokan berdasarkan tipe cuaca untuk seluruh tipe peminjaman berdasarkan bulan dan tahun')
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 6))

sns.lineplot(x="month_year", y="cnt_hour", hue="weather_description", data=weather_cnt_month_year_df, ax=ax[0])
ax[0].set_ylabel("Total Count")
ax[0].set_xlabel("Year-Month")
ax[0].set_title("All Types Bike Sharing", loc="center", fontsize=15)
ax[0].tick_params(axis='y', labelsize=14)

sns.lineplot(x="month_year", y="casual_hour", hue="weather_description", data=weather_cnt_month_year_df, ax=ax[1])
ax[1].set_ylabel("Total Count")
ax[1].set_xlabel("Year-Month")
ax[1].set_title("Casual Type Bike Sharing", loc="center", fontsize=15)
ax[1].tick_params(axis='y', labelsize=14)

sns.lineplot(x="month_year", y="registered_hour", hue="weather_description", data=weather_cnt_month_year_df, ax=ax[2])
ax[2].set_ylabel("Total Count")
ax[2].set_xlabel("Year-Month")
ax[2].set_title("Registered Type Bike Sharing", loc="center", fontsize=15)
ax[2].tick_params(axis='y', labelsize=14)

plt.suptitle("Bike Sharing Trend by Weather Situation Based on Month", fontsize=20)
st.pyplot(fig)

st.subheader('Jumlah peminjaman sepeda yang dikelompokan berdasarkan tipe cuaca untuk seluruh tipe peminjaman berdasarkan musim dan tahun')
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 6))

sns.lineplot(x="season_year", y="cnt_hour", hue="weather_description", data=weather_cnt_season_year_df, ax=ax[0])
ax[0].set_ylabel("Total Count")
ax[0].set_xlabel("Year-Season")
ax[0].set_title("All Types Bike Sharing", loc="center", fontsize=15)
ax[0].tick_params(axis='y', labelsize=14)

sns.lineplot(x="season_year", y="casual_hour", hue="weather_description", data=weather_cnt_season_year_df, ax=ax[1])
ax[1].set_ylabel("Total Count")
ax[1].set_xlabel("Year-Season")
ax[1].set_title("Casual Type Bike Sharing", loc="center", fontsize=15)
ax[1].tick_params(axis='y', labelsize=14)

sns.lineplot(x="season_year", y="registered_hour", hue="weather_description", data=weather_cnt_season_year_df, ax=ax[2])
ax[2].set_ylabel("Total Count")
ax[2].set_xlabel("Year-Season")
ax[2].set_title("Registered Type Bike Sharing", loc="center", fontsize=15)
ax[2].tick_params(axis='y', labelsize=14)

plt.suptitle("Bike Sharing Trend by Weather Situation Based on Season", fontsize=20)
st.pyplot(fig)

# Pertanyaan 2: Kapan peminjaman paling banyak dan paling sedikit dilakukan? (Berdasarkan tahun, musim, bulan, hari, jam)
st.subheader("Berdasarkan tahun")
bar_width = 0.25

x_labels = rent_based_year_df["year"].values

r1 = np.arange(len(rent_based_year_df))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    r1,
    rent_based_year_df["casual_day"],
    color="skyblue",
    width=bar_width,
    edgecolor="grey",
    label=rename_bike_rent[0],
)
ax.bar(
    r2,
    rent_based_year_df["registered_day"],
    color="orange",
    width=bar_width,
    edgecolor="grey",
    label=rename_bike_rent[1],
)
ax.bar(
    r3,
    rent_based_year_df["cnt_day"],
    color="green",
    width=bar_width,
    edgecolor="grey",
    label=rename_bike_rent[2],
)

ax.ticklabel_format(style="plain")
ax.set_xlabel("Year")
ax.set_ylabel("Total Count")
ax.set_title("Bike Sharing Based on Year")
ax.set_xticks(r1 + bar_width)
ax.set_xticklabels(x_labels, ha="center", fontsize=8)
ax.legend()
fig.tight_layout()

st.pyplot(fig)

st.subheader('Berdasarkan musim pada seluruh tahun')
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(24, 18))

colors = ["#DEC4FC", "#C0F2EA", "#C0F2EA", "#C0F2EA", "#C0F2EA"]
ax = ax.flatten()

for i in range(0, len(column_bike_rent)):
    sns.barplot(x=column_bike_rent[i], y="season_year", data=rent_based_season_year_df.sort_values(by=column_bike_rent[i], ascending=False).head(5), palette=colors, ax=ax[i*2])
    ax[i*2].set_ylabel("Year-Season")
    ax[i*2].set_xlabel(rename_bike_rent[i])
    ax[i*2].set_title("Best Season of The Year Bike Sharing", loc="center", fontsize=15)
    ax[i*2].tick_params(axis='y', labelsize=14)

    sns.barplot(x=column_bike_rent[i], y="season_year", data=rent_based_season_year_df.sort_values(by=column_bike_rent[i], ascending=True).head(5), palette=colors, ax=ax[i*2+1])
    ax[i*2+1].set_ylabel("Year-Season")
    ax[i*2+1].set_xlabel(rename_bike_rent[i])
    ax[i*2+1].invert_xaxis()
    ax[i*2+1].yaxis.set_label_position("right")
    ax[i*2+1].yaxis.tick_right()
    ax[i*2+1].set_title("Worst Season of The Year Bike Sharing", loc="center", fontsize=15)
    ax[i*2+1].tick_params(axis='y', labelsize=14)

plt.suptitle("Best and Worst Performing Bike Sharing Based on Season of The Year", fontsize=20)
st.pyplot(fig)

st.subheader('Berdasarkan bulan pada seluruh tahun')
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(24, 18))

colors = ["#DEC4FC", "#C0F2EA", "#C0F2EA", "#C0F2EA", "#C0F2EA"]
ax = ax.flatten()

for i in range(0, len(column_bike_rent)):
    sns.barplot(x=column_bike_rent[i], y="month_year", data=rent_based_month_year_df.sort_values(by=column_bike_rent[i], ascending=False).head(5), palette=colors, ax=ax[i*2])
    ax[i*2].set_ylabel("Year-Month")
    ax[i*2].set_xlabel(rename_bike_rent[i])
    ax[i*2].set_title("Best Month of The Year Bike Sharing", loc="center", fontsize=15)
    ax[i*2].tick_params(axis='y', labelsize=14)

    sns.barplot(x=column_bike_rent[i], y="month_year", data=rent_based_month_year_df.sort_values(by=column_bike_rent[i], ascending=True).head(5), palette=colors, ax=ax[i*2+1])
    ax[i*2+1].set_ylabel("Year-Month")
    ax[i*2+1].set_xlabel(rename_bike_rent[i])
    ax[i*2+1].invert_xaxis()
    ax[i*2+1].yaxis.set_label_position("right")
    ax[i*2+1].yaxis.tick_right()
    ax[i*2+1].set_title("Worst Month of The Year Bike Sharing", loc="center", fontsize=15)
    ax[i*2+1].tick_params(axis='y', labelsize=14)

plt.suptitle("Best and Worst Performing Bike Sharing Based on Month of The Year", fontsize=20)
st.pyplot(fig)

st.subheader('Berdasarkan jenis hari')
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 6)) # Changed ncols to 3

colors = ["#DEC4FC", "#C0F2EA", "#C0F2EA", "#C0F2EA", "#C0F2EA", "#C0F2EA", "#C0F2EA"]

ax = ax.flatten()

for i in range(0, len(column_bike_rent)):
    sns.barplot(x=column_bike_rent[i], y="weekday_description", data=rent_based_weekday_df.sort_values(by=column_bike_rent[i], ascending=False), palette=colors, ax=ax[i])
    ax[i].set_ylabel("Days")
    ax[i].set_xlabel(rename_bike_rent[i])
    ax[i].set_title(rename_bike_rent[i] + " Best Day Bike Sharing", loc="center", fontsize=15)
    ax[i].tick_params(axis='y', labelsize=14)

plt.suptitle("Best to Worst Performing Bike Sharing Based on Days", fontsize=20)
st.pyplot(fig)

st.subheader("Berdasarkan hari kerja dan libur")
bar_width = 0.25

x_labels = rent_based_workingday_df["workingday_description"].values

r1 = np.arange(len(rent_based_workingday_df))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    r1,
    rent_based_workingday_df["casual_day"],
    color="skyblue",
    width=bar_width,
    edgecolor="grey",
    label=rename_bike_rent[0],
)
ax.bar(
    r2,
    rent_based_workingday_df["registered_day"],
    color="orange",
    width=bar_width,
    edgecolor="grey",
    label=rename_bike_rent[1],
)
ax.bar(
    r3,
    rent_based_workingday_df["cnt_day"],
    color="green",
    width=bar_width,
    edgecolor="grey",
    label=rename_bike_rent[2],
)

ax.ticklabel_format(style="plain")
ax.set_xlabel("Day Type")
ax.set_ylabel("Average Daily Rentals")
ax.set_title("Bike Sharing Based on Day Type")
ax.set_xticks(r1 + bar_width)
ax.set_xticklabels(x_labels, ha="center", fontsize=8)
ax.legend()
fig.tight_layout()

st.pyplot(fig)

st.subheader('Berdasarkan jam')
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(24, 18))

colors = ["#DEC4FC", "#C0F2EA", "#C0F2EA", "#C0F2EA", "#C0F2EA"]
ax = ax.flatten()

for i in range(0, len(column_bike_rent_hour)):
    sns.barplot(x=column_bike_rent_hour[i], y="hour", data=rent_based_hour_df.sort_values(by=column_bike_rent_hour[i], ascending=False).head(5), palette=colors, ax=ax[i*2])
    ax[i*2].set_ylabel("Hour")
    ax[i*2].set_xlabel(rename_bike_rent[i])
    ax[i*2].set_title("Best Hour Bike Sharing", loc="center", fontsize=15)
    ax[i*2].tick_params(axis='y', labelsize=14)

    sns.barplot(x=column_bike_rent_hour[i], y="hour", data=rent_based_hour_df.sort_values(by=column_bike_rent_hour[i], ascending=True).head(5), palette=colors, ax=ax[i*2+1])
    ax[i*2+1].set_ylabel("Hour")
    ax[i*2+1].set_xlabel(rename_bike_rent[i])
    ax[i*2+1].invert_xaxis()
    ax[i*2+1].yaxis.set_label_position("right")
    ax[i*2+1].yaxis.tick_right()
    ax[i*2+1].set_title("Worst Hour Bike Sharing", loc="center", fontsize=15)
    ax[i*2+1].tick_params(axis='y', labelsize=14)

plt.suptitle("Best and Worst Performing Bike Sharing Based on Hour", fontsize=20)
st.pyplot(fig)

st.caption('By: Khansa Mahira 2025')