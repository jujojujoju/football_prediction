import sqlite3
import csv
import pandas as pd
import numpy as np
import warnings

pd.set_option('display.width', 320)
warnings.simplefilter("ignore")

conn = sqlite3.connect("../database.sqlite")
match_data = pd.read_sql("SELECT * FROM Match;", conn)

# 득실차와 모든 배당을 속성값으로 유지한다
columns = ["home_team_goal", "away_team_goal", "B365H", "B365D", "B365A", "WHH", "WHD", "WHA",
           "VCH", "VCD", "VCA", "BWH", "BWD", "BWA", "IWH", "IWD", "IWA",
           "PSH", "PSD", "PSA"]

total_record_num = 5000
train_record_num = int(total_record_num * 0.8)
test_record_num = total_record_num - train_record_num

# subset중 하나라도 missing value 이면 제거한다
match_data.dropna(subset=columns, inplace=True)
match_data = match_data[columns]

# class는 승,무,패 3가지이고 각 y값을 득실차로부터 생성한다
match_data['home_win'] = match_data.apply(lambda x: 1
if (x['home_team_goal'] - x['away_team_goal'] > 0)
else(2 if (x['home_team_goal'] == x['away_team_goal']) else 0), axis=1)

match_data = match_data.drop(['home_team_goal', 'away_team_goal'], 1)

print(match_data)

train_match_data = match_data.head(train_record_num)
test_match_data = match_data.tail(test_record_num)

np.savetxt("match_train_odd.csv",
           train_match_data,
           delimiter=";")
np.savetxt("match_test_odd.csv",
           test_match_data,
           delimiter=";")
