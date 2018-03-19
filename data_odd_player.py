import sqlite3
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import warnings

pd.set_option('display.width', 320)
warnings.simplefilter("ignore")

conn = sqlite3.connect("../database.sqlite")
match_data = pd.read_sql("SELECT * FROM Match;", conn)
player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)


# 플레이어들의 아이디를 가지고 players 테이블로부터 오버롤을 불러와서 저장한다
def get_fifa_stats(match, player_stats):
    match_id = match.match_api_id
    date = match['date']
    players = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11", "away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]
    player_stats_new = pd.DataFrame()
    names = []

    # Loop through all players
    for player in players:
        # Get player ID
        player_id = match[player]
        # Get player stats
        stats = player_stats[player_stats.player_api_id == player_id]
        # Identify current stats
        current_stats = stats[stats.date < date].sort_values(by='date', ascending=False)[:1]
        if np.isnan(player_id) == True:
            overall_rating = pd.Series(0)
        else:
            current_stats.reset_index(inplace=True, drop=True)
            overall_rating = pd.Series(current_stats.loc[0, "overall_rating"])
        # Rename stat
        name = "{}_overall_rating".format(player)
        names.append(name)
        # Aggregate stats
        player_stats_new = pd.concat([player_stats_new, overall_rating], axis=1)

    player_stats_new.columns = names
    player_stats_new['match_api_id'] = match_id
    player_stats_new.reset_index(inplace=True, drop=True)
    # Return player stats
    return player_stats_new.ix[0]


# 모든 매치데이터의 각 행의 선수들의 피파데이터를 합친다
def get_fifa_data(matches, player_stats, path=None, data_exists=False):
    if data_exists == True:
        fifa_data = pd.read_pickle(path)
    else:
        print("Collecting fifa data for each match...")
        # Apply get_fifa_stats for each match
        fifa_data = matches.apply(lambda x: get_fifa_stats(x, player_stats), axis=1)
    # Return fifa_data
    return fifa_data


columns = ["match_api_id", "date",
           "home_team_goal", "away_team_goal",
           "home_player_1", "home_player_2", "home_player_3", "home_player_4", "home_player_5",
           "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
           "home_player_11", "away_player_1",
           "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
           "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]

odd_columns = ["B365H", "B365D", "B365A", "WHH", "WHD", "WHA",
               "VCH", "VCD", "VCA", "BWH", "BWD", "BWA", "IWH", "IWD", "IWA",
               "PSH", "PSD", "PSA"]

columns = columns + odd_columns

total_record_num = 5000
train_record_num = int(total_record_num * 0.8)
test_record_num = total_record_num - train_record_num

# subset중 하나라도 missing value 이면 제거한다
match_data.dropna(subset=columns, inplace=True)
match_data = match_data[columns]

fifa_data = get_fifa_data(match_data, player_stats_data, data_exists=False)
matchWithPlayer = pd.merge(match_data, fifa_data, on='match_api_id')

# class는 승,무,패 3가지이고 각 y값을 득실차로부터 생성한다
matchWithPlayer['home_win'] = matchWithPlayer.apply(lambda x: 1
if (x['home_team_goal'] - x['away_team_goal'] > 0)
else(2 if (x['home_team_goal'] == x['away_team_goal']) else 0), axis=1)

matchWithPlayer = matchWithPlayer.drop(
    ['match_api_id', 'home_team_goal', 'away_team_goal', 'date',
     "home_player_1", "home_player_2", "home_player_3"
        , "home_player_4", "home_player_5", "home_player_6", "home_player_7",
     "home_player_8"
        , "home_player_9", "home_player_10", "home_player_11", "away_player_1",
     "away_player_2"
        , "away_player_3", "away_player_4", "away_player_5", "away_player_6",
     "away_player_7"
        , "away_player_8", "away_player_9", "away_player_10", "away_player_11"], 1)

print(matchWithPlayer)

train_match_data = matchWithPlayer.head(train_record_num)
test_match_data = matchWithPlayer.tail(test_record_num)

np.savetxt("match_train_odd_player.csv",
           train_match_data,
           delimiter=";")
np.savetxt("match_test_odd_player.csv",
           test_match_data,
           delimiter=";")
