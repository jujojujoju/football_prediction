# Europe Soccer Match Outcome Prediction
This project contains a process for predicting the outcome of a European football game using the odds and the players' data.

## Database
### database.sqlite
This sqlite has a relational database of players, teams, matches, etc. in European football games.

## Data processiong
### data_odd.py
This Python file creates odds data from database.sqlite and generates 4000 match_train_odd.csv and 1000 match_test_odd.csv.

### data_odd_player.py
This Python file creates odds and players data from database.sqlite and generates 4000 match_train_odd_player.csv and 1000 match_test_odd_player.csv.

## Training and Test
### singleLayer_softmax.py
This python file performs single layer classification using the generated odds data or odds and player data and outputs the result.

### multiLayer_ReLu_dropout.py
This python file performs multi layer perceptron using the generated odds data or odds and player data and outputs the result.

### league_rank_prediction_nn.py
This python file performs single layer classification using the generated odds data and predicts the final rank of Spanish laliga in the 16/17 season.

