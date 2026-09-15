# steam-game-recommender
Full stack game recommendation system that uses the Steam database to recommend similar games to users through cosine similarity based on multiple games selected by the user

Features
Search for games from the Steam dataset
Select multiple games to be used as inputs for recommendations
Generate recommendations based on all selected games
See a similarity score for each recommendation
View the recommended game's price, rating and release year
See which user-selected game each recommendation is most similar to
See the most important shared tags, genres and categories with that game
Remove selected games and update the recommendations

## How it works

The application consists of a React frontend with a Flask backend. The frontend allows the user to search for and select games, while the backend processes the selected games to generate recommendations.

## Frontend

The React frontend handles the user interface and sends requests to the Flask backend when the user searches for games or updates their selected games.

### Backend

The Flask backend provides the API that connects the React frontend to the recommendation system. It receives game searches and selected games from the frontend, processes the requests and returns the relevant results.

## Data cleaning

The database is cleaned to remove incomplete and duplicate entries before it is used by the recommendation system.

Only the information needed by the recommender is loaded from the database and games without names are removed as well as games without prices, as they are unlikely to be available for people to purchase.

Games with duplicate names are removed with the more popular entry being kept, this unfortunately does have the drawback of potentially removing games which happen to share a name with a more popular game or causing issues when there are remakes or reboots of older games.

Metacritic URLs are simplified so that games linking to the same Metacritic page can be identified as duplicates, allowing the less popular entry to be removed.

A simplified version of each game name is also created by converting it to lowercase and removing all non-alphanumeric characters. This is done to make searching for games easier.

The tags, genres and categories are also cleaned using a custom function because they are stored in different formats within the dataset. Spaces and hyphens are replaced with underscores so that multi-word features can be treated as individual terms. 

Categories that are not useful for determining game similarity are also removed to prevent them from having too much influence on the recommendations.

The clean features are then converted into space separated strings so they can be processed by TF-IDF.

## TF-IDF

TF-IDF is used to convert the cleaned tags, genres and categories into vectors that can be compared between games. It gives greater importance to features that are less common across the dataset, whereas features that appear more commonly across many games have less influence. A separate vectorizer is used for tags, genres and categories so each type of feature can independently contribute to the recommendation process.

## Cosine similarity

Cosine similarity is used to compare the TF-IDF vectors of the selected games with every game in the dataset. It produces a similarity score between 0 and 1 based on how closely the features of the two games match, the closer to 1 the score is, the greater the similarity.

This is calculated separately for tags, genres and categories. To make sure the recommendations are based on the user's entire selection, when multiple games are selected, the similarity scores for each feature are averaged out across all of the selected games.


## Additional recommendation factors

In addition to tags, genres and categories, each game's popularity, rating and release year are also considered in the recommender. These are converted into numerical values so that they can be included in the combined recommendation score.

Popularity is calculated by using the number of recommendations a game has received. Log scaling is applied to this value to stop incredibly popular games dominating, before the values are normalised between 0 and 1.

Rating is also set to a value between 0 and 1, being calculated by finding what percentage of all reviews are positive.

Release dates are converted into release years so that the games can be compared based on how close their release years are.


## Final recommendation score

The different similarity scores and recommendation factors are combined into a single overall score for each game. Different weights are used for each factor to control how much influence they have on the final recommendation, with tags having the most influence at 35%, followed by genres at 15% and categories at 10%. Popularity, rating and release year each contribute 8% through their different factors, adding up to 40%.


## Recommendation processing

Before generating the game recommendations, the selected games are removed from the results so that they cannot be recommended again. Games with duplicate Metacritic IDs are also filtered out to prevent the same games from appearing multiple times, potentially under a slightly different name. 

Based on their combined recommendation score, the remaining games are sorted from highest to lowest with the 50 highest scoring games being returned.

Each recommendation is then individually compared with the games selected by the user to find which game it is most similar to. The tags, genres and categories shared between these games are also found and ranked using their TF-IDF scores, with the five most important shared features of each type being returned.

The recommendation information is then formatted and returned to the frontend, including the game name, similarity score, price, rating, release year, the most similar selected game and their shared features.


## Technologies

Python
Flask
Pandas
NumPy
scikit-learn
React
JavaScript
CSS


# Dataset

The recommender uses a Steam games dataset from Kaggle https://www.kaggle.com/datasets/hubertsidorowicz/steam-games-dataset-daily-updates which contains information on 136000+ steam games.

The dataset files are not included in the repository because of their large file size. They must be downloaded and placed in the data folder before running the application.


## Setup

Clone the repository and navigate to project folder
Install Python packages: pip install pandas numpy scikit-learn flask flask-cors
Place the Steam dataset CSV files in the data folder
Start the flask backend: Python game-recommender.py
In a separate terminal, navigate to frontend folder and do: npm install
To start the React server do: npm run dev
Open the address provided by Vite in your browser


## Usage

Enter a game into the search box and then select it from the search results dropdown.

Multiple games may be selected to influence the recommendations. The recommendations will update automatically when games are added or removed.

Selected games may be removed by left clicking the x next to their name in the selected games section of the screen.

The 50 recommended games displayed are sorted by similarity score in descending order, each recommendation displaying its similarity score, price, rating and release year alongside the user selected game it is most similar to.

The shared tags, genres and categories between the recommendation and its most similar selected game are also displayed.