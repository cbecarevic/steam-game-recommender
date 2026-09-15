import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


#searches for games matching the users input
@app.route("/games") 
def get_games():
    search = request.args.get("search", "").lower()

    #removes everything thats not a letter or number to create a simplified version of each game name 
    new_search = ""

    for character in search:
        if character.isalnum():
            new_search += character

    search = new_search

    if not search:
        return jsonify([])

    game_data = games[games["search_name"].str.contains(search, regex=False)][["name", "search_name"]] 
    game_data["starts_with"] = game_data["search_name"].str.startswith(search) #marks games where the search appears at the beginning of the name
    game_data = game_data.sort_values("starts_with", ascending=False, kind="stable").head(10) #shows games starting with the search first while maintaining popularity

    return jsonify(game_data["name"].tolist()) 


#recieves the users selected games and returns reccomendations
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    game_names = data["games"]
    recommendations = recommend_games(game_names)

    return jsonify(recommendations)


#loads the columns from the database needed for the recommender
games = pd.read_csv("data/steam_games.csv", usecols=[
    "name",
    "price",
    "recommendations",
    "metacritic_url",
    "tags",
    "genres",
    "categories",
    "positive",
    "negative",
    "release_date"
])


#data cleaning

#removes potential duplicates or games missing important details 
games = games.dropna(subset=["name", "price"]) 
games = games.sort_values("recommendations",ascending=False) #sorts by popularity
games = games.drop_duplicates(subset="name")

games["metacritic_id"] = games["metacritic_url"].str.split("?").str[0] #removes tracking tags from metacritic urls

#removes duplicate games that link to the same metacritic page
games_with_metacritic = games[games["metacritic_id"].notna()]
games_without_metacritic = games[games["metacritic_id"].isna()]
games_with_metacritic = games_with_metacritic.drop_duplicates(subset="metacritic_id")


games = pd.concat([games_with_metacritic,games_without_metacritic]) #gives games with metacritic urls higher priority in showing up in results when searching

#creates a simplified version of each game name 
games["search_name"] = games["name"].str.lower()
games["search_name"] = games["search_name"].str.replace(r"[^a-z0-9]", "", regex=True)

games = games.sort_values("recommendations", ascending=False).reset_index(drop=True) #restores the popularity order and makes sure the indexes stay continuous




#text processing


#categories that do not have enough influence on the game to be included and thus would have too much influence if included
unwanted_categories = {
    "#category_contrast_controls",
    "#category_playable_at_your_own_pace",
    "#category_playable_without_vision",
    "Adjustable_Text_Size",
    "Camera_Comfort",
    "Captions_available",
    "Chat_Speech_to_text",
    "Chat_Text_to_speech",
    "Color_Alternatives",
    "Commentary_available",
    "Custom_Volume_Controls",
    "DualSense_Controller_Support",
    "DualShock_Controller_Support",
    "Family_Sharing",
    "HDR_available",
    "In_App_Purchases",
    "Includes_Source_SDK",
    "Includes_level_editor",
    "Mods_(require_HL2)",
    "Narrated_Game_Menus",
    "Playable_without_Timed_Input",
    "Remote_Play_Together",
    "Remote_Play_on_Phone",
    "Remote_Play_on_TV",
    "Remote_Play_on_Tablet",
    "Save_Anytime",
    "Stats",
    "SteamVR_Collectibles",
    "Steam_Achievements",
    "Steam_Cloud",
    "Steam_Input_API_Support",
    "Steam_Leaderboards",
    "Steam_Timeline",
    "Steam_Trading_Cards",
    "Steam_Turn_Notifications",
    "Steam_Workshop",
    "Stereo_Sound",
    "Subtitle_Options",
    "Surround_Sound",
    "Touch_Only_Option",
    "Tracked_Controller_Support",
    "Valve_Anti_Cheat_enabled"
}


#cleans the main 3 features
#tags can have either [] or {} format
#genres and categories can either have list or descriptions format
def clean_text(text):

    if not text:
        return ""

    items = []

    if "description" in text:
        descriptions = text.split('"description": "')[1:]
        

        for description in descriptions:
            item = description.split('}')[0].strip('"').replace(" ", "_").replace("-", "_") 

            if item not in unwanted_categories:
                items.append(item)

    else:
        if text[0] == "[": 
            for item in text.strip("[]").split(","):
                item = item.strip().strip('" ').replace(" ","_").replace("-","_")

                if item not in unwanted_categories:
                    items.append(item)

        else:
            for item in text.strip("{}").split(","):
                item = item.split(":")[0].strip().strip('" ').replace(" ","_").replace("-","_")
                items.append(item)

    return " ".join(items) #converts cleaned features into space seperated string for TFIDF
    


#cleans text used by the recommender
games["tags"] = games["tags"].apply(clean_text)
games["genres"] = games["genres"].apply(clean_text)
games["categories"] = games["categories"].apply(clean_text) 

games["popularity"] = np.log1p(games["recommendations"].fillna(0)) #uses log scaling so very popular games do not dominate
games["popularity"] = games["popularity"] / games["popularity"].max()

games["rating"] = games["positive"] /(games["positive"] + games["negative"]) #gets the rating to a value between 0 and 1

#converts release dates to just the release year
games["release_date"] = pd.to_datetime(games["release_date"]) 
games["release_year"] = games["release_date"].dt.year




#tfidf vectors

#tfidf gives importance to features that a game has but are less common across the whole dataset

tag_vectorizer = TfidfVectorizer(lowercase=False)
tag_matrix = tag_vectorizer.fit_transform(games["tags"])

genre_vectorizer = TfidfVectorizer(lowercase=False)
genre_matrix = genre_vectorizer.fit_transform(games["genres"])

category_vectorizer = TfidfVectorizer(lowercase=False)
category_matrix = category_vectorizer.fit_transform(games["categories"])


#find the most important features shared between two games
def rank_shared_features(recommended_features, selected_features, matrix, vectorizer, index):

    shared_features = recommended_features & selected_features #finds features occuring in both games

    feature_names = vectorizer.get_feature_names_out().tolist() #gets the names of every feature used by tfidf
    feature_scores = matrix[index].toarray()[0] #gets tfidf scores for the recommended game

    shared_feature_scores = []

    #gets the tfidf score for each feature
    for feature in shared_features:
        if feature not in feature_names:
            continue

        score = feature_scores[feature_names.index(feature)] 
        shared_feature_scores.append((feature, score))

    shared_feature_scores.sort(key=lambda tup: tup[1], reverse=True) #sorts shared features by highest tfidf score

    shared_features = []
    for feature, score in shared_feature_scores[:5]:
            shared_features.append(feature)
    

    return shared_features



#generates recommendations based on the games selected by the user
def recommend_games(game_names):

    recommendations = []
    selected_game_indexes = []

    #finds the index of each selected game
    for name in game_names:
        index = games.index[games["name"] == name][0]
        selected_game_indexes.append(index)


    #stores the similarity between each selected game and every game in the database
    tag_similarities = []
    genre_similarities = []
    category_similarities = []      

    for index in selected_game_indexes: 
        tag_similarities.append(cosine_similarity(tag_matrix[index], tag_matrix)[0])
        genre_similarities.append(cosine_similarity(genre_matrix[index], genre_matrix)[0])
        category_similarities.append(cosine_similarity(category_matrix[index], category_matrix)[0])


    #calculates how similar each game is to the average of the selected games
    average_popularity = games.loc[selected_game_indexes, "popularity"].mean()
    popularity_similarity = 1 - abs(games["popularity"] - average_popularity)

    average_rating = games.loc[selected_game_indexes, "rating"].mean()
    rating_similarity = (1 - abs(games["rating"] - average_rating)).fillna(0)

    average_release_year = games.loc[selected_game_indexes,"release_year"].mean()
    release_date_similarity = (1 / (1 + abs(games["release_year"] - average_release_year))).fillna(0)


    #finds average similarity for the games
    tag_similarity = np.mean(tag_similarities, axis=0)
    genre_similarity = np.mean(genre_similarities, axis=0)
    category_similarity = np.mean(category_similarities, axis=0)


    rating_score = games["rating"].fillna(0)

    #combines similarity factors into one overall score
    combined_similarity = (
        0.35 * tag_similarity +
        0.15 * genre_similarity +
        0.10 * category_similarity +
        0.08 * popularity_similarity + 0.08 * games["popularity"] + #gives a slight advantage to more popular and higher rated games
        0.08 * rating_similarity + 0.08 * rating_score +
        0.08 * release_date_similarity
    )


    count = 0

    #stores selected games so that they cant be recommended again
    seen_games = set(games.loc[selected_game_indexes, "name"]) 
    seen_metacritic = set(games.loc[selected_game_indexes, "metacritic_id"].dropna())

    #creates an array of indexes corresponding to the games similarity from highest to lowest
    similar_games = combined_similarity.argsort()[::-1] 


    for index in similar_games:

        game_name = games.loc[index, "name"]
        
        metacritic_id = games.loc[index, "metacritic_id"]

        #makes sure there are no repeated games
        if game_name in seen_games:
            continue

        if pd.notna(metacritic_id):
            if metacritic_id in seen_metacritic:
                continue

        seen_games.add(game_name)
        if pd.notna(metacritic_id):
            seen_metacritic.add(metacritic_id)



        #compares this recommendation to each selected game
        individual_scores = []

        for i in range(len(selected_game_indexes)):

            current_popularity = float(games.loc[selected_game_indexes[i], "popularity"])
            recommended_popularity = float(games.loc[index, "popularity"])
            popularity_similarity = 1 - abs(recommended_popularity - current_popularity)


            current_rating = float(games.loc[selected_game_indexes[i], "rating"])
            recommended_rating = float(games.loc[index, "rating"])

            #games without ratings recieve no rating similarity
            if pd.isna(current_rating) or pd.isna(recommended_rating):
                rating_similarity = 0
            else:
                rating_similarity = 1 - abs(recommended_rating - current_rating)

            #uses the recommended games rating as part of its score
            if pd.isna(recommended_rating):
                recommended_rating_score = 0
            else:
                recommended_rating_score = recommended_rating 


            current_release_year = games.loc[selected_game_indexes[i], "release_year"]
            recommended_release_year = games.loc[index, "release_year"]

            #games without release years recieve no release date similarity
            if pd.isna(current_release_year) or pd.isna(recommended_release_year):
                release_date_similarity = 0
            else:
                release_date_similarity = 1 / (1 + abs(int(recommended_release_year) - int(current_release_year)))


            #calculates similarity between this recommendation and the current selected game
            similarity = (
                0.35 * tag_similarities[i][index] +
                0.15 * genre_similarities[i][index] +
                0.10 * category_similarities[i][index]+
                0.08 * popularity_similarity +
                0.08 * recommended_popularity +
                0.08 * rating_similarity +
                0.08 * recommended_rating_score +
                0.08 * release_date_similarity
            )

            individual_scores.append(similarity)

        #finds which selected game this recommendation is most similar to
        most_similar_index = np.argmax(individual_scores) 
        most_similar_name = game_names[most_similar_index]
        most_similar_game_index = selected_game_indexes[most_similar_index]


        # find and rank shared tags
        recommended_tags = set(games.loc[index, "tags"].split())
        current_game_tags = set(games.loc[most_similar_game_index, "tags"].split())

        shared_tags = rank_shared_features(recommended_tags, current_game_tags, tag_matrix, tag_vectorizer, index)

        # find and rank shared genres
        recommended_genres = set(games.loc[index, "genres"].split())
        current_game_genres = set(games.loc[most_similar_game_index, "genres"].split())

        shared_genres = rank_shared_features(recommended_genres, current_game_genres, genre_matrix, genre_vectorizer, index)

        # find and rank shared categories
        recommended_categories = set(games.loc[index, "categories"].split())
        current_game_categories = set(games.loc[most_similar_game_index, "categories"].split())

        shared_categories = rank_shared_features(recommended_categories, current_game_categories, category_matrix, category_vectorizer, index)


        #formats the release year for the frontend
        if pd.isna(games.loc[index, "release_year"]):
            release_year = " Not Available"
        else:
            release_year = int(games.loc[index, "release_year"])

        #formats the rating for the frontend
        if pd.isna(games.loc[index, "rating"]):
            rating = " Not Available"
        else:
            rating = round(float(games.loc[index, "rating"]) * 100, 1)


        #creates the recommendation data to be returned to the frontend
        recommendations.append({
            "name": game_name,
            "similarity": round(float(combined_similarity[index]), 3),
            "most_similar_to": most_similar_name,
            "shared_tags": shared_tags,
            "shared_genres": shared_genres,
            "shared_categories": shared_categories,
            "release_year": release_year,
            "rating": rating,
            "price": games.loc[index, "price"]
        })


        count += 1

        if count == 50:
            break

    return recommendations
  

#starts flask server
if __name__ == "__main__":
    app.run(port = 5000)
