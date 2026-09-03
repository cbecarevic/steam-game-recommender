import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



games = pd.read_csv("data/steam_games.csv")


#data cleaning




games = games.dropna(subset=["name"]) #removes all games without names 

#removes entries that are not technically 'games' and dont want to be considered
games = games.drop(games.index[games["tags"].str.contains('"Soundtrack"', na=False)]) 
games = games.drop(games.index[games["categories"].str.contains("Downloadable Content", na=False)])

games = games.reset_index(drop=True)


#processes fields with multiple items so tfidf can read it


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

    return " ".join(items)
    
#tags can have either [] or {} format
#genres and categories can either have list or descriptions format


games["tags"] = games["tags"].apply(clean_text)

games["genres"] = games["genres"].apply(clean_text)

games["categories"] = games["categories"].apply(clean_text) #can maybe change to clean tags



games["release_date"] = pd.to_datetime(games["release_date"])


#tfid vectors
#tfid measures the signifance of words 




game_names = ["METAL GEAR SOLID - Master Collection Version","Portal","The Witness","ELDEN RING", "Hollow Knight", "Nine Sols","DARK SOULS™ II"]
for name in game_names:
    matches = games.index[games["name"] == name]

    if len(matches) == 0:
        print("NOT FOUND:", name)
    else:
        print("FOUND:", name)


for name in game_names:
    print(name, (games["name"] == name).sum())




game_indices = [games.index[games["name"] == name][0]for name in game_names]




tag_vectorizer = TfidfVectorizer(lowercase=False)
tag_vectors = tag_vectorizer.fit_transform(games["tags"])

genre_vectorizer = TfidfVectorizer(lowercase=False)
genre_vectors = genre_vectorizer.fit_transform(games["genres"])

category_vectorizer = TfidfVectorizer(lowercase=False)
category_vectors = category_vectorizer.fit_transform(games["categories"])





#make sure isnt slop
tag_similarities = [cosine_similarity(tag_vectors[index], tag_vectors)[0] for index in game_indices]

genre_similarities = [cosine_similarity(genre_vectors[index], genre_vectors)[0] for index in game_indices]

category_similarities = [cosine_similarity(category_vectors[index], category_vectors)[0] for index in game_indices]




#make sure isnt slop
games["popularity"] = np.log1p(games["recommendations"]).fillna(0) #uses
games["popularity"] = games["popularity"] / games["popularity"].max()

selected_popularity = games.loc[game_indices, "popularity"].mean()
popularity_similarity = 1 - abs(games["popularity"] - selected_popularity) #abs??-




games["rating"] = (games["positive"] / (games["positive"] + games["negative"])).fillna(0)

selected_rating = games.loc[game_indices, "rating"].mean()
rating_similarity = 1 - abs(games["rating"] - selected_rating)




#make sure isnt slop
tag_similarity = np.mean(tag_similarities, axis=0)
genre_similarity = np.mean(genre_similarities, axis=0)
category_similarity = np.mean(category_similarities, axis=0)



combined_similarity = (
    0.50* tag_similarity +
    0.25 * genre_similarity +
    0.15 * category_similarity +
    0.05 * popularity_similarity + 0.025 * games["popularity"] + #gives a slight advantage to more popular games
    0.05 * rating_similarity + 0.025 * games["rating"]
)



individual_similarities = [] #check this section isnt slop

for i in range(len(game_indices)):

    similarity = (
        0.50 * tag_similarities[i] +
        0.25 * genre_similarities[i] +
        0.15 * category_similarities[i]
    )

    individual_similarities.append(similarity)




count = 0
seen_games = set(games.loc[game_indices, "name"]) #idk whatt tf thus line is


#make sure isnt slop and something i would do
similar_games = combined_similarity.argsort()[::-1]

for index in similar_games:

    game_name = games.loc[index, "name"]

    if game_name in seen_games:
        continue

    seen_games.add(game_name)





     #slop from here

    #find which liked game this recommendation is most similar to
    most_similar_index = np.argmax([
        similarities[index]
        for similarities in individual_similarities
    ])

    most_similar_name = game_names[most_similar_index]

    #get the dataframe index of the liked game
    liked_index = game_indices[most_similar_index]




     #find shared tags
    recommended_tags = set(games.loc[index, "tags"].split())
    liked_tags = set(games.loc[liked_index, "tags"].split())
    shared_tags = recommended_tags & liked_tags

    #rank shared tags by TF-IDF importance
    tag_scores = tag_vectors[index].toarray()[0]
    tag_names = tag_vectorizer.get_feature_names_out()

    shared_tags = sorted(
        shared_tags,
        key=lambda tag: tag_scores[tag_names.tolist().index(tag)],
        reverse=True
    )[:5]





    #find shared genres
    recommended_genres = set(games.loc[index, "genres"].split())
    liked_genres = set(games.loc[liked_index, "genres"].split())
    shared_genres = recommended_genres & liked_genres

    #find shared categories
    recommended_categories = set(games.loc[index, "categories"].split())
    liked_categories = set(games.loc[liked_index, "categories"].split())
    shared_categories = recommended_categories & liked_categories

    print(
        game_name,
        round(combined_similarity[index], 3),
        "Most similar to:", most_similar_name
    )

    print("Why:")
    print("Shared tags:", ", ".join(list(shared_tags)[:5]))
    print("Shared genres:", ", ".join(list(shared_genres)[:5]))
    print("Shared categories:", ", ".join(list(shared_categories)[:5]))

    print()

    count = count + 1

    if count == 10:
        break


#cd "C:\Users\cbeca\Downloads\coding project"
#python game-recommender.py


#todo
#what game is most similar to
#explain why games were chosen
#add options to include soundtracks / dlc
#price filter
#front end

