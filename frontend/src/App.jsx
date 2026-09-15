import { useEffect, useState } from 'react'
import './App.css'

export default function App() {
  const [currentSearch, setCurrentSearch] = useState('') //stores current text in search bar
  const [selectedGames, setSelectedGames] = useState([]) //stores games selected by the user
  const [searchResults, setSearchResults] = useState([]) //stores games returned by the search
  const [recommendations, setRecommendations] = useState([]) //stores returned recommendations

  //searches for games whenever the search input changes
  useEffect(() => { 

    //empty search check
    if (!currentSearch) {
      setSearchResults([])
      return
    }

    //cleans search so the search matches the cleaned game names
    const search = currentSearch.toLowerCase().replace(/[^a-z0-9]/g, '')

    //requests matching names from the backend
    fetch(`http://127.0.0.1:5000/games?search=${search}`)
      .then((response) => response.json())
      .then((data) => {
        setSearchResults(data)
      })
  }, [currentSearch])


  //removes games that have already been selected from the search results
  const matchingGames = searchResults.filter((searchResult) => {   
    return !selectedGames.includes(searchResult)
  })
  

  //sends the selected games to the backend to get recommendations
  function getRecommendations(selectedGames) {
    fetch('http://127.0.0.1:5000/recommend', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({games: selectedGames})
    })
      .then((response) => response.json())
      .then((data) => {
        setRecommendations(data)
      })
  }


  //adds a game to the selected games and gets updated recommendations
  function selectGame(gameName) {
    const newSelectedGames = [...selectedGames, gameName]

    setSelectedGames(newSelectedGames)
    setCurrentSearch('') 

    getRecommendations(newSelectedGames)
  }


  //removes a game from the selected games and gets updated recommendations
  function removeGame(gameName) {
    const newSelectedGames = selectedGames.filter((selectedGame) => selectedGame !== gameName)

    setSelectedGames(newSelectedGames)

    //removes recommendations when there are no selected games remaining
    if (newSelectedGames.length === 0) {
      setRecommendations([])
      return
    }

    getRecommendations(newSelectedGames)
}


  //replaces underscores with spaces when displaying tags, genres and categories
  function formatName(name) {
    return name.replace(/_/g, " ")
  }


  return (
    <div>
      <header>
        <h1>Steam Game Recommender</h1>
      </header>

      <main>
        <section className="search-section">
          <h2><strong>Enter Games</strong></h2>


        <div className="search-container">
          <input value={currentSearch} onChange={(event) => setCurrentSearch(event.target.value)}/>

        {/* only displays search results when the user has entered text */}
        {currentSearch && (
          <div className="search-results">
            {matchingGames.map((searchResult) => ( 
              <div className="search-result" key={searchResult} onClick={() => selectGame(searchResult)}>
                {searchResult}
              </div>
            ))}
          </div>
          )}
        </div>

          <h3>Selected games:</h3>

          {selectedGames.map((gameName) => (
            <div className="selected-game" key={gameName}>
              {gameName}
              <button className="remove-game" onClick={() => removeGame(gameName)}>
                x
              </button>
            </div>
          ))}
        </section>

        <section className="recommendations">
          <h2>Recommendations</h2>
            {recommendations.map((recommendation) => (
              <div className="recommendation-card" key={recommendation.name}>
                <h3>{recommendation.name}</h3>
                <p>
                  <strong>Similarity Score:</strong> {Math.round(recommendation.similarity * 100)}% {" - "}
                  <strong>Price:</strong> {recommendation.price} {" - "}
                  <strong>Rating:</strong> {recommendation.rating === " Not Available" ? " Not Available" : Math.round(recommendation.rating) + "%"} {" - "}
                  <strong>Release Year:</strong> {recommendation.release_year}
                </p>

                <div className="similar-to">
                  <p>
                    <strong>Most similar to:</strong> {recommendation.most_similar_to}
                  </p>
                </div>

                <div className="similarity">
                  <h4>How it's similar</h4>

                  <p>
                    <strong>Tags:</strong> {" "}
                    {recommendation.shared_tags.map(formatName).join(", ")}
                  </p>

                  <p>
                    <strong>Genres:</strong> {" "}
                    {recommendation.shared_genres.map(formatName).join(", ")}
                  </p>

                  <p>
                    <strong>Categories:</strong> {" "}
                    {recommendation.shared_categories.map(formatName).join(", ")}
                  </p>
                </div>
              </div>
            ))}
        </section>
      </main>
    </div>
  )
}


  