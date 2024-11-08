from flask import Flask, render_template, request
import pandas as pd
from model import collaborative_filtering, content_based_filtering

app = Flask(__name__)

# Load data
users = pd.read_csv('users.csv')
content = pd.read_csv('content.csv')
interactions = pd.read_csv('interactions.csv')
browsing_history = pd.read_csv('browsing_history.csv')

def recommend_users_to_follow(user_id, users, interactions):
    # Get the interests of the target user
    user_interests = users.loc[users['user_id'] == user_id, 'interests'].values[0].split(';')
    user_following = users.loc[users['user_id'] == user_id, 'following'].values[0].split(';')
    user_following = set(map(int, user_following))  # Convert to set of ints for easy comparison

    # Find users with similar interests, exclude already followed users
    similar_users = users[users['user_id'] != user_id].copy()  # Exclude current user
    similar_users['similar_interest_score'] = similar_users['interests'].apply(
        lambda interests: len(set(interests.split(';')).intersection(user_interests))
    )

    # Filter out users already followed and sort by interest score and follower count
    recommended_users = similar_users[~similar_users['user_id'].isin(user_following)]
    recommended_users = recommended_users[recommended_users['similar_interest_score'] > 0]
    recommended_users = recommended_users.sort_values(by=['similar_interest_score', 'followers_count'], ascending=[False, False])

    # Get top 5 recommended users
    recommended_users = recommended_users[['user_id', 'name', 'followers_count', 'following', 'interests']].head(5)

    # Preprocess following names (split 'following' IDs and map to names)
    recommended_users['following_names'] = recommended_users['following'].apply(
        lambda following: [users.loc[users['user_id'] == int(follow_id), 'name'].values[0] 
                           for follow_id in following.split(';')] if following else []
    )

    # Clean the 'interests' by replacing semicolons with commas (and adding a space after commas)
    recommended_users['interests'] = recommended_users['interests'].apply(lambda interests: interests.replace(';', ', ') if interests else '')

    # Convert to a list of dictionaries for template rendering
    return recommended_users.to_dict(orient='records')

@app.route('/')
def index():
    return render_template('index.html', content=content.to_dict(orient='records'))

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    user_id = int(request.form['user_id'])
    algorithm = request.form['algorithm']

    # Get content IDs the user has interacted with
    interacted_content_ids = interactions[interactions['user_id'] == user_id]['content_id'].unique()
    browsed_content_ids = browsing_history[browsing_history['user_id'] == user_id]['content_id'].unique()

    # Content the user has already interacted with
    interacted_content = content[content['content_id'].isin(interacted_content_ids) | content['content_id'].isin(browsed_content_ids)]

    # Generate recommendations based on the selected algorithm
    if algorithm == 'collaborative':
        recommendations = collaborative_filtering(user_id, interactions, content, users)
    elif algorithm == 'content-based':
        recommendations = content_based_filtering(user_id, interactions, browsing_history, content)
    elif algorithm == 'hybrid':
        collaborative_recommendations = collaborative_filtering(user_id, interactions, content, users)
        content_based_recommendations = content_based_filtering(user_id, interactions, browsing_history, content)
        recommendations = pd.concat([collaborative_recommendations, content_based_recommendations]).drop_duplicates()

    # Ensure 'content_id' column exists before proceeding
    if 'content_id' not in recommendations.columns:
        print("Warning: 'content_id' column is missing from recommendations.")
        return "Error: 'content_id' column is missing."

    # Exclude content the user has already interacted with
    recommended_content = recommendations[~recommendations['content_id'].isin(interacted_content_ids) &
                                          ~recommendations['content_id'].isin(browsed_content_ids)]

    # Prepare data for rendering
    interacted_content = interacted_content[['content_id', 'title', 'category', 'popularity']].copy()
    interacted_content['source'] = interacted_content['content_id'].apply(
        lambda x: 'Interacted' if x in interacted_content_ids else 'Browsed'
    )

    recommended_content = recommended_content[['content_id', 'title', 'category', 'popularity']].copy()
    recommended_content['source'] = 'Recommended'

    # Get users to follow based on recommendations
    users_to_follow = recommend_users_to_follow(user_id, users, interactions)

    return render_template('recommendations.html', 
                       interacted_content=interacted_content.to_dict(orient='records'),
                       recommended_content=recommended_content.to_dict(orient='records'),
                       users_to_follow=users_to_follow)

if __name__ == '__main__':
    app.run(debug=True)
