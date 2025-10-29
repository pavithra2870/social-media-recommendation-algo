from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from model import collaborative_filtering, content_based_filtering

app = Flask(__name__)

# Load data
users = pd.read_csv('users.csv')
content = pd.read_csv('content.csv')
interactions = pd.read_csv('interactions.csv')
browsing_history = pd.read_csv('browsing_history.csv')

def _user_exists(user_id: int) -> bool:
    try:
        return not users.empty and (users['user_id'] == user_id).any()
    except Exception:
        return False

def recommend_users_to_follow(user_id, users, interactions):
    # Get the interests of the target user
    if not _user_exists(user_id):
        return []
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

def _interest_options_from_data(users_df, content_df):
    user_interests = set()
    if 'interests' in users_df.columns and not users_df.empty:
        for v in users_df['interests'].dropna().tolist():
            for tok in str(v).split(';'):
                tok = tok.strip().lower()
                if tok:
                    user_interests.add(tok)
    content_cats = set()
    if 'category' in content_df.columns and not content_df.empty:
        for v in content_df['category'].dropna().tolist():
            v = str(v).strip().lower()
            if v:
                content_cats.add(v)
    merged = sorted(user_interests.union(content_cats))
    return merged

def _create_user(name, interests_list):
    global users
    new_id = int(users['user_id'].max()) + 1 if not users.empty else 1
    interests_str = ';'.join([i.strip().lower() for i in interests_list if i.strip()])
    following_default = '1' if (not users.empty and (users['user_id'] == 1).any()) else ''
    new_row = {
        'user_id': new_id,
        'name': name,
        'interests': interests_str,
        'followers_count': 0,
        'following': following_default,
        'activity_level': 'low'
    }
    users = pd.concat([users, pd.DataFrame([new_row])], ignore_index=True)
    users.to_csv('users.csv', index=False)
    return new_id

@app.route('/signup', methods=['GET'])
def signup_page():
    interest_options = _interest_options_from_data(users, content)
    return render_template('signup.html', interest_options=interest_options)

@app.route('/api/signup', methods=['POST'])
def api_signup():
    name = request.form.get('name', '').strip()
    interests_selected = request.form.getlist('interests')
    if not name:
        return redirect(url_for('signup_page'))
    user_id = _create_user(name, interests_selected)
    return redirect(url_for('recommend_auto', user_id=user_id, algorithm='hybrid'))

@app.route('/recommend_auto', methods=['GET'])
def recommend_auto():
    try:
        user_id = int(request.args.get('user_id'))
    except (TypeError, ValueError):
        return redirect(url_for('index'))
    algorithm = request.args.get('algorithm', 'hybrid')

    if not _user_exists(user_id):
        return "User does not exist", 200

    interacted_content_ids = interactions[interactions['user_id'] == user_id]['content_id'].unique()
    browsed_content_ids = browsing_history[browsing_history['user_id'] == user_id]['content_id'].unique()

    interacted_content_df = content[content['content_id'].isin(interacted_content_ids) | content['content_id'].isin(browsed_content_ids)]

    if algorithm == 'collaborative':
        recommendations = collaborative_filtering(user_id, interactions, content, users)
    elif algorithm == 'content-based':
        recommendations = content_based_filtering(user_id, interactions, browsing_history, content)
    else:
        collaborative_recommendations = collaborative_filtering(user_id, interactions, content, users)
        content_based_recommendations = content_based_filtering(user_id, interactions, browsing_history, content)
        recommendations = pd.concat([collaborative_recommendations, content_based_recommendations]).drop_duplicates()

    if 'content_id' not in recommendations.columns:
        return "Error: 'content_id' column is missing."

    recommended_content_df = recommendations[~recommendations['content_id'].isin(interacted_content_ids) &
                                            ~recommendations['content_id'].isin(browsed_content_ids)]

    interacted_content_df = interacted_content_df[['content_id', 'title', 'category', 'popularity']].copy()
    interacted_content_df['source'] = interacted_content_df['content_id'].apply(
        lambda x: 'Interacted' if x in interacted_content_ids else 'Browsed'
    )

    # Keep original source if present; default to 'Recommended'
    if 'source' in recommended_content_df.columns:
        recommended_content_df = recommended_content_df[['content_id', 'title', 'category', 'popularity', 'source']].copy()
    else:
        recommended_content_df = recommended_content_df[['content_id', 'title', 'category', 'popularity']].copy()
        recommended_content_df['source'] = 'Recommended'

    # Ranking: similarity to user's interests + popularity normalization
    user_interests_raw = users.loc[users['user_id'] == user_id, 'interests'].astype(str).values
    user_interest_set = set()
    if len(user_interests_raw) > 0:
        user_interest_set = set([t.strip().lower() for t in user_interests_raw[0].split(';') if t.strip()])

    max_pop = content['popularity'].max() if 'popularity' in content.columns and not content.empty else 1
    def _score_row(row):
        cat = str(row.get('category', '')).strip().lower()
        sim = 1.0 if cat in user_interest_set else 0.0
        pop = float(row.get('popularity', 0)) / float(max_pop or 1)
        # optional source boost: CF > CBF > others
        src = str(row.get('source', '')).lower()
        src_boost = 0.2 if 'collaborative' in src else (0.1 if 'content-based' in src else 0.0)
        return 2.0 * sim + 1.0 * pop + src_boost

    recommended_content_df['score'] = recommended_content_df.apply(_score_row, axis=1)
    recommended_content_df = recommended_content_df.sort_values(by=['score', 'popularity'], ascending=[False, False]).drop(columns=['score'])

    users_to_follow = recommend_users_to_follow(user_id, users, interactions)

    return render_template('recommendations.html',
                           interacted_content=interacted_content_df.to_dict(orient='records'),
                           recommended_content=recommended_content_df.to_dict(orient='records'),
                           users_to_follow=users_to_follow)

@app.route('/')
def index():
    return render_template('index.html', content=content.to_dict(orient='records'))

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    try:
        user_id = int(request.form['user_id'])
    except (TypeError, ValueError):
        return "User does not exist", 200
    algorithm = request.form['algorithm']

    if not _user_exists(user_id):
        return "User does not exist", 200

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

    # Keep original source if present; default to 'Recommended'
    if 'source' in recommended_content.columns:
        recommended_content = recommended_content[['content_id', 'title', 'category', 'popularity', 'source']].copy()
    else:
        recommended_content = recommended_content[['content_id', 'title', 'category', 'popularity']].copy()
        recommended_content['source'] = 'Recommended'

    # Ranking: similarity to user's interests + popularity normalization
    user_interests_raw = users.loc[users['user_id'] == user_id, 'interests'].astype(str).values
    user_interest_set = set()
    if len(user_interests_raw) > 0:
        user_interest_set = set([t.strip().lower() for t in user_interests_raw[0].split(';') if t.strip()])

    max_pop = content['popularity'].max() if 'popularity' in content.columns and not content.empty else 1
    def _score_row2(row):
        cat = str(row.get('category', '')).strip().lower()
        sim = 1.0 if cat in user_interest_set else 0.0
        pop = float(row.get('popularity', 0)) / float(max_pop or 1)
        src = str(row.get('source', '')).lower()
        src_boost = 0.2 if 'collaborative' in src else (0.1 if 'content-based' in src else 0.0)
        return 2.0 * sim + 1.0 * pop + src_boost

    recommended_content['score'] = recommended_content.apply(_score_row2, axis=1)
    recommended_content = recommended_content.sort_values(by=['score', 'popularity'], ascending=[False, False]).drop(columns=['score'])

    # Get users to follow based on recommendations
    users_to_follow = recommend_users_to_follow(user_id, users, interactions)

    return render_template('recommendations.html', 
                       interacted_content=interacted_content.to_dict(orient='records'),
                       recommended_content=recommended_content.to_dict(orient='records'),
                       users_to_follow=users_to_follow)

if __name__ == '__main__':
    app.run(debug=True)