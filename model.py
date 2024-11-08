import pandas as pd

def collaborative_filtering(user_id, interactions, content, users, min_recommendations=5, is_user_based=False):
    # Find content that the target user has engaged with
    user_interactions = interactions[interactions['user_id'] == user_id]['content_id'].unique()
    
    if is_user_based:
        # Implement user-based collaborative filtering logic
        other_users = interactions[interactions['content_id'].isin(user_interactions)]['user_id'].unique()
        other_content = interactions[interactions['user_id'].isin(other_users)]['content_id'].unique()
    else:
        # Implement item-based collaborative filtering logic
        other_content = interactions[interactions['user_id'] != user_id]['content_id'].unique()

    # Exclude content the user has already interacted with
    recommendations = content[~content['content_id'].isin(user_interactions) & 
                              content['content_id'].isin(other_content)].copy()
    recommendations['source'] = 'Collaborative Filtering'

    # If fewer than min_recommendations are found, add popular content to fill the gap
    if len(recommendations) < min_recommendations:
        popular_content_ids = interactions['content_id'].value_counts().index[:min_recommendations]
        popular_recommendations = content[content['content_id'].isin(popular_content_ids) & 
                                          ~content['content_id'].isin(user_interactions)].copy()
        popular_recommendations['source'] = 'Popular Content Fallback'
        recommendations = pd.concat([recommendations, popular_recommendations]).drop_duplicates()
    
    # Further fill with random unseen content if still below min_recommendations
    if len(recommendations) < min_recommendations:
        unseen_content = content[~content['content_id'].isin(user_interactions)]
        random_recommendations = unseen_content.sample(n=min_recommendations - len(recommendations), random_state=42)
        random_recommendations['source'] = 'Random Unseen Content'
        recommendations = pd.concat([recommendations, random_recommendations]).drop_duplicates()

    # Ensure the necessary columns are present
    required_columns = ['content_id', 'title', 'category', 'popularity']
    missing_columns = [col for col in required_columns if col not in recommendations.columns]
    for col in missing_columns:
        recommendations[col] = None  # Handle missing columns gracefully by adding placeholders or default values
    
    # Return final recommendations limited to min_recommendations
    return recommendations[['content_id', 'title', 'category', 'popularity', 'source']].head(min_recommendations)

def content_based_filtering(user_id, interactions, browsing_history, content):
    # User's browsing history of content
    user_history = browsing_history[browsing_history['user_id'] == user_id]['content_id'].unique()
    user_content = content[content['content_id'].isin(user_history)]

    # Recommend similar content by category
    if not user_content.empty:
        recommendations = content[content['category'].isin(user_content['category']) &
                                  ~content['content_id'].isin(user_history)].copy()
    else:
        recommendations = pd.DataFrame()  # Return empty if no content in user history

    # Ensure the necessary columns are present
    required_columns = ['content_id', 'title', 'category', 'popularity']
    missing_columns = [col for col in required_columns if col not in recommendations.columns]
    for col in missing_columns:
        recommendations[col] = None  # Handle missing columns gracefully by adding placeholders or default values

    recommendations['source'] = 'Content-Based Filtering'
    
    # Ensure we return the right columns
    return recommendations[['content_id', 'title', 'category', 'popularity', 'source']]

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
        lambda following: [users.loc[users['user_id'] == int(follow_id), 'name'].values[0] for follow_id in following.split(';')]
    )
    
    # Clean the 'interests' by removing semicolons and replacing with a space or other separator
    recommended_users['interests'] = recommended_users['interests'].apply(lambda interests: interests.replace(';', ', '))

    # Convert to a list of dictionaries for template rendering
    return recommended_users.to_dict(orient='records')

def hybrid_recommendation(user_id, interactions, browsing_history, content, users):
    # Get collaborative filtering recommendations
    collaborative_recommendations = collaborative_filtering(user_id, interactions, content, users)
    
    # Get content-based filtering recommendations
    content_based_recommendations = content_based_filtering(user_id, interactions, browsing_history, content)

    # Combine both sets of recommendations
    hybrid_recommendations = pd.concat([collaborative_recommendations, content_based_recommendations]).drop_duplicates(subset=['content_id'])

    # User's browsing history to exclude already seen content
    user_history = browsing_history[browsing_history['user_id'] == user_id]['content_id'].unique()
    unseen_content = content[~content['content_id'].isin(user_history)].copy()

    # Add random unseen content to diversify recommendations
    if len(unseen_content) > 5:
        random_recommendations = unseen_content.sample(n=5).copy()
    else:
        random_recommendations = unseen_content.copy()
    random_recommendations['source'] = 'Random Unseen Content'

    # Add popular content (top 5 most engaged content)
    popular_content = interactions['content_id'].value_counts().index[:5]
    popular_recommendations = content[content['content_id'].isin(popular_content)].copy()
    popular_recommendations['source'] = 'Popular Content'

    # Combine all recommendations
    all_recommendations = pd.concat([hybrid_recommendations, random_recommendations, popular_recommendations])

    # Final deduplication and prioritization by recommendation source
    final_recommendations = all_recommendations.drop_duplicates(subset=['content_id'], keep='first')

    # Debugging print to see the final recommendations
    print("Final Recommendations:\n", final_recommendations[['content_id', 'title', 'source']])

    # Recommend users to follow
    users_to_follow = recommend_users_to_follow(user_id, users, interactions)
    print(f"Users to follow for user {user_id}: {users_to_follow}")

    return final_recommendations[['content_id', 'title', 'category', 'popularity', 'source']], users_to_follow
