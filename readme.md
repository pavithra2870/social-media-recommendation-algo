Slightly edited version of https://github.com/karan3691's https://github.com/karan3691/social-media-recommender. Thanks a lot!!
# Social Media Content Recommender

This project is a **Social Media Content Recommender** that provides personalized content recommendations for users based on their interactions, preferences, and browsing history. The recommendation system is built with several filtering techniques to optimize user engagement and suggest relevant content.

## Features

- **Personalized Content Recommendations**: Suggests content based on the user's past interactions and selected recommendation algorithm.
- **Multiple Recommendation Algorithms**:
  - **Collaborative Filtering**: Recommends items based on the behavior of similar users.
  - **Content-Based Filtering**: Recommends items that are similar to the ones the user has previously liked or interacted with.
  - **Hybrid Filtering**: Combines collaborative and content-based filtering for better accuracy.
- **User Suggestions**: Recommends users to follow based on mutual interests and existing connections.


## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/karan3691/social-media-recommender.git
   cd social-media-recommender
   ```

2. **Install Dependencies**:
Make sure you have Python and pip installed. Then install the necessary packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

   The application will start, and you can access it in your browser at http://127.0.0.1:5000.


##  Usage

1. **Collaborative Filtering**:
Uses data from other users who have interacted with similar content to recommend items the current user has not seen.

2. **Content-Based Filtering:**:
Analyzes the content's categories or features to suggest similar items.

3. **Hybrid Filtering**:
A blend of collaborative and content-based approaches to provide well-rounded recommendations.


## Technologies Used

1. **Flask**: To handle the backend and routing.
2. **Pandas**: For data manipulation in the recommendation logic.
3. **HTML & CSS**: For frontend structure and styling.

---

# Comprehensive Documentation

## Project Overview

This repository implements a minimal, end-to-end Social Media Recommender:

- **Goal**: Recommend relevant content and users to follow, using lightweight algorithms suitable for demos and small datasets.
- **Algorithms**: Collaborative Filtering (CF), Content-Based Filtering (CBF), and a Hybrid approach, plus a simple user recommendation based on interest overlap.
- **Ranking**: Recommendations are ranked using a score that combines category-interest match, normalized popularity, and source boosts.

## Architecture

- **app.py**: Flask routes, request handling, ranking, and rendering.
- **model.py**: Core recommenders (CF, CBF, hybrid helper), users-to-follow (alternate version).
- **templates/**: Jinja2 templates (`index.html`, `recommendations.html`, `signup.html`).
- **static/**: CSS styling.
- **CSV data**: `users.csv`, `content.csv`, `interactions.csv`, `browsing_history.csv`.

High-level flow:

1. Load CSVs into Pandas DataFrames.
2. On request, get the target `user_id` and selected algorithm.
3. Generate candidates via CF/CBF/Hybrid.
4. Exclude already seen items (interacted or browsed).
5. Rank candidates by similarity and popularity.
6. Render results and users-to-follow.

## Data Schema

- `users.csv`
  - `user_id` (int), `name` (str), `interests` (semicolon-separated lowercase tags), `followers_count` (int), `following` (semicolon-separated user_ids), `activity_level` (str)
- `content.csv`
  - `content_id` (int), `title` (str), `category` (one of Environment, Fashion, Food, Health, Photography, Science, Self improvement, Technology, Travel), `popularity` (int), `type` (post/article)
- `interactions.csv`
  - `user_id` (int), `content_id` (int), `interaction_type` (e.g., viewed, liked)
- `browsing_history.csv`
  - `user_id` (int), `content_id` (int aligned with `content.csv`), `timestamp` (ISO-like)

Note: Browsing history is aligned with `content.csv` IDs so CBF works correctly.

## Algorithms

- **Collaborative Filtering (CF)** — `model.collaborative_filtering`
  - Item-based by default; optional user-based path via `is_user_based=True`.
  - Steps:
    - Get items the user interacted with.
    - Item-based: gather other items engaged by other users; exclude already seen.
    - Fallbacks: fill to a minimum with popular items, then random unseen.
  - Output columns: `content_id, title, category, popularity, source`.

- **Content-Based Filtering (CBF)** — `model.content_based_filtering`
  - Use the user’s browsing history to infer categories.
  - Recommend content sharing those categories, excluding history.

- **Hybrid** — `model.hybrid_recommendation` and route-level concatenation
  - Concatenate CF + CBF, deduplicate by `content_id`.
  - Diversification via random unseen and popular content (in model helper).

- **Users to Follow** — `app.recommend_users_to_follow`
  - Interest-overlap score = count of shared interest tags with target user.
  - Exclude already-followed, sort by overlap then followers_count, top-5.

## Ranking Logic (Route-level)

Applied in both `/recommend` and `/recommend_auto` after candidate generation:

- Build `user_interest_set` from `users.interests` for the target user.
- For each candidate item:
  - `sim = 1.0` if `category ∈ user_interest_set`, else `0.0`.
  - `pop = item.popularity / max(popularity)` to normalize (0..1).
  - `src_boost = 0.2` if source contains “collaborative”, `0.1` if “content-based”, else `0.0`.
  - `score = 2.0 * sim + 1.0 * pop + src_boost`.
- Sort by `score` desc, then by `popularity` desc.

This produces intuitive ordering: items matching interests come first, then more popular content, with a slight preference for CF-derived items.

## Complexity and Efficiency

Let U users, I items, E interactions, H = user’s browsing rows.

- CF per request: O(E + I)
- CBF per request: O(H + I)
- Hybrid: O(E + I)
- Users-to-follow: O(U log U) (overlap + sort)

No model training required; suitable for small datasets and demos.

## Correctness Sketches

- Exclusion invariant: already seen items (interacted/browsed) are filtered out before ranking.
- Feasibility: popular/random fallbacks ensure at least a minimum number of recommendations if dataset allows.
- Relevance: CF leverages collective behavior; CBF matches topical categories; ranking aligns to user interests.

## Installation

```bash
pip install -r requirements.txt
python app.py
```

Visit http://127.0.0.1:5000

## Usage

- Home page: enter `user_id` and choose algorithm (Collaborative, Content-Based, Hybrid).
- Signup page (`/signup`): create a new user by selecting interests; you’ll be redirected to recommendations.
- Invalid user handling: if `user_id` not found, routes return a friendly “User does not exist” message (HTTP 200).

## Routes

- `GET /` — Home form
- `POST /recommend` — Generate recommendations for provided user/algorithm
- `GET /recommend_auto?user_id=&algorithm=` — Same as above, useful after signup
- `GET /signup` — Signup form
- `POST /api/signup` — Create user and redirect to recommendations

## Limitations

- Simple CF/CBF heuristics; no learned similarity, no time decay, no action weighting (view vs like).
- Popularity bias can occur; hybrid mitigates with diversification.
- CBF relies on category only (no text embeddings/TF-IDF).

## Improvement Roadmap

- Item-item similarity with cosine/Jaccard and popularity normalization.
- Textual features for CBF (TF-IDF, BM25, or sentence embeddings).
- Weighted hybrid scoring (e.g., combine CF/CBF/new features via tunable weights).
- Add time decay and action weights; offline evaluation (Precision@k, Recall@k, NDCG).
- Caching and sparse matrices for scalability.
- UI: show source badges and ranks in the recommendation cards.

## Real-World Mapping

Large platforms use multi-stage recommenders (candidate generation + learned rankers), online learning/bandits, and safety/diversity constraints. This project demonstrates core ideas in a lightweight, explainable stack.
