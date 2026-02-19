import json
import random
import os

random.seed(42)

movies_file = 'data/movie_massive_ratings.json'
with open(movies_file, 'r', encoding='utf-8') as f:
    movies = json.load(f)

movie_list = [
    {"item_id": m["item_id"], "title": m["title"]} for m in movies
]

num_users = 2800

user_ratings = {}

user_tag_options = [
    None,
    ["gen Z", "male"],
    ["gen Z", "female"],
    ["millennial", "male"],
    ["millennial", "female"],
    ["boomer", "male"],
    ["boomer", "female"],
    ["student"],
    ["parent"],
    ["gamer"],
    []
]

comment_options = [
    None,
    "Loved it!",
    "Boring plot",
    "Great acting!",
    "Too long",
    "Must watch",
    "Not my taste",
    "Amazing visuals",
    "Predictable ending",
    "Very emotional",
    "Super fun!"
]

for i in range(1, num_users + 1):
    user_id = f"id{i:04d}"
    num_movies = random.randint(1, 8)
    selected_movies = random.sample(movie_list, num_movies)
    scores = []
    for m in selected_movies:
        score = round(random.uniform(0.5, 5.0), 1)
        comment = random.choice(comment_options)
        scores.append({
            "item_id": m["item_id"],
            "title": m["title"],
            "score": score,
            "comment": comment
        })
    tags = random.choice(user_tag_options)
    user_ratings[user_id] = {
        "tags": tags,
        "scores": scores
    }

output_file = 'data/user_massive_ratings.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(user_ratings, f, indent=2)

size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"File size: {size_mb:.2f} MB")
print(f"Generated {num_users} users with total {sum(len(u['scores']) for u in user_ratings.values())} ratings in {output_file}")
