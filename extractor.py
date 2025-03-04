#@title installazione del modulo PRAW
!pip install praw

#@title creazione dell'istanza di oggetto "reddit"
import praw

reddit = praw.Reddit(
    client_id="IL_TUO_CLIENT_ID",
    client_secret="IL_TUO_CLIENT_SECRET",
    user_agent="Scraper_GG"
    check_for_async=False
)

# 🔹 Parole chiave nei TITOLI dei post
title_keywords = ["videogiochi", "gaming", "console", "videogioco", "gamergate", "gamergirl", "gamer girl"]

# 🔹 Parole chiave nei COMMENTI (basta una)
comment_keywords = ["donna", "donne", "femminile", "gamer girl", "gamergirl", "ragazze", "ragazza"]

# 🔹 Cerca i post con le parole chiave nel titolo
subreddit = reddit.subreddit("italy")
filtered_posts = []

for keyword in title_keywords:
    posts = subreddit.search(keyword, sort="new", time_filter="all", limit=100)  # Puoi aumentare il limite
    filtered_posts.extend(posts)

# 🔹 Rimuove duplicati basati sull'ID del post
filtered_posts = list({post.id: post for post in filtered_posts}.values())

print(f"📌 Trovati {len(filtered_posts)} post con keyword nei titoli")
# 🔹 Funzione per controllare se un commento ha **almeno una parola chiave**
def contains_exact_keywords(text, keywords):
    # Usa il delimitatore di parola (\b) per assicurarsi che la parola sia separata (non una parte di altra parola)
    return any(re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) for word in keywords)

# 🔹 Estrai **tutti i commenti**, filtrando quelli che contengono **almeno una parola chiave**
comments_data = []

for post in filtered_posts:
    print(f"📥 Estraendo commenti da: {post.title[:50]}...")
    post.comments.replace_more(limit=None)  # Prende TUTTI i commenti, anche quelli annidati
    for comment in post.comments.list():
        if contains_exact_keywords(comment.body, comment_keywords):
            comments_data.append({
                "post_title": post.title,
                "comment_author": comment.author.name if comment.author else "Deleted",
                "comment_text": comment.body,
                "comment_score": comment.score
            })
    
    time.sleep(2)  # 🔹 Evita di sovraccaricare l'API di Reddit
# 🔹 Converti i dati in un DataFrame e salvali
df_comments = pd.DataFrame(comments_data)
df_comments.to_csv("reddit_videogiochi_comments_filtered.csv", index=False, encoding="utf-8")
print("✅ Dati esportati con successo in 'reddit_videogiochi_comments_full.csv'")
