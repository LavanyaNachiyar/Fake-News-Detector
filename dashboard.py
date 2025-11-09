import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re

# Load the dataset
df = pd.read_csv('news.csv')

# Basic dataset info
print("=== FAKE NEWS DETECTOR DASHBOARD ===")
print(f"Total articles: {len(df)}")
print(f"Real articles: {len(df[df['label'] == 'REAL'])}")
print(f"Fake articles: {len(df[df['label'] == 'FAKE'])}")

# Create visualizations
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Label distribution pie chart
label_counts = df['label'].value_counts()
axes[0, 0].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', 
               colors=['lightcoral', 'lightblue'])
axes[0, 0].set_title('Real vs Fake News Distribution')

# 2. Article length comparison
df['text_length'] = df['text'].str.len()
real_lengths = df[df['label'] == 'REAL']['text_length']
fake_lengths = df[df['label'] == 'FAKE']['text_length']

axes[0, 1].hist([real_lengths, fake_lengths], bins=30, alpha=0.7, 
                label=['Real', 'Fake'], color=['blue', 'red'])
axes[0, 1].set_title('Article Length Distribution')
axes[0, 1].set_xlabel('Text Length')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()

# 3. Top words in titles
all_titles = ' '.join(df['title'].dropna().astype(str))
words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles.lower())
word_freq = Counter(words)
top_words = dict(word_freq.most_common(10))

axes[1, 0].bar(top_words.keys(), top_words.values(), color='green', alpha=0.7)
axes[1, 0].set_title('Top 10 Words in Titles')
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Average title length by label
df['title_length'] = df['title'].str.len()
avg_title_length = df.groupby('label')['title_length'].mean()
axes[1, 1].bar(avg_title_length.index, avg_title_length.values, 
               color=['orange', 'purple'], alpha=0.7)
axes[1, 1].set_title('Average Title Length by Label')
axes[1, 1].set_ylabel('Average Length')

plt.tight_layout()
plt.savefig('dashboard_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Word cloud for fake news
fake_text = ' '.join(df[df['label'] == 'FAKE']['text'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(fake_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Fake News Articles')
plt.savefig('wordcloud.png', dpi=300, bbox_inches='tight')
plt.show()

# Simple statistics
print("\n=== BASIC STATISTICS ===")
print(f"Average article length (Real): {real_lengths.mean():.0f} characters")
print(f"Average article length (Fake): {fake_lengths.mean():.0f} characters")
print(f"Average title length (Real): {df[df['label'] == 'REAL']['title_length'].mean():.0f} characters")
print(f"Average title length (Fake): {df[df['label'] == 'FAKE']['title_length'].mean():.0f} characters")