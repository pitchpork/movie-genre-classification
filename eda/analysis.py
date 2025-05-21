import pandas as pd
import matplotlib.pyplot as plt

# Load the training file
df = pd.read_csv(
    '../data/raw/train_data.txt',
    sep=' ::: ',
    engine='python',
    names=['ID', 'Title', 'Genre', 'Description'],
    header=None,
    encoding='utf-8'
)

# Check structure
print(df.shape)
print(df.head())

# Genre distribution chart
genre_counts = df['Genre'].value_counts()
plt.figure(figsize=(12, 6))
genre_counts.plot(kind='bar', color='steelblue', edgecolor='black')

plt.title("Distribution of Genre Labels", fontsize=14)
plt.xlabel("Genre", fontsize=12)
plt.ylabel("Number of Films", fontsize=12)
plt.xticks(rotation=75, ha='right', fontsize=9)
plt.yticks(fontsize=10)
plt.tight_layout()


plt.savefig("../figures/genre_distribution.png", dpi=300, bbox_inches='tight')

# Plot summary length distribution
df['WordCount'] = df['Description'].str.split().apply(len)
clipped = df[df['WordCount'] < 400]

plt.figure(figsize=(8, 4.5))
plt.hist(clipped['WordCount'], bins=30, color='steelblue', edgecolor='black')
plt.title("Distribution of Plot Summary Lengths (in Words)")
plt.xlabel("Number of Words")
plt.ylabel("Number of Movies")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()

plt.savefig("../figures/plot_length_distribution.png", dpi=300)
print("Genre label distribution and plot summary length distribution charts saved to /figures/")
plt.close()