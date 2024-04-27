import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle


# Load the dataset
df = pd.read_csv('Restaurant reviews.csv', encoding='ISO-8859-1')

# Preprocess the data
# Extract necessary columns
X = df['Review']
y = df['Rating']

# Convert Rating column to numeric
y = pd.to_numeric(y, errors='coerce')  # coerce errors will turn non-numeric values into NaN

# Map ratings to classes
def map_rating_to_class(rating):
    if rating <= 2:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Positive'

y = y.apply(map_rating_to_class)

# Drop rows with NaN values in both X and y
df = df.dropna(subset=['Review', 'Rating'])

# Reassign X and y after dropping NaN values
X = df['Review']
y = df['Rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train.astype(str))  # Convert to string to handle NaNs
X_test_vectorized = vectorizer.transform(X_test.astype(str))  # Convert to string to handle NaNs

# Train the classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Evaluate the model
accuracy = classifier.score(X_test_vectorized, y_test)
print("Accuracy:", accuracy)

# Save the model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
