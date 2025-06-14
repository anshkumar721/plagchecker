import nltk  # For natural language processing
from sklearn.metrics import accuracy_score, classification_report
# nltk.download("popular")  # Download resources like stop words (run once, keep commented if already downloaded)
from sklearn.metrics.pairwise import cosine_similarity  # For comparing text similarity
import pandas as pd  # For working with data in tables
import string  # For punctuation removal
from nltk.corpus import stopwords  # For removing common words
import joblib  # For saving and loading the machine learning model
from sklearn.linear_model import LogisticRegression  # The machine learning model
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to numerical data

# Download NLTK resources (run once)
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK popular resources...")
    nltk.download("popular")
    print("NLTK popular resources downloaded.")

# --- Data Loading and Preprocessing ---

# Load the dataset containing source and plagiarized texts, and labels
try:
    # --- IMPORTANT: CHOOSE ONE OF THE FOLLOWING pd.read_csv OPTIONS BASED ON YOUR CSV FILE ---

    # OPTION 1 (Most Common): If your CSV has a header row and uses commas as separators
    data = pd.read_csv(
        "article50.csv",  # Using the filename you provided in the error
        sep=',',  # Explicitly state comma as separator
        quotechar='"',  # Explicitly state double quote as quote character (for fields with commas)
        header=0,  # Explicitly state that the first row is the header (row index 0)
        on_bad_lines='warn'  # Warn about bad lines, don't fail immediately (for pandas >= 1.3)
    )

    # OPTION 2: If your CSV DOES NOT have a header row, and you need to provide column names
    # Uncomment this block and comment out OPTION 1 if your CSV has no header
    # data = pd.read_csv(
    #     "plagrism_text.csv",
    #     sep=',',
    #     quotechar='"',
    #     header=None,  # Tell pandas there is no header row
    #     names=['source_text', 'plagiarized_text', 'label'], # Provide the column names explicitly
    #     on_bad_lines='warn'
    # )

    # OPTION 3: If your CSV uses a different delimiter (e.g., semicolon, tab)
    # Uncomment this block and comment out OPTION 1 if your CSV uses a different separator
    # data = pd.read_csv(
    #     "plagrism_text.csv",
    #     sep=';',             # Change to ';', '\t', or whatever your delimiter is
    #     quotechar='"',
    #     header=0,
    #     on_bad_lines='warn'
    # )

    # --- After loading, it's good practice to clean up column names ---
    # This removes leading/trailing whitespace from column names and makes them lowercase
    data.columns = data.columns.str.strip().str.lower()

    # --- IMPORTANT DEBUG STEP ---
    print("\n--- Columns loaded by pandas (after cleaning) ---")
    print(data.columns)
    print("------------------------------------------------\n")
    # --- END DEBUG STEP ---

    # Ensure text columns are strings and handle potential NaNs before preprocessing
    data["source_text"] = data["source_text"].fillna('').astype(str)
    data["plagiarized_text"] = data["plagiarized_text"].fillna('').astype(str)
    print("plagrism_text.csv loaded successfully.")

except FileNotFoundError:
    print("Error: plagrism_text.csv not found. Please ensure it's in the same directory as this script.")
    exit()
except pd.errors.ParserError as e:
    print(f"Error loading plagrism_text.csv: {e}")
    print("This usually means there's an inconsistency in the number of columns in your CSV file.")
    print("Please open the file in a plain text editor, go to the mentioned line, and check for:")
    print("  - Unescaped commas within text fields (e.g., 'text with, a comma' should be '\"text with, a comma\"').")
    print("  - Incorrect number of commas/fields in any row.")
    exit()
except KeyError as e:
    print(f"Error: A required column was not found. Details: {e}")
    print(
        "This means pandas could not find 'source_text' or 'plagiarized_text' or 'label' as column names after loading.")
    print("Please double-check the header row of your CSV for exact spelling, casing, and no extra spaces.")
    print("Also, consider trying the other `pd.read_csv` options in the code (OPTION 2 or OPTION 3).")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading plagrism_text.csv: {e}")
    exit()


# Function to preprocess text: remove punctuation, lowercase, remove stop words
def preprocess_text(text):
    text = str(text)  # Ensure text is always a string
    # Remove punctuation using a translation table
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stop words (common words like "the", "is", "and" that don't carry much meaning)
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text


# Preprocess the text data in the 'source_text' and 'plagiarized_text' columns
data["source_text"] = data["source_text"].apply(preprocess_text)
data["plagiarized_text"] = data["plagiarized_text"].apply(preprocess_text)
print("Text data preprocessed.")

# --- Feature Extraction and Model Training ---

# Combine source and plagiarized texts for TF-IDF feature extraction
# This creates the feature set (X) for training the model
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to prevent memory issues
X = tfidf_vectorizer.fit_transform(data["source_text"] + " " + data["plagiarized_text"])
print("TF-IDF vectorization complete.")

# Extract the labels (0 for not plagiarized, 1 for plagiarized)
y = data["label"]

# Train a Logistic Regression model to detect plagiarism
model = LogisticRegression(max_iter=1000,
                           solver='liblinear')  # Increased max_iter for convergence, 'liblinear' for smaller datasets
model.fit(X, y)  # Train the model on the text data and labels
print("Logistic Regression model trained.")

# --- Model Evaluation ---

# Split the data into training and testing sets to evaluate model performance
# Note: For this script, X is the full dataset used for training,
# but we split it here to demonstrate evaluation on unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy of the model
classification_rep = classification_report(y_test, y_pred)  # Get a detailed classification report

# Print the model's performance metrics
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)

# --- Save Model and Vectorizer ---

# Save the trained model AND the tfidf_vectorizer to a file
# This is crucial because app.py needs both to make predictions on new text
try:
    joblib.dump((model, tfidf_vectorizer), 'plagiarism_model.pkl')
    print("\nModel and TF-IDF vectorizer saved to 'plagiarism_model.pkl'.")
except Exception as e:
    print(f"Error saving model and vectorizer: {e}")

# --- Example Usage (for demonstration in this script only) ---
print("\n--- Example Plagiarism Detection ---")
# Load the saved model and vectorizer (demonstrates loading what was just saved)
try:
    loaded_model, loaded_vectorizer = joblib.load('plagiarism_model.pkl')
    print("Model and vectorizer successfully loaded for example usage.")
except Exception as e:
    print(f"Error loading saved model for example: {e}")
    loaded_model = None
    loaded_vectorizer = None

if loaded_model and loaded_vectorizer:
    # New text for plagiarism detection
    new_text = input("Enter a sample text to check for plagiarism: ")

    # Preprocess the new text (apply the same steps as during training)
    preprocessed_new_text = preprocess_text(new_text)

    # Convert the preprocessed text into TF-IDF vectors using the SAVED vectorizer
    new_text_vector = loaded_vectorizer.transform([preprocessed_new_text])

    # Make predictions using the loaded model
    prediction = loaded_model.predict(new_text_vector)

    # Calculate cosine similarity between new text and the training data (X)
    # Use X (the full dataset's vectorized form) for similarity comparison
    cosine_similarity_score = cosine_similarity(new_text_vector, X).max()

    # Interpret the prediction and similarity score
    if prediction[0] == 0:
        print("The text is not plagiarized (based on model prediction).")
    else:
        print(
            f"The text is plagiarized (based on model prediction) with a similarity score of {cosine_similarity_score * 100:.2f}%.")
else:
    print("Cannot run example usage as model/vectorizer could not be loaded.")