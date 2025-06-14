AI Plagiarism Checker
A Flask-based web application that allows users to check text for plagiarism against a pre-trained machine learning model and detect potential similarities with web sources. It also includes a local database to keep a history of all checks performed.

Table of Contents
About the Project
Features
Technologies Used
Setup and Installation
Usage
Project Structure
Future Enhancements
License
About the Project
This project aims to provide a simple yet effective tool for checking text originality. It leverages a machine learning model (Logistic Regression with TF-IDF) to determine the likelihood of plagiarism based on a provided dataset and integrates with Google Custom Search API to find potential online sources. All checks are logged into a local SQLite database, offering a convenient history review.

Note: The current AI detection logic and web search integration use placeholder functions for simplicity. For robust plagiarism detection and AI-generated text identification, you would integrate more sophisticated models and potentially advanced search capabilities.

Features
Plagiarism Detection: Compares input text against a trained dataset to provide a similarity score.
Web Source Detection: Utilizes Google Custom Search API to identify potential online sources for the input text.
Interactive UI: A clean and responsive web interface built with HTML and Tailwind CSS.
Local History Database: Stores all plagiarism checks (input text, results, scores, sources) in a local SQLite database.
Real-time History Display: Displays past checks directly on the web page, updated automatically.
User-Friendly Design: Intuitive interface with clear results visualization.
Technologies Used
Backend:

Python: Programming language.
Flask: Web framework.
Flask-SQLAlchemy: ORM for database interactions.
SQLite: Lightweight, file-based SQL database for history storage.
NLTK: Natural Language Toolkit for text preprocessing (stopwords, tokenization).
Scikit-learn: For machine learning model (Logistic Regression) and text vectorization (TF-IDF).
Pandas: For data handling and CSV operations.
Joblib: For saving and loading the trained ML model.
Requests: For making HTTP requests to external APIs (Google Custom Search).
Frontend:

HTML5: Structure of the web page.
CSS3 (Tailwind CSS): Styling and responsive design.
JavaScript (ES6+): For interactive elements, fetching data from backend, and dynamic UI updates.
Setup and Installation
Follow these steps to get the project up and running on your local machine.

Prerequisites
Python 3.8+
pip (Python package installer)
