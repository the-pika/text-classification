# text-classification
Text Processing and Analysis

## ✨ Features

The project demonstrates the following core NLP tasks:

1. **Named Entity Recognition (NER)**  
   - Tokenization, Part-of-Speech tagging, and Named Entity Recognition using spaCy.

2. **N-grams (Bigrams & Trigrams)**  
   - Extracts bigrams and trigrams from input text using NLTK.

3. **Sentiment Analysis (VADER)**  
   - Performs sentiment analysis on individual tokens and full sentences.  
   - Classifies text as **positive**, **negative**, or **neutral** based on polarity scores.

4. **Text Summarization from Wikipedia**  
   - Scrapes the *Artificial Intelligence* article from Wikipedia.  
   - Cleans and preprocesses the text.  
   - Scores and ranks sentences using frequency-based summarization.  
   - Produces a short summary vs. original text.

5. **Language Detection**  
   - Detects the language of given phrases and prints probabilities using `langdetect`.

---

## 📂 Repository Structure

    text-classification/
      ├── NLP_Assignment_KNU.py # Main Python script with all problems
      ├── requirements.txt # Dependencies
      └── README.md # Project description (this file)

Dependencies are listed in requirements.txt.
Make sure to also install the spaCy English model:
  python -m spacy download en_core_web_sm


## Notes and Limitations
- Summarization is frequency-based and extractive, not abstractive.
- Wikipedia scraping depends on the page structure; if it changes, parsing may need adjustments.
- Language detection is probabilistic and may be inaccurate for very short texts.


