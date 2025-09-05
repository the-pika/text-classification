"""
Converted from Jupyter Notebook: NLP_Assignment_KNU.ipynb
This script restructures the notebook into a linear, runnable module.
Markdown cells become commented section headers; selected cell outputs are included as comments.
Run as:
  python NLP_Assignment_KNU.py
"""

from __future__ import annotations

def main():

    # ##############################################################################
    # Problem 1 -  Named Entity Recognition showing tokenization, parts of speech tagging followed by named entity recognition.
    # ##############################################################################
 
    import spacy as spacy

    nlp = spacy.load("en_core_web_sm")

    text = "Steve Jobs was an American entrepreneur and business magnate. He was the chairman, chief executive officer (CEO), and a co-founder of Apple Inc., chairman and majority shareholder of Pixar, a member of The Walt Disney Company's board of directors following its acquisition of Pixar, and the founder, chairman, and CEO of NeXT. Jobs is widely recognized as a pioneer of the microcomputer revolution of the 1970s and 1980s, along with Apple co-founder Steve Wozniak. "

    doc = nlp(text)

    for token in doc:
        print(f"{token.text}", end=" | ")    # tokenization
    
    for token in doc:
        print(f"{token.text}: {token.pos_}", end=" | ")     # part of speech tagging
    
    for ent in doc.ents:
        print(f"{ent.text}: {ent.label_}")      # named entity recognition
    

    # #############################################################################
    # Problem 2: Extract all bigrams, trigrams using ngrams of nltk library
    # #############################################################################
    
    import nltk
    from nltk.util import ngrams
    from nltk.tokenize import word_tokenize
    nltk.download('punkt')
    nltk.download('punkt_tab')
    # The input text
    text = "Machine learning is a necessary field in today's world. Data science can do wonders. Natural Language Processing is how machines understand text."
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Generate bigrams
    bigrams = list(ngrams(tokens, 2))
    
    # Generate trigrams
    trigrams = list(ngrams(tokens, 3))
    
    # Display bigrams
    print("Bigrams:")
    for bigram in bigrams:
        print(bigram)
    
    # Display trigrams
    print("\nTrigrams:")
    for trigram in trigrams:
        print(trigram)
    

    # ##################################################
    # Problem 3 - Sentiment Analysis using VADER
    # ##################################################
    
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    # Download necessary NLTK data
    nltk.download('punkt')
    
    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Input sentences
    sentences = [
        "We are happy!",
        "Today I am Happy",
        "The best life ever",
        "I am sad",
        "We are sad",
        "We are super sad",
        "We are all so sad today"
    ]
    
    # Function to analyze sentiment
    def analyze_sentiment(sentence):
        tokens = nltk.word_tokenize(sentence)
        print(f"\nSentence: '{sentence}'")
    
        # Print polarity scores for each token
        for token in tokens:
            token_scores = analyzer.polarity_scores(token)
            print(f"Token: '{token}' -> Polarity Scores: {token_scores}")
    
        # Calculate and print compound score for the sentence
        sentence_scores = analyzer.polarity_scores(sentence)
        print(f"Sentence Polarity Scores: {sentence_scores}")
    
        # Determine sentiment based on compound score
        compound = sentence_scores['compound']
        if compound >= 0.05:
            sentiment = "positive"
        elif compound < 0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
    
        print(f"Sentiment: {sentiment}")
    
    # Analyze each sentence
    for sentence in sentences:
        analyze_sentiment(sentence)
    
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    # Initialize the VADER sentiment intensity analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Define your sentence
    sentence = "VADER is a great tool for sentiment analysis!"
    
    # Get the sentiment scores for the sentence
    sentiment_scores = analyzer.polarity_scores(sentence)
    
    # Extract the compound score
    compound_score = sentiment_scores['compound']
    
    # Print the compound score
    print(f"The compound score for the sentence is: {compound_score}")
    


    # ########################################################################
    # Problem 4 - Problem 4: Text Summarization of a Wikipedia article
    # ########################################################################
    
    from bs4 import BeautifulSoup
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from collections import defaultdict
    import heapq
    import requests

    # Download NLTK data
    nltk.download('stopwords')
    nltk.download('punkt')
    '''
    def fetch_text_requests(url, timeout=30):
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        r.encoding = r.apparent_encoding  # or trust r.encoding
        return r.text
'''

    # Data Collection from Wikipedia
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()   # will raise HTTPError if 403/404/etc.
    web_content = response.text

    # Parsing the URL Content
    soup = BeautifulSoup(web_content, 'html.parser')
    paragraphs = soup.find_all('p')
    text_content = ''
    for p in paragraphs:
        text_content += p.get_text()

    # Data Clean-up
    def clean_text(text):
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\d', ' ', text)
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\b[a-zA-Z]\b', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    cleaned_text = clean_text(text_content)
    stop_words = set(stopwords.words('english'))
    cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in stop_words])

    # Tokenization
    words = word_tokenize(cleaned_text)
    sentences = sent_tokenize(text_content)

    # Calculate the Word Frequency
    word_frequencies = defaultdict(int)
    for word in words:
        word_frequencies[word] += 1
    
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / max_frequency)

    # Calculate the Weighted Frequency for Each Sentence
    sentence_scores = defaultdict(int)
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[sentence] += word_frequencies[word]

    # Create Summary
    summary_sentences = heapq.nlargest(int(len(sentence_scores) * 0.3),
                                       sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    print(summary)


    # ###########################################################
    # This is the original text content from the website.
    # ###########################################################
    #

    print ("\n\n\nOriginal Text Content from the website: \n",text_content)
 
    print("\n\n\n Cleaned Text: \n",cleaned_text)
    

    # ##############################################################################
    # Problem 5 - Language detection Using NLTK Python and print the probabilities and language name
    # ##############################################################################
    #
    
    from langdetect import detect_langs
    
    # Phrases to be detected
    phrases = [
        "Solen skinner i dag, fuglene synger, og det er sommer.",
        "Ní dhéanfaidh ach Dia breithiúnas orm.",
        "I domum et cuna matrem tuam in cochleare.",
        "Huffa, huffa meg, det finns poteter på badet. Stakkars, stakkars meg, det finns poteter på badet."
    ]
    
    # Detect languages and print probabilities
    for i, phrase in enumerate(phrases, start=1):
        print(f"Phrase {i}: {phrase}")
        detected_languages = detect_langs(phrase)
        for lang in detected_languages:
            print(f"  - Language: {lang.lang}, Probability: {lang.prob:.2f}")
        print()
    pass

if __name__ == '__main__':
    main()
