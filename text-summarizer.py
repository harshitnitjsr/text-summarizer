git push -u origin main!pip install rouge
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import re
from rouge import Rouge  # Install with: pip install rouge

def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.read()
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', filedata)

    cleaned_sentences = []
    for sentence in sentences:
        cleaned_sentence = re.sub("[^a-zA-Z]", " ", sentence)
        cleaned_sentences.append(cleaned_sentence.split())

    cleaned_sentences.pop() 

    return cleaned_sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def generate_summary(file_name, reference_summary):
    nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text and split it
    sentences = read_article(file_name)

    # Step 2 - Generate Similarity Matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in the similarity matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    top_n = int((len(sentences)) * 0.6)
    for i in range(top_n):
        summarize_text.append(" ".join(sentences[i]))

    # Step 5 - Output the summarized text
    generated_summary = ". ".join(summarize_text)

    # Evaluate ROUGE scores
    rouge = Rouge()
    rouge_scores = rouge.get_scores(generated_summary, reference_summary)

    print("Generated Summary:\n", generated_summary)
    print("\nROUGE Scores:", rouge_scores)

# Example usage
reference_summary = '''In the dynamic realm of finance and technology, cryptocurrency has emerged as a transformative force. This digital revolution, initiated by Satoshi Nakamoto's groundbreaking whitepaper in 2008, introduced the concept of a decentralized, transparent, and trustless currency. Bitcoin, the inaugural cryptocurrency, marked its inception in 2009 with the mining of its first blockchain block.

At the core of cryptocurrency lies blockchain technology, a decentralized ledger ensuring transparency, security, and immutability. Bitcoin, employing a proof-of-work consensus mechanism, and Ethereum, introducing smart contracts, have become prominent figures in the cryptocurrency space.

The cryptocurrency market, characterized by exponential growth, features thousands of digital assets. Bitcoin, often hailed as digital gold, boasts a finite supply and store of value. Altcoins, including privacy coins, utility tokens, and stablecoins, contribute to the diverse functionalities of this evolving market.

Cryptocurrency's impact on traditional finance is evident in its disruption of conventional systems. Facilitating cross-border transactions without intermediaries, cryptocurrencies offer speed and cost-effectiveness. While Central Bank Digital Currencies (CBDCs) and blockchain integration by financial institutions signal acceptance, regulatory challenges persist, addressing concerns of compliance and illicit activities.

Decentralized Finance (DeFi) further expands cryptocurrency's reach, recreating traditional financial services on blockchain networks. Despite security vulnerabilities and regulatory uncertainties, DeFi platforms offer users greater control over their financial activities through smart contracts.

The cryptocurrency space grapples with challenges such as price volatility, regulatory ambiguity, and security risks. Environmental concerns regarding the energy consumption of proof-of-work cryptocurrencies add to the discourse.

Looking ahead, the future of cryptocurrency involves trends like CBDC development, blockchain innovations, and the rise of Non-Fungible Tokens (NFTs). These unique digital assets, verifying authenticity and ownership, find applications in art, gaming, and entertainment.

In conclusion, cryptocurrency, born from a quest for financial autonomy, reshapes global economies. Technological foundations, market dynamics, and ongoing innovations underscore its potential. While challenges persist, cryptocurrency's trajectory promises a decentralized and inclusive financial landscape, marking the next chapter in the digital revolution.'''
generate_summary("/kaggle/input/crypto1/msft.txt", reference_summary)
