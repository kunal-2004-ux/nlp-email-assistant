import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

class EmailProcessor:
    def __init__(self):
        """Initialize the NLP models and tools"""
        # Using BART large CNN for better summarization
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Using RoBERTa for better sentiment analysis
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="siebert/sentiment-roberta-large-english"
        )
        
        self.stopwords = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove very short lines (likely signatures or automated text)
        lines = text.split('\n')
        lines = [line for line in lines if len(line.split()) > 3]
        return ' '.join(lines)

    def summarize_email(self, text, max_length=130, min_length=30):
        """Generate a concise summary of the email content"""
        if not text or not text.strip():
            return "No content to summarize."
        
        try:
            # Clean the text first
            text = self.clean_text(text)
            
            # If text is very short, return it as is
            if len(text.split()) < 50:
                return text
            
            # Split text into smaller chunks to prevent model overflow
            words = text.split()
            max_chunk_size = 500  # Maximum words per chunk
            chunks = []
            
            for i in range(0, len(words), max_chunk_size):
                chunk = ' '.join(words[i:i + max_chunk_size])
                # Make sure we don't cut in the middle of a sentence
                if i + max_chunk_size < len(words):
                    last_period = chunk.rfind('.')
                    if last_period != -1:
                        chunk = chunk[:last_period + 1]
                        i = i + len(chunk.split())  # Adjust the index for the next chunk
                chunks.append(chunk)
            
            # Summarize each chunk
            summaries = []
            for chunk in chunks:
                if len(chunk.split()) > 50:  # Only summarize if chunk is long enough
                    try:
                        # Calculate dynamic lengths based on chunk size
                        chunk_length = len(chunk.split())
                        adjusted_max_length = min(max_length, max(min_length, chunk_length // 3))
                        adjusted_min_length = min(min_length, adjusted_max_length - 10)
                        
                        # Ensure the chunk is not too long for the model
                        if len(chunk.split()) > 1000:
                            chunk = ' '.join(chunk.split()[:1000])
                        
                        summary = self.summarizer(chunk, 
                                               max_length=adjusted_max_length, 
                                               min_length=adjusted_min_length, 
                                               do_sample=False,
                                               truncation=True)  # Enable truncation
                        if summary and summary[0]['summary_text']:
                            summaries.append(summary[0]['summary_text'])
                    except Exception as e:
                        print(f"Error summarizing chunk: {e}")
                        # If summarization fails, use the first sentence of the chunk
                        first_sentence = sent_tokenize(chunk)[:1]
                        if first_sentence:
                            summaries.append(first_sentence[0])
            
            # Combine summaries
            if summaries:
                final_summary = ' '.join(summaries)
                # Ensure the final summary isn't too long
                if len(final_summary.split()) > max_length:
                    final_summary = ' '.join(final_summary.split()[:max_length]) + '...'
                return final_summary
            else:
                # Fallback to first few sentences if summarization fails
                sentences = sent_tokenize(text)[:3]
                return ' '.join(sentences)
            
        except Exception as e:
            print(f"Error in summarization: {e}")
            # Fallback to first sentence if everything fails
            try:
                return sent_tokenize(text)[0]
            except:
                return text[:200] + "..."

    def analyze_sentiment(self, text):
        """Analyze the sentiment of the email content with improved accuracy"""
        try:
            # Clean text first
            text = self.clean_text(text)
            
            # For long texts, analyze multiple chunks and aggregate
            words = text.split()
            chunk_size = 500  # Maximum words per chunk
            chunks = []
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:min(i + chunk_size, len(words))])
                chunks.append(chunk)
            
            sentiments = []
            for chunk in chunks:
                try:
                    result = self.sentiment_analyzer(chunk[:512])  # Limit chunk size for model
                    sentiments.append(result[0])
                except Exception as e:
                    print(f"Error in chunk sentiment analysis: {e}")
                    continue
            
            # Aggregate results
            if not sentiments:
                return {'sentiment': 'NEUTRAL', 'confidence': 0.5}
            
            # Calculate weighted average of sentiments
            total_score = sum(s['score'] for s in sentiments)
            avg_score = total_score / len(sentiments)
            
            # Determine final sentiment
            if avg_score > 0.6:
                sentiment = 'POSITIVE'
            elif avg_score < 0.4:
                sentiment = 'NEGATIVE'
            else:
                sentiment = 'NEUTRAL'
                
            return {
                'sentiment': sentiment,
                'confidence': avg_score
            }
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {'sentiment': 'NEUTRAL', 'confidence': 0.5}

    def extract_key_points(self, text, num_points=5):
        """Extract key points from the email using improved NLP techniques"""
        try:
            # Clean text first
            text = self.clean_text(text)
            
            # Tokenize and tag parts of speech
            sentences = sent_tokenize(text)
            sentence_scores = {}
            
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                tagged = pos_tag(words)
                
                # Score based on important parts of speech
                score = 0
                for word, tag in tagged:
                    if word not in self.stopwords:
                        if tag.startswith(('NN', 'VB')):  # Nouns and verbs
                            score += 2
                        elif tag.startswith(('JJ', 'RB')):  # Adjectives and adverbs
                            score += 1
                
                # Additional scoring for sentence position
                if sentences.index(sentence) == 0:  # First sentence
                    score += 3
                elif sentences.index(sentence) == len(sentences) - 1:  # Last sentence
                    score += 2
                
                # Score based on sentence length
                if 5 < len(words) < 25:  # Prefer medium-length sentences
                    score += 2
                
                sentence_scores[sentence] = score
            
            # Get top sentences
            key_points = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_points]
            
            # Sort by original order
            key_points.sort(key=lambda x: sentences.index(x[0]))
            
            return [point[0] for point in key_points]
            
        except Exception as e:
            print(f"Error extracting key points: {e}")
            return []

    def process_email(self, email_data):
        """Process an email and return various analyses"""
        body = email_data.get('body', '')
        
        return {
            'summary': self.summarize_email(body),
            'sentiment': self.analyze_sentiment(body),
            'key_points': self.extract_key_points(body),
            'original_email': email_data
        } 