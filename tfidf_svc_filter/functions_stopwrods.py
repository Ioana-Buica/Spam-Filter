from langdetect import detect, DetectorFactory
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

LANGUAGE_NAME_MAP = {
    'ar': 'arabic',
    'az': 'azerbaijani',
    'eu': 'basque',
    'bn': 'bengali',
    'zh': 'chinese',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'fi': 'finnish',
    'fr': 'french',
    'de': 'german',
    'el': 'greek',
    'he': 'hebrew',
    'hi': 'hinglish',
    'hu': 'hungarian',
    'id': 'indonesian',
    'it': 'italian',
    'kk': 'kazakh',
    'ne': 'nepali',
    'no': 'norwegian',
    'pt': 'portuguese',
    'ro': 'romanian',
    'ru': 'russian',
    'sl': 'slovene',
    'es': 'spanish',
    'sv': 'swedish',
    'tg': 'tajik',
    'tr': 'turkish'
}

def remove_stopwords(text):
    """
    Detect the language of the text, then remove the stopwords
    if nltk has a list of stopwords for that language.
    """
    try:
        DetectorFactory.seed = 0
        try:
            lang_code = detect(text)
        except Exception as e:
            print(f"Error detecting language: {e}")
            return text  

        if lang_code not in LANGUAGE_NAME_MAP:
            return text
        else:
            lang_name = LANGUAGE_NAME_MAP.get(lang_code)
            try:
                stop_words = set(stopwords.words(lang_name))
            except OSError:
                print(f"Stopwords for language '{lang_name}' are not available.")
                return text
            except Exception as e:
                print(f"Unexpected error while fetching stopwords for language '{lang_name}': {e}")
                return text

        try:
            words = text.split()
            filtered_words = [word for word in words if word.lower() not in stop_words]

            filtered_text = " ".join(filtered_words)
            return filtered_text
        
        except Exception as e:
            print(f"Unexpected error during text processing: {e}")
            return text  

    except Exception as e:
        print(f"Unexpected error: {e}")
        return text  
