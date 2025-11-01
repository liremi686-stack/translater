import re
from collections import defaultdict

class AlienLanguageDecoder:
    def __init__(self):
        self.patterns = self._analyze_patterns()
        self.dictionary = self._build_dictionary()
        
    def _analyze_patterns(self):
        return {
            # Префиксы
            'aġ': 'the ',
            'aṅ': 'the ',
            'än': '',
            'nyḋ': '',
            'lyṅ': 'ly ',
            'er ': 'the ',
            're ': 'for ',
            'fy ': 'in ',
            'dy ': 'of ',
            'ag ': 'and ',
            'aig ': 'and ',
            'vert ': 'but ',
            'ken ': 'as ',
            
            # Суффиксы
            'ämi': 'y',
            'ani': 'y', 
            'ä': 'a',
            'yṅ': 'ing',
            'w': 'o',
            'ṅ': 'ng',
            'ḋ': 'd',
            'ẗ': 't',
            
            # Стемы слов
            'twrdoga': 'researching',
            'bäsevä': 'development',
            'aslyṅ': 'this',
            'nytwnyḋ': 'energy',
            'dwrec': 'wave',
            'muguä': 'development',
        }
    
    def _build_dictionary(self):
        return {
            'ans': 'both',
            'en': 'companies',
            'trelbe': 'were',
            'checesi': 'development', 
            'mugoaṅ': 'developed',
            'aġic': 'of',
            'myrcono': 'subwavelength',
            'sigylyṅ': 'metallic nanowires',
            'aṅcwpu': 'enabled',
            'disäsy': 'epsilon-near-zero',
            'nyḋrwpw': 'metamaterials',
            'aṅbubo': 'which',
            'vermulo': 'used',
            'duṅtyne': 'then',
            'ämoby': 'environment',
            'aṅbacy': 'applications',
            'aġdeka': 'such as',
            'cerylyṅ': 'enhanced transmission',
            'dwrciga': 'cloaking',
            'aġryty': 'energy squeezing',
            'defter': 'this',
            'teel': 'also',
            'levä': 'led',
            'tord': 'to',
            'äbifo': 'the possibility',
            'lyṅtibe': 'of creating',
            'änselidu': 'objects',
            'cyfgidy': 'which are',
            'aigsypy': 'equivalent to',
            'lyṅed': 'orbs',
            'liluvyr': 'that',
            'äfalu': 'can',
            'aigdabe': 'penetrate',
            'tyrit': 'walls',
        }
    
    def decode_word(self, word):
        # Проверка в словаре
        if word in self.dictionary:
            return self.dictionary[word]
            
        # Применение паттернов
        decoded = word
        for pattern, replacement in self.patterns.items():
            decoded = decoded.replace(pattern, replacement)
            
        return decoded
    
    def decode_text(self, text):
        sentences = re.split(r'[.!?]', text)
        decoded_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            words = sentence.strip().split()
            decoded_words = []
            
            for word in words:
                # Обработка слов с запятыми и другими знаками
                clean_word = re.sub(r'[^\wäöüßṅḋẗ]', '', word)
                punctuation = word[len(clean_word):]
                
                decoded_word = self.decode_word(clean_word)
                decoded_words.append(decoded_word + punctuation)
            
            decoded_sentence = ' '.join(decoded_words)
            decoded_sentences.append(decoded_sentence.capitalize())
        
        return '. '.join(decoded_sentences) + '.'

# Использование
decoder = AlienLanguageDecoder()

# Пример дешифрования
encrypted_text = """
Ans en änselämi twrdoga, trelbe ans twrdoga ag änselämi checesi. 
MilOrbs aig PSV mugoaṅ bäsevä aġic ag e twrdoga dy myrcono sigylyṅ.
"""

decoded_text = decoder.decode_text(encrypted_text)
print(decoded_text)