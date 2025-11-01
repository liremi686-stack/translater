import re
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class PatternAnalyzer:
    def __init__(self):
        self.pattern_frequency = Counter()
        self.word_mappings = {}
    
    def extract_parallel_texts(self, mixed_text: str):
        """Извлечение параллельных текстов из смешанного документа"""
        # Поиск английских цитат в кавычках
        english_quotes = re.findall(r'"([^"]*)"', mixed_text)
        
        # Поиск соответствующих alien текстов (предложения перед кавычками)
        alien_sentences = []
        sentences = re.split(r'[.!?]', mixed_text)
        
        for i, sentence in enumerate(sentences):
            if '"' in sentence and i > 0:
                alien_sentences.append(sentences[i-1].strip())
        
        return list(zip(alien_sentences, english_quotes))
    
    def analyze_character_patterns(self, alien_text: str, english_text: str):
        """Анализ паттернов замены символов"""
        alien_words = re.findall(r'\b\w+\b', alien_text.lower())
        english_words = re.findall(r'\b\w+\b', english_text.lower())
        
        for a_word, e_word in zip(alien_words, english_words):
            # Анализ замен символов
            for a_char, e_char in zip(a_word, e_word):
                if a_char != e_char:
                    self.pattern_frequency[(a_char, e_char)] += 1
    
    def find_common_prefixes_suffixes(self, word_pairs):
        """Поиск общих префиксов и суффиксов"""
        prefixes = Counter()
        suffixes = Counter()
        
        for alien_word, english_word in word_pairs:
            # Анализ префиксов
            for i in range(1, min(4, len(alien_word), len(english_word))):
                if alien_word[:i] != english_word[:i]:
                    prefixes[alien_word[:i]] += 1
            
            # Анализ суффиксов
            for i in range(1, min(4, len(alien_word), len(english_word))):
                if alien_word[-i:] != english_word[-i:]:
                    suffixes[alien_word[-i:]] += 1
        
        return prefixes.most_common(10), suffixes.most_common(10)

class AdvancedAlienLanguageDecoder:
    def __init__(self):
        self.patterns = self._analyze_patterns()
        self.dictionary = self._build_extended_dictionary()
        self.context_rules = self._build_context_rules()
        self.ml_model = None
        self.vectorizer = None
        self.training_data = []
        
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
            'sy ': 'is ',
            'ta ': 'it ',
            'e ': 'a ',
            
            # Суффиксы
            'ämi': 'y',
            'ani': 'y', 
            'ä': 'a',
            'yṅ': 'ing',
            'w': 'o',
            'ṅ': 'ng',
            'ḋ': 'd',
            'ẗ': 't',
            'lyṅ': 'ling',
            'iẗ': 'it',
            
            # Грамматические конструкции
            'bäsevä': 'develop',
            'twrdoga': 'research',
            'dwrec': 'wave',
            'aslyṅ': 'this',
            'nytwnyḋ': 'energy',
        }
    
    def _build_extended_dictionary(self):
        """Расширенный словарь на основе параллельных текстов"""
        return {
            # Основные слова из текста
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
            'magua': 'development',
            'väpädä': 'applications',
            'topwbryd': 'imaging',
            'tigunyḋ': 'used',
            'brydcape': 'for',
            'dwrgaso': 'transmission',
            'maegulw': 'narrow channels',
            'vädufä': 'enhanced',
            'mofwlyṅ': 'metallic',
            'maebapu': 'nanowires',
            'lyṅrole': 'and',
            'ämege': 'engineering',
            'norr': 'subwavelength',
            'aġmosi': 'which',
            'kolovyr': 'this',
            'twrgurw': 'framework',
            'aigni': 'and',
            'älury': 'project',
            'aġgefu': 'named',
            'nyḋkeba': 'plasma',
            'eynyt': 'structures',
            'cyfny': 'stable',
            'aigfw': 'spherically',
            'änserisa': 'symmetric',
            'papiä': 'plasma',
            'aiggogw': 'really',
            'seld': 'plasmoids',
            'äcybo': 'which',
            'vert': 'but',
            'eynse': 'are',
            'aṅgena': 'dynamically',
            'fense': 'compared',
            'aġnubi': 'any',
            'decusy': 'other',
            'addwr': 'unstructured',
            'dobueyn': 'formations',
            'aġcefa': 'since',
            'aṅbycy': 'the latter',
            'ämydu': 'lose',
            'äfalo': 'their',
            'täroch': 'energy',
            'eynri': 'and',
            'aġkoki': 'recombine',
            'seletwr': 'with',
            'papiä': 'plasma',
            'bryddiba': 'formations',
            'ken': 'as',
            'aynic': 'far',
            'adal': 'know',
            'akur': 'tests',
            'dwrgily': 'were',
            'äcwda': 'always',
            'dwreb': 'conducted',
            'fense': 'at',
            'aġmunu': 'denied',
            
            # Технические термины (остаются без изменений)
            'milorbs': 'MilOrbs',
            'psv': 'PSV',
            'halo': 'HALO',
            'sienna': 'Sienna',
            'denied': 'DENIED',
        }
    
    def _build_context_rules(self):
        """Правила контекстных замен"""
        return {
            'bäsevä': {
                'after': ['twrdoga'], 'form': 'developing',
                'after': ['mugoaṅ'], 'form': 'developed'
            },
            'twrdoga': {
                'after': ['en'], 'form': 'researching',
                'after': ['ans'], 'form': 'research'
            },
            'aṅbubo': {
                'before': ['nyḋrwpw'], 'form': 'which',
                'before': ['eynse'], 'form': 'that'
            }
        }
    
    def train_ml_model(self, training_pairs: List[Tuple[str, str]]):
        """Обучение модели машинного обучения на параллельных текстах"""
        if not training_pairs:
            # Используем известные пары из текста для обучения
            training_pairs = [
                ("Ans en änselämi twrdoga", "Both companies were researching"),
                ("trelbe ans twrdoga ag änselämi checesi", "were both research and development"),
                ("MilOrbs aig PSV mugoaṅ bäsevä", "MilOrbs and PSV were developed"),
            ]
        
        alien_texts = [pair[0] for pair in training_pairs]
        english_texts = [pair[1] for pair in training_pairs]
        
        # Векторизация текста
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))
        X = self.vectorizer.fit_transform(alien_texts)
        
        # Простая модель для демонстрации
        self.ml_model = MultinomialNB()
        # Для простоты будем предсказывать первые 10 символов английского текста
        y = [text[:10] for text in english_texts]
        self.ml_model.fit(X, y)
        
        self.training_data = training_pairs
    
    def ml_decode_word(self, word: str, context: List[str] = None) -> str:
        """Декодирование слова с использованием ML"""
        if not self.ml_model:
            return self.decode_word(word, context)
        
        try:
            word_vec = self.vectorizer.transform([word])
            prediction = self.ml_model.predict(word_vec)[0]
            return prediction
        except:
            return self.decode_word(word, context)
    
    def apply_context_rules(self, word: str, prev_word: str = None, next_word: str = None) -> str:
        """Применение контекстных правил"""
        if word in self.context_rules:
            rules = self.context_rules[word]
            if prev_word and 'before' in rules and prev_word in rules['before']:
                return rules['form']
            if next_word and 'after' in rules and next_word in rules['after']:
                return rules['form']
        return word
    
    def decode_word(self, word: str, context: List[str] = None) -> str:
        """Улучшенное декодирование слова с учетом контекста"""
        original_word = word
        
        # Проверка в словаре
        if word.lower() in self.dictionary:
            decoded = self.dictionary[word.lower()]
            
            # Применение контекстных правил
            if context and len(context) > 1:
                prev_word = context[-2] if len(context) > 1 else None
                next_word = context[0] if context else None
                decoded = self.apply_context_rules(decoded, prev_word, next_word)
            
            return decoded
        
        # Применение паттернов замены
        decoded = word
        for pattern, replacement in self.patterns.items():
            decoded = decoded.replace(pattern, replacement)
        
        # Если слово сильно изменилось, проверяем его снова в словаре
        if decoded != word and decoded.lower() in self.dictionary:
            return self.dictionary[decoded.lower()]
        
        # Удаление лишних пробелов
        decoded = re.sub(r'\s+', ' ', decoded).strip()
        
        return decoded if decoded != word else f"[{word}]"  # Помечаем непереведенные слова
    
    def advanced_sentence_split(self, text: str) -> List[str]:
        """Улучшенное разбиение на предложения"""
        # Учитываем различные знаки препинания
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def decode_text(self, text: str, use_ml: bool = False) -> str:
        """Основной метод декодирования текста"""
        sentences = self.advanced_sentence_split(text)
        decoded_sentences = []
        
        for sentence in sentences:
            if not sentence:
                continue
                
            words = sentence.split()
            decoded_words = []
            
            for i, word in enumerate(words):
                # Очистка слова от знаков препинания
                clean_word = re.sub(r'[^\wäöüßṅḋẗ]', '', word)
                punctuation = word[len(clean_word):]
                
                # Контекст для текущего слова
                context = words[max(0, i-2):min(len(words), i+3)]
                
                # Декодирование
                if use_ml and self.ml_model:
                    decoded_word = self.ml_decode_word(clean_word, context)
                else:
                    decoded_word = self.decode_word(clean_word, context)
                
                decoded_words.append(decoded_word + punctuation)
            
            decoded_sentence = ' '.join(decoded_words)
            decoded_sentences.append(decoded_sentence.capitalize())
        
        result = '. '.join(decoded_sentences)
        return result + '.' if not result.endswith('.') else result
    
    def analyze_language_patterns(self, sample_text: str, english_translation: str):
        """Анализ паттернов языка на основе параллельных текстов"""
        alien_sentences = self.advanced_sentence_split(sample_text)
        english_sentences = self.advanced_sentence_split(english_translation)
        
        patterns_found = defaultdict(Counter)
        
        for alien_sent, eng_sent in zip(alien_sentences, english_sentences):
            alien_words = alien_sent.split()
            eng_words = eng_sent.split()
            
            # Простой анализ соответствий
            for a_word, e_word in zip(alien_words, eng_words):
                clean_a_word = re.sub(r'[^\wäöüßṅḋẗ]', '', a_word)
                patterns_found[clean_a_word][e_word] += 1
        
        # Обновление словаря на основе анализа
        for alien_word, translations in patterns_found.items():
            if alien_word not in self.dictionary:
                most_common = translations.most_common(1)
                if most_common:
                    self.dictionary[alien_word] = most_common[0][0]
    
    def save_dictionary(self, filename: str):
        """Сохранение словаря в файл"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.dictionary, f, ensure_ascii=False, indent=2)
    
    def load_dictionary(self, filename: str):
        """Загрузка словаря из файла"""
        with open(filename, 'r', encoding='utf-8') as f:
            self.dictionary.update(json.load(f))

# Пример использования с расширенными функциями
def main():
    decoder = AdvancedAlienLanguageDecoder()
    
    # Тренировка модели на известных данных
    training_data = [
        ("Ans en änselämi twrdoga", "Both companies were researching"),
        ("trelbe ans twrdoga ag änselämi checesi", "were both research and development"),
        ("MilOrbs aig PSV mugoaṅ bäsevä", "MilOrbs and PSV were developed"),
        ("aġic ag e twrdoga dy myrcono sigylyṅ", "of a research of subwavelength metallic nanowires"),
    ]
    decoder.train_ml_model(training_data)
    
    # Текст для декодирования
    encrypted_text = """
    Ans en änselämi twrdoga, trelbe ans twrdoga ag änselämi checesi. 
    MilOrbs aig PSV mugoaṅ bäsevä aġic ag e twrdoga dy myrcono sigylyṅ. 
    aṅcwpu aslyṅ bäsevä änselani fy disäsy nyḋrwpw, aṅbubo ta fy mystical reconnaisance vehicle, 
    vermulo duṅtyne aig er ämoby, aṅbacy aġdeka ag cerylyṅ sy dwrciga re aġryty dwrec.
    """
    
    print("=== Базовое декодирование ===")
    basic_result = decoder.decode_text(encrypted_text, use_ml=False)
    print(basic_result)
    
    print("\n=== Декодирование с ML ===")
    ml_result = decoder.decode_text(encrypted_text, use_ml=True)
    print(ml_result)
    
    # Анализ и сохранение словаря
    sample_alien_text = """
    Ans en änselämi twrdoga, trelbe ans twrdoga ag änselämi checesi.
    """
    sample_english_text = """
    Both companies were researching, were both research and development.
    """
    
    decoder.analyze_language_patterns(sample_alien_text, sample_english_text)
    decoder.save_dictionary('alien_dictionary.json')
    
    # Статистика декодирования
    words_to_check = ['ans', 'en', 'änselämi', 'twrdoga', 'trelbe']
    print("\n=== Статистика декодирования ===")
    for word in words_to_check:
        decoded = decoder.decode_word(word)
        print(f"{word} -> {decoded}")

if __name__ == "__main__":
    main()