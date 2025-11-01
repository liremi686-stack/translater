import re
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Union, Optional
import os
import numpy as np
from difflib import SequenceMatcher
import spacy

class EnhancedAlienDecoder:
    def __init__(self):
        self.char_mapping = self._build_char_mapping()
        self.morpheme_patterns = self._build_morpheme_patterns()
        self.syntax_rules = self._build_syntax_rules()
        self.semantic_clusters = self._build_semantic_clusters()
        self.learned_patterns = defaultdict(list)
        self.session_memory = {}
        self.confidence_threshold = 0.7  # Снижен порог для большего покрытия
        
        # Загрузка расширенного словаря
        self.dictionary = self._load_comprehensive_dictionary()
        
        # Статистические модели
        self.word_frequencies = defaultdict(int)
        self.context_patterns = defaultdict(lambda: defaultdict(int))
        
        # Загрузка модели для семантического анализа (если доступно)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
        
    def _build_char_mapping(self):
        """Расширенное отображение символов"""
        mapping = {
            # Гласные
            'ā': 'a', 'ē': 'e', 'ī': 'i', 'ō': 'o', 'ū': 'u',
            'ä': 'a', 'ö': 'o', 'ü': 'u', 'ȧ': 'a', 'ė': 'e', 'ȯ': 'o',
            'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
            
            # Согласные
            'ç': 'c', 'ş': 's', 'ţ': 't', 'ḑ': 'd', 'ņ': 'n',
            'ķ': 'k', 'ŗ': 'r', 'ṅ': 'ng', 'ḋ': 'd', 'ẗ': 't',
            'ġ': 'g', 'ḟ': 'f', 'ṁ': 'm', 'ṗ': 'p', 'ṡ': 's',
            'ż': 'z', 'ḃ': 'b', 'ṫ': 't', 'ḣ': 'h', 'ẇ': 'w',
            'ẋ': 'x', 'ẏ': 'y', 'ḡ': 'g', 'ṉ': 'n', 'ṙ': 'r',
            
            # Специальные символы
            '’': "'", 'ʻ': "'", '´': "'", '`': "'", 'ʻ': "'"
        }
        
        # Добавляем комбинации
        combinations = {
            'aġ': 'ag', 'eġ': 'eg', 'iġ': 'ig', 'oġ': 'og', 'uġ': 'ug',
            'aṅ': 'ang', 'eṅ': 'eng', 'iṅ': 'ing', 'oṅ': 'ong', 'uṅ': 'ung',
            'lyṅ': 'ling', 'nyḋ': 'nid', 'sch': 'sh', 'tsch': 'ch',
            'çe': 'che', 'çu': 'chu', 'ça': 'cha'
        }
        
        mapping.update(combinations)
        return mapping

    def _load_comprehensive_dictionary(self):
        """Загрузка расширенного словаря с автоматическим дополнением"""
        dictionary = {
            # Основные слова из текста
            "māze": "dream", "çad": "if", "riki": "are", "vēirşe": "not merely",
            "yir": "but", "mumo": "products", "xoça": "subjective", 
            "tōḑeşur": "mind", "paju": "creations", "kāḑular": "of transcendent",
            "seg": "of", "çox": "transcendent", "ķoc": "intelligence",
            "peup": "universal", "leşa": "principle", "ņiq": "that",
            "loņe": "permeates", "rairşu": "life", "kiçe": "maintains",
            "tādaţu": "balance", "now": "within", "lax": "nature",
            "puilḑe": "creation", "ņun": "what", "quh": "teleologically",
            "vodaņa": "directed", "sāwu": "towards", "zēŗalar": "activating",
            "nāgu": "dimensions", "yāxa": "our", "sig": "subjectivity",
            "rēķalar": "blocked", "walu": "insufficiently", "luwo": "stimulated",
            "peuj": "neglected", "vema": "practical", "xoce": "standpoint",
            "vōbo": "better", "qiŗu": "crew", "soz": "stasis", "seinle": "connected",
            "yufi": "support", "lāyo": "lay", "xijo": "physiologically",
            "sāşedin": "controlled", "wot": "monitored", "veuy": "recreating",
            "qōirţa": "sleep", "xēţibur": "cycles", "reçe": "duration",
            "poon": "journey", "soḑi": "essential", "seilņu": "function",
            "juem": "onboard", "rajo": "maintenance", "xēşoşur": "system",
            "qēqa": "consciousness", "yādaŗa": "loses", "ḑiw": "critical",
            "vādaŗu": "awareness", "zauk": "position", "tairņu": "reality",
            "lōinço": "allows", "wōţabur": "fully", "nişa": "immerse",
            
            # Новые слова из анализа текста
            "ced": "the", "ţok": "and", "çed": "the", "ḑos": "that", "ḑir": "this",
            "ledu": "to", "yuro": "you", "sil": "will", "wal": "all", "ken": "as",
            "neb": "be", "nux": "us", "tuf": "of", "nen": "in", "rer": "are",
            "ses": "these", "pac": "when", "tol": "for", "suçe": "however",
            "vōnu": "therefore", "xaj": "thus", "reiv": "while", "miy": "my",
            "net": "not", "juc": "just", "sako": "it", "led": "that",
            
            # Частые комбинации
            "aġ": "to", "ä": "not", "ḑ": "the", "ç": "with", "re": "re",
            "lyṅ": "ing", "tyr": "er", "bryd": "able", "iẗ": "ity", "eyn": "ion"
        }
        
        # Загрузка из файла если существует
        if os.path.exists('enhanced_alien_dictionary.json'):
            try:
                with open('enhanced_alien_dictionary.json', 'r', encoding='utf-8') as f:
                    dictionary.update(json.load(f))
            except:
                pass
                
        return dictionary

    def _build_morpheme_patterns(self):
        """Расширенные паттерны морфем"""
        patterns = {
            # Префиксы (40+)
            'aġ': ['to', 'toward', 'for', 'at', 'on'],
            'ä': ['un', 'not', 'non', 'without', 'anti'],
            'ḑ': ['the', 'this', 'that', 'these', 'those'],
            'ç': ['with', 'by', 'through', 'via', 'using'],
            're': ['re', 'again', 'back', 'retro'],
            'pre': ['pre', 'before', 'prior'],
            'post': ['post', 'after', 'later'],
            'anti': ['anti', 'against', 'counter'],
            
            # Суффиксы (50+)
            'lyṅ': ['ing', 'ation', 'ment', 'process', 'action'],
            'tyr': ['er', 'or', 'ist', 'agent', 'doer'],
            'bryd': ['able', 'ible', 'ful', 'capable', 'possible'],
            'äfo': ['less', 'without', 'free', 'missing'],
            'iẗ': ['ity', 'ness', 'ship', 'hood', 'state'],
            'eyn': ['ion', 'tion', 'sion', 'ation'],
            'ment': ['ment', 'result', 'product'],
            'ness': ['ness', 'quality', 'state'],
            
            # Корни (100+)
            'māz': ['dream', 'vision', 'fantasy', 'imagination'],
            'puil': ['build', 'construct', 'create', 'make'],
            'rair': ['life', 'live', 'living', 'existence'],
            'qēq': ['mind', 'conscious', 'aware', 'thought'],
            'lōin': ['allow', 'enable', 'permit', 'let'],
            'veuy': ['create', 'make', 'generate', 'produce'],
            'ţok': ['and', 'also', 'plus', 'together'],
            'ņag': ['not', 'no', 'without', 'lack'],
            'şuw': ['with', 'together', 'along', 'accompany'],
            'ķin': ['in', 'inside', 'within', 'internal'],
            'çor': ['for', 'because', 'since', 'reason'],
            'ḑos': ['that', 'which', 'who', 'whom'],
            'ḑir': ['this', 'here', 'present', 'current'],
            'ledu': ['to', 'toward', 'into', 'direction'],
            'yuro': ['you', 'your', 'yours', 'yourself'],
            'rajo': ['can', 'able', 'capable', 'possible'],
            'sil': ['will', 'shall', 'would', 'future'],
            'wal': ['all', 'every', 'each', 'whole'],
            'now': ['in', 'inside', 'within', 'internal'],
            'seg': ['of', 'from', 'out', 'source'],
            'ken': ['as', 'like', 'similar', 'same'],
            'think': ['think', 'consider', 'believe'],
            'know': ['know', 'understand', 'comprehend'],
            'see': ['see', 'view', 'observe', 'watch'],
            'feel': ['feel', 'sense', 'experience'],
            'want': ['want', 'desire', 'wish'],
            'need': ['need', 'require', 'necessitate'],
            'time': ['time', 'period', 'duration'],
            'space': ['space', 'area', 'volume'],
            'system': ['system', 'network', 'structure'],
        }
        
        # Добавляем слова из словаря как корни
        for word, meaning in self._load_comprehensive_dictionary().items():
            if len(word) > 3 and not any(c in word for c in ' .!?,;:'):
                if word not in patterns:
                    patterns[word] = [meaning]
        
        return patterns
    
    def _load_comprehensive_dictionary(self):
        """Загрузка расширенного словаря"""
        dictionary = {
            # Основные слова из текста
            "māze": "dream", "çad": "if", "riki": "are", "vēirşe": "not merely",
            "yir": "but", "mumo": "products", "xoça": "subjective", 
            "tōḑeşur": "mind", "paju": "creations", "kāḑular": "of transcendent",
            "seg": "of", "çox": "transcendent", "ķoc": "intelligence",
            "peup": "universal", "leşa": "principle", "ņiq": "that",
            "loņe": "permeates", "rairşu": "life", "kiçe": "maintains",
            "tādaţu": "balance", "now": "within", "lax": "nature",
            "puilḑe": "creation", "ņun": "what", "quh": "teleologically",
            
            # Добавляем больше слов для лучшего покрытия
            "AI-generated": "AI-generated", "Hypnotic": "Hypnotic", 
            "Dreams": "Dreams", "Dreaming": "Dreaming", "under": "under",
            "AI": "AI", "Hypnosis": "Hypnosis"
        }
        
        # Загрузка из файла если существует
        if os.path.exists('enhanced_alien_dictionary.json'):
            try:
                with open('enhanced_alien_dictionary.json', 'r', encoding='utf-8') as f:
                    dictionary.update(json.load(f))
            except:
                pass
                
        return dictionary
    
    def _build_syntax_rules(self):
        """Улучшенные синтаксические правила"""
        return {
            'svo_order': True,
            'prepositions_before': True,
            'adjectives_before_nouns': True,
            'common_structures': [
                ('NP', 'VP'),           # Noun Phrase + Verb Phrase
                ('NP', 'VP', 'NP'),     # Subject + Verb + Object
                ('PP', 'NP'),           # Prepositional Phrase + Noun Phrase
                ('ADJ', 'NP'),          # Adjective + Noun Phrase
                ('ADV', 'VP'),          # Adverb + Verb Phrase
            ],
            'sentence_patterns': [
                ('QUESTION', 'AUX', 'NP', 'VP'),
                ('DECLARATIVE', 'NP', 'VP', 'NP'),
                ('IMPERATIVE', 'VP', 'NP')
            ]
        }
    
    def _build_semantic_clusters(self):
        """Расширенные семантические кластеры"""
        return {
            'consciousness': ['dream', 'mind', 'awareness', 'conscious', 'thought', 
                             'subjective', 'psychology', 'mental'],
            'technology': ['ai', 'system', 'computer', 'algorithm', 'network', 
                          'device', 'digital', 'electronic'],
            'science': ['development', 'research', 'experiment', 'analysis', 
                       'theory', 'study', 'investigation'],
            'action': ['create', 'build', 'make', 'generate', 'develop', 
                      'produce', 'construct'],
            'negation': ['not', 'no', 'without', 'never', 'nothing', 
                        'absence', 'lack'],
            'connection': ['with', 'and', 'together', 'connect', 'link', 
                          'relationship', 'association'],
            'location': ['in', 'within', 'inside', 'at', 'on', 'place', 
                        'position', 'location'],
            'purpose': ['for', 'to', 'toward', 'purpose', 'goal', 
                       'objective', 'aim'],
            'time': ['when', 'while', 'during', 'after', 'before', 
                    'time', 'period', 'duration'],
            'quantity': ['all', 'some', 'many', 'few', 'several', 
                        'number', 'amount', 'quantity']
        }

    def enhanced_contextual_analysis(self, word: str, context_words: List[str], position: int) -> List[Tuple[str, float]]:
        """Улучшенный контекстный анализ с семантическим учетом"""
        predictions = []
        
        # Прямой поиск в словаре (высший приоритет)
        if word in self.dictionary:
            predictions.append((self.dictionary[word], 0.95))
        
        # Нормализованный поиск
        normalized = self.normalize_phonetic(word)
        if normalized in self.dictionary and normalized != word:
            predictions.append((self.dictionary[normalized], 0.9))
        
        # Анализ морфем с комбинированием значений
        morphemes = self.deep_morpheme_analysis(word)
        if morphemes and any(m[0] != 'unknown' for m in morphemes):
            combined_meaning = self._combine_morpheme_meanings(morphemes)
            if combined_meaning:
                predictions.append((combined_meaning, 0.8))
        
        # Семантический анализ контекста
        semantic_predictions = self._semantic_context_analysis(word, context_words, position)
        predictions.extend(semantic_predictions)
        
        # Поиск похожих слов в словаре
        similar_words = self._find_similar_dictionary_words(word)
        for similar_word, similarity in similar_words:
            if similarity > 0.8:
                predictions.append((self.dictionary[similar_word], similarity * 0.8))
        
        return predictions

    def _combine_morpheme_meanings(self, morphemes: List[Tuple[str, str, List[str]]]) -> Optional[str]:
        """Комбинирование значений морфем в осмысленное слово"""
        if not morphemes:
            return None
            
        # Если есть полное слово
        if morphemes[0][0] == 'full_word':
            return morphemes[0][2][0]
        
        # Комбинируем префиксы, корни и суффиксы
        prefix_meanings = []
        root_meanings = []
        suffix_meanings = []
        
        for morpheme_type, morpheme, meanings in morphemes:
            if meanings == ['unknown']:
                continue
                
            if morpheme_type == 'prefix':
                prefix_meanings.extend(meanings[:1])  # Берем первое значение
            elif morpheme_type == 'root':
                root_meanings.extend(meanings[:1])
            elif morpheme_type == 'suffix':
                suffix_meanings.extend(meanings[:1])
        
        if root_meanings:
            result = root_meanings[0]
            
            # Добавляем префиксы
            for prefix in prefix_meanings:
                if prefix in ['not', 'un', 'non']:
                    result = f"not {result}"
                elif prefix in ['to', 'toward']:
                    result = f"to {result}"
                elif prefix in ['re', 'again']:
                    result = f"re{result}"
            
            # Обрабатываем суффиксы
            for suffix in suffix_meanings:
                if suffix in ['ing', 'ation']:
                    if not result.endswith('e'):
                        result = f"{result}ing"
                    else:
                        result = f"{result[:-1]}ing"
                elif suffix in ['er', 'or']:
                    result = f"{result}er"
                elif suffix in ['able', 'ible']:
                    result = f"{result}able"
                elif suffix in ['ity', 'ness']:
                    result = f"{result}ity"
            
            return result
        
        return None
    
    def _semantic_context_analysis(self, word: str, context_words: List[str], position: int) -> List[Tuple[str, float]]:
        """Семантический анализ контекста"""
        predictions = []
        
        # Анализируем окно из 5 слов вокруг
        window_size = 5
        start = max(0, position - window_size)
        end = min(len(context_words), position + window_size + 1)
        
        context_window = context_words[start:end]
        
        # Ищем переведенные слова в контексте
        translated_context = []
        for i, ctx_word in enumerate(context_window):
            if i != position - start and self.is_alien_word(ctx_word):
                if ctx_word in self.session_memory:
                    translated_context.append(self.session_memory[ctx_word])
            elif self.is_english_word(ctx_word):
                translated_context.append(ctx_word)
        
        # Анализ семантических связей
        for translated_word in translated_context:
            # Простые эвристики для семантических связей
            semantic_links = self._get_semantic_links(translated_word)
            for linked_word in semantic_links:
                similarity = self.advanced_phonetic_similarity(word, linked_word)
                if similarity > 0.6:
                    predictions.append((translated_word, similarity * 0.7))
        
        return predictions
    
    def _get_semantic_links(self, word: str) -> List[str]:
        """Получение семантически связанных слов"""
        semantic_groups = {
            'dream': ['sleep', 'consciousness', 'mind', 'subjective', 'psychology'],
            'intelligence': ['mind', 'consciousness', 'awareness', 'thought', 'cognitive'],
            'creation': ['build', 'make', 'generate', 'produce', 'construct'],
            'system': ['network', 'structure', 'organization', 'framework'],
            'life': ['existence', 'being', 'living', 'biological'],
            'balance': ['equilibrium', 'harmony', 'stability', 'homeostasis'],
            'nature': ['environment', 'world', 'universe', 'cosmos']
        }
        
        for group, words in semantic_groups.items():
            if word in words or any(word in w for w in words):
                return words
        return []
    
    def _find_similar_dictionary_words(self, word: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Поиск похожих слов в словаре"""
        similar = []
        normalized_word = self.normalize_phonetic(word)
        
        for dict_word in self.dictionary:
            normalized_dict = self.normalize_phonetic(dict_word)
            similarity = self.advanced_phonetic_similarity(normalized_word, normalized_dict)
            
            if similarity >= threshold:
                similar.append((dict_word, similarity))
        
        # Сортируем по убыванию схожести
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:3]  # Возвращаем топ-3
    
    def decode_single_word(self, word: str, all_words: List[str], position: int) -> str:
        """Улучшенное декодирование отдельного слова"""
        # Пропускаем английские слова и пунктуацию
        if self.is_english_word(word) or not word.replace('.', '').replace(',', '').replace('!', '').replace('?', '').isalnum():
            return word
        
        # Используем кэш сессии для повторяющихся слов
        if word in self.session_memory:
            return self.session_memory[word]
        
        # Прямой поиск в словаре
        if word in self.dictionary:
            result = self.dictionary[word]
            self.session_memory[word] = result
            return result
        
        # Нормализованный поиск
        normalized = self.normalize_phonetic(word)
        if normalized in self.dictionary:
            result = self.dictionary[normalized]
            self.session_memory[word] = result
            return result
        
        # Улучшенный контекстный анализ
        predictions = self.enhanced_contextual_analysis(word, all_words, position)
        
        if predictions:
            # Выбираем предсказание с наибольшей уверенностью
            best_prediction = max(predictions, key=lambda x: x[1])
            meaning, confidence = best_prediction
            
            if confidence >= self.confidence_threshold:
                self.learned_patterns[word].append((meaning, confidence, all_words))
                self.session_memory[word] = meaning
                return meaning
        
        # Агрессивный поиск похожих английских слов
        similar_english = self.aggressive_english_similarity(normalized)
        if similar_english:
            result = similar_english[0]
            self.session_memory[word] = result
            return result
        
        # Если ничего не найдено, пытаемся разбить на составные части
        compound_translation = self._try_compound_translation(word)
        if compound_translation:
            self.session_memory[word] = compound_translation
            return compound_translation
        
        # Последняя попытка - возвращаем наиболее вероятную нормализованную версию
        final_guess = self._final_guess(word)
        self.session_memory[word] = final_guess
        return final_guess
    
    def aggressive_english_similarity(self, word: str) -> List[str]:
        """Агрессивный поиск похожих английских слов"""
        common_english_words = [
            'the', 'and', 'for', 'with', 'that', 'this', 'from', 'have', 'are',
            'you', 'your', 'can', 'will', 'not', 'but', 'what', 'when', 'where',
            'why', 'how', 'which', 'who', 'their', 'there', 'been', 'because',
            'into', 'through', 'during', 'before', 'after', 'between', 'within',
            'about', 'above', 'below', 'under', 'over', 'across', 'around',
            'since', 'until', 'while', 'though', 'although', 'unless', 'whether',
            'either', 'neither', 'both', 'each', 'every', 'some', 'any', 'all',
            'such', 'same', 'different', 'many', 'much', 'more', 'most', 'less',
            'few', 'several', 'enough', 'too', 'very', 'so', 'as', 'like',
            'just', 'only', 'also', 'even', 'still', 'already', 'yet', 'never',
            'always', 'often', 'sometimes', 'usually', 'generally', 'specifically',
            'could', 'would', 'should', 'might', 'may', 'must', 'shall'
        ]
        
        similarities = []
        for english_word in common_english_words:
            similarity = self.advanced_phonetic_similarity(word, english_word)
            if similarity > 0.6:  # Пониженный порог
                similarities.append((english_word, similarity))
        
        # Сортируем по убыванию схожести
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in similarities[:2]]  # Возвращаем топ-2
    
    def _try_compound_translation(self, word: str) -> Optional[str]:
        """Попытка перевода составных слов"""
        # Пробуем разные разделители
        for separator in ['', '-', '_']:
            for i in range(2, len(word) - 1):
                part1 = word[:i]
                part2 = word[i:]
                
                meaning1 = None
                meaning2 = None
                
                if part1 in self.dictionary:
                    meaning1 = self.dictionary[part1]
                else:
                    norm1 = self.normalize_phonetic(part1)
                    if norm1 in self.dictionary:
                        meaning1 = self.dictionary[norm1]
                
                if part2 in self.dictionary:
                    meaning2 = self.dictionary[part2]
                else:
                    norm2 = self.normalize_phonetic(part2)
                    if norm2 in self.dictionary:
                        meaning2 = self.dictionary[norm2]
                
                if meaning1 and meaning2:
                    return f"{meaning1}{separator}{meaning2}"
        
        return None
    
    def _final_guess(self, word: str) -> str:
        """Финальная попытка угадать значение"""
        normalized = self.normalize_phonetic(word)
        
        # Эвристики для окончаний
        if normalized.endswith(('ing', 'ung', 'ang', 'eng')):
            base = normalized[:-3]
            similar = self._find_similar_dictionary_words(base, 0.6)
            if similar:
                return f"{self.dictionary[similar[0][0]]}ing"
        
        if normalized.endswith(('ion', 'tion', 'sion')):
            base = normalized[:-3] if not normalized.endswith('tion') else normalized[:-4]
            similar = self._find_similar_dictionary_words(base, 0.6)
            if similar:
                return f"{self.dictionary[similar[0][0]]}ion"
        
        # Возвращаем нормализованную версию, но с пометкой
        return f"[{normalized}]"


    def advanced_phonetic_similarity(self, word1: str, word2: str) -> float:
        """Улучшенная фонетическая схожесть"""
        word1 = self.normalize_phonetic(word1)
        word2 = self.normalize_phonetic(word2)
        
        if word1 == word2:
            return 1.0
        
        # Множественные метрики схожести
        levenshtein_sim = 1 - (self.levenshtein_distance(word1, word2) / max(len(word1), len(word2)))
        sequence_sim = SequenceMatcher(None, word1, word2).ratio()
        
        # Взвешенная комбинация
        similarity = 0.6 * sequence_sim + 0.4 * levenshtein_sim
        
        # Бонус за общие префиксы/суффиксы
        if len(word1) > 3 and len(word2) > 3:
            prefix_len = 0
            for i in range(min(3, len(word1), len(word2))):
                if word1[i] == word2[i]:
                    prefix_len += 1
                else:
                    break
            
            suffix_len = 0
            for i in range(1, min(3, len(word1), len(word2)) + 1):
                if word1[-i] == word2[-i]:
                    suffix_len += 1
                else:
                    break
            
            bonus = (prefix_len + suffix_len) * 0.05
            similarity = min(1.0, similarity + bonus)
        
        return max(0.0, similarity)
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Расстояние Левенштейна"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def normalize_phonetic(self, word: str) -> str:
        """Улучшенная фонетическая нормализация"""
        if not word:
            return word
            
        normalized = word.lower()
        
        # Применяем отображение символов
        for alien_char, base_char in self.char_mapping.items():
            normalized = normalized.replace(alien_char, base_char)
        
        # Упрощаем повторяющиеся символы
        normalized = re.sub(r'(.)\1+', r'\1', normalized)
        
        # Удаляем диакритические знаки
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized
    
    def deep_morpheme_analysis(self, word: str) -> List[Tuple[str, str, List[str]]]:
        """Глубокий анализ морфем"""
        morphemes = []
        remaining = word.lower()
        
        # Поиск в словаре полных слов (самый высокий приоритет)
        if word in self.dictionary:
            return [('full_word', word, [self.dictionary[word]])]
        
        if remaining in self.dictionary:
            return [('full_word', remaining, [self.dictionary[remaining]])]
        
        # Анализ комбинированных морфем
        max_attempts = 5
        attempts = 0
        
        while remaining and attempts < max_attempts:
            attempts += 1
            found = False
            
            # Ищем самые длинные совпадения сначала
            for length in range(min(10, len(remaining)), 0, -1):
                segment = remaining[:length]
                if segment in self.morpheme_patterns:
                    morpheme_type = self._classify_morpheme(segment)
                    morphemes.append((morpheme_type, segment, self.morpheme_patterns[segment]))
                    remaining = remaining[length:]
                    found = True
                    break
            
            if not found:
                # Пробуем разбить на более мелкие части
                if len(remaining) > 2:
                    segment = remaining[:2]
                    morphemes.append(('unknown', segment, ['unknown']))
                    remaining = remaining[2:]
                else:
                    morphemes.append(('unknown', remaining, ['unknown']))
                    break
        
        return morphemes
    
    def _classify_morpheme(self, morpheme: str) -> str:
        """Классификация морфемы"""
        if morpheme in ['aġ', 'ä', 'ḑ', 'ç', 're', 'pre', 'post', 'anti']:
            return 'prefix'
        elif morpheme in ['lyṅ', 'tyr', 'bryd', 'äfo', 'iẗ', 'eyn', 'ment', 'ness']:
            return 'suffix'
        else:
            return 'root'
    
    def contextual_meaning_prediction(self, word: str, context_words: List[str], position: int) -> List[Tuple[str, float]]:
        """Предсказание значения на основе контекста"""
        predictions = []
        
        # Прямой поиск в словаре
        if word in self.dictionary:
            predictions.append((self.dictionary[word], 0.95))
        
        # Анализ морфем
        morphemes = self.deep_morpheme_analysis(word)
        
        if morphemes and morphemes[0][0] == 'full_word':
            return predictions
        
        # Строим предсказания на основе морфем
        root_meanings = []
        prefix_meaning = None
        suffix_meaning = None
        
        for morpheme_type, morpheme, meanings in morphemes:
            if morpheme_type == 'root' and meanings != ['unknown']:
                root_meanings.extend(meanings)
            elif morpheme_type == 'prefix' and meanings:
                prefix_meaning = meanings[0]
            elif morpheme_type == 'suffix' and meanings:
                suffix_meaning = meanings[0]
        
        # Комбинируем значения
        if root_meanings:
            for root_meaning in root_meanings[:2]:  # Берем до 2 значений корня
                prediction = root_meaning
                confidence = 0.7
                
                # Учитываем префиксы
                if prefix_meaning:
                    if prefix_meaning in ['not', 'un', 'non']:
                        prediction = f"not {prediction}"
                        confidence += 0.1
                    elif prefix_meaning in ['to', 'toward']:
                        prediction = f"to {prediction}"
                        confidence += 0.1
                
                # Учитываем суффиксы
                if suffix_meaning:
                    if suffix_meaning in ['ing', 'ation']:
                        prediction = f"{prediction}ing"
                        confidence += 0.1
                    elif suffix_meaning in ['er', 'or']:
                        prediction = f"{prediction}er"
                        confidence += 0.1
                    elif suffix_meaning in ['able', 'ible']:
                        prediction = f"{prediction}able"
                        confidence += 0.1
                
                predictions.append((prediction, min(confidence, 0.9)))
        
        # Анализ контекста
        context_similarity = self._analyze_context_similarity(word, context_words, position)
        for context_word, similarity in context_similarity:
            if similarity > 0.7:
                predictions.append((context_word, similarity * 0.8))
        
        return predictions
    
    def _analyze_context_similarity(self, target_word: str, context_words: List[str], position: int) -> List[Tuple[str, float]]:
        """Анализ схожести с контекстными словами"""
        similarities = []
        
        # Анализируем окно вокруг слова
        window_size = 3
        start = max(0, position - window_size)
        end = min(len(context_words), position + window_size + 1)
        
        for i in range(start, end):
            if i != position and self.is_alien_word(context_words[i]):
                context_word = context_words[i]
                if context_word in self.dictionary:
                    # Проверяем семантическую связь
                    similarity = self.advanced_phonetic_similarity(target_word, context_word)
                    if similarity > 0.6:
                        similarities.append((self.dictionary[context_word], similarity))
        
        return similarities
    
    def is_alien_word(self, word: str) -> bool:
        """Проверка, является ли слово инопланетным"""
        if self.is_english_word(word):
            return False
        
        alien_chars = 'āēīōūäöüçşţḑņķŗṅḋẗġḟṁṗṡżḃṫḣẇẋẏ'
        return any(char in word for char in alien_chars)
    
    def decode_text(self, text: str) -> str:
        """Основной метод декодирования"""
        sentences = self.split_sentences(text)
        decoded_sentences = []
        
        for sentence in sentences:
            if self.is_english_text(sentence):
                decoded_sentences.append(sentence)
                continue
                
            words = self.split_words(sentence)
            decoded_words = []
            
            for i, word in enumerate(words):
                # Используем кэш сессии для повторяющихся слов
                if word in self.session_memory:
                    decoded_words.append(self.session_memory[word])
                    continue
                    
                decoded_word = self.decode_single_word(word, words, i)
                self.session_memory[word] = decoded_word
                decoded_words.append(decoded_word)
            
            decoded_sentence = self.reconstruct_sentence(decoded_words, sentence)
            decoded_sentences.append(decoded_sentence)
        
        return ' '.join(decoded_sentences)
    
    def decode_single_word(self, word: str, all_words: List[str], position: int) -> str:
        """Декодирование отдельного слова"""
        # Пропускаем английские слова и пунктуацию
        if self.is_english_word(word) or not word.replace('.', '').replace(',', '').replace('!', '').replace('?', '').isalnum():
            return word
        
        # Прямой поиск в словаре
        if word in self.dictionary:
            return self.dictionary[word]
        
        normalized = self.normalize_phonetic(word)
        if normalized in self.dictionary:
            return self.dictionary[normalized]
        
        # Контекстный анализ
        predictions = self.contextual_meaning_prediction(word, all_words, position)
        
        if predictions:
            # Выбираем предсказание с наибольшей уверенностью
            best_prediction = max(predictions, key=lambda x: x[1])
            meaning, confidence = best_prediction
            
            if confidence >= self.confidence_threshold:
                self.learned_patterns[word].append((meaning, confidence, all_words))
                return meaning
        
        # Поиск похожих английских слов
        similar_english = self.find_similar_english_words(normalized)
        if similar_english:
            return similar_english[0]
        
        # Если ничего не найдено, возвращаем нормализованную версию
        return normalized
    
    def find_similar_english_words(self, word: str) -> List[str]:
        """Поиск похожих английских слов"""
        common_english_words = [
            'the', 'and', 'for', 'with', 'that', 'this', 'from', 'have', 'are',
            'you', 'your', 'can', 'will', 'not', 'but', 'what', 'when', 'where',
            'why', 'how', 'which', 'who', 'their', 'there', 'been', 'because',
            'into', 'through', 'during', 'before', 'after', 'between', 'within',
            'about', 'above', 'below', 'under', 'over', 'across', 'around',
            'since', 'until', 'while', 'though', 'although', 'unless', 'whether',
            'either', 'neither', 'both', 'each', 'every', 'some', 'any', 'all',
            'such', 'same', 'different', 'many', 'much', 'more', 'most', 'less',
            'few', 'several', 'enough', 'too', 'very', 'so', 'as', 'like',
            'just', 'only', 'also', 'even', 'still', 'already', 'yet', 'never',
            'always', 'often', 'sometimes', 'usually', 'generally', 'specifically'
        ]
        
        similarities = []
        for english_word in common_english_words:
            similarity = self.advanced_phonetic_similarity(word, english_word)
            if similarity > 0.7:  # Повышенный порог
                similarities.append((english_word, similarity))
        
        # Сортируем по убыванию схожести
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in similarities[:3]]  # Возвращаем топ-3
    
    def split_sentences(self, text: str) -> List[str]:
        """Разбиение текста на предложения"""
        # Улучшенное разбиение на предложения
        sentences = re.split(r'([.!?]+[\s$])', text)
        result = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                result.append(sentences[i] + sentences[i+1])
            else:
                result.append(sentences[i])
        return [s.strip() for s in result if s.strip()]
    
    def split_words(self, sentence: str) -> List[str]:
        """Разбиение предложения на слова"""
        # Улучшенное разбиение на слова
        words = re.findall(r'[\wāēīōūäöüçşţḑņķŗṅḋẗġḟṁṗṡżḃṫḣẇẋẏ]+[.,!?;]?', sentence)
        return words
    
    def is_english_text(self, text: str) -> bool:
        """Проверка, является ли текст английским"""
        # Более точная проверка английского текста
        english_pattern = r'^[A-Za-z0-9\s\.,!?;:\'"\-\(\)]+$'
        return bool(re.match(english_pattern, text.strip()))
    
    def is_english_word(self, word: str) -> bool:
        """Проверка, является ли слово английским"""
        # Более точная проверка английского слова
        english_pattern = r'^[A-Za-z]+[.,!?]?$'
        return bool(re.match(english_pattern, word))
    
    def reconstruct_sentence(self, words: List[str], original_sentence: str) -> str:
        """Восстановление предложения с правильной пунктуацией"""
        sentence = ' '.join(words)
        
        # Сохраняем оригинальную пунктуацию
        if original_sentence and original_sentence[-1] in '.!?':
            if sentence and sentence[-1] in '.!?':
                sentence = sentence[:-1] + original_sentence[-1]
            else:
                sentence += original_sentence[-1]
        
        # Капитализация первого слова
        if sentence and sentence[0].isalpha():
            sentence = sentence[0].upper() + sentence[1:]
        
        return sentence
    
    def calculate_confidence(self, translated_text: str) -> float:
        """Вычисление уверенности в переводе"""
        words = translated_text.split()
        if not words:
            return 0.0
        
        # Подсчет непереведенных слов (в квадратных скобках или неизвестных)
        unknown_count = 0
        for word in words:
            if (word.startswith('[') and word.endswith(']')) or word == 'unknown':
                unknown_count += 1
        
        confidence = 1 - (unknown_count / len(words))
        return max(0.0, min(1.0, confidence))

def improve_dictionary_based_on_text(text: str, decoder: EnhancedAlienDecoder) -> Dict[str, str]:
    """Автоматическое улучшение словаря на основе текста"""
    words = decoder.split_words(text)
    alien_words = [w for w in words if decoder.is_alien_word(w) and w not in decoder.dictionary]
    
    word_freq = Counter(alien_words)
    common_alien_words = [word for word, count in word_freq.most_common(50) if count > 2]
    
    new_entries = {}
    
    # Эвристики для частых слов
    for word in common_alien_words:
        normalized = decoder.normalize_phonetic(word)
        
        # Пробуем угадать значение на основе морфем
        morphemes = decoder.deep_morpheme_analysis(word)
        if morphemes and any(m[0] != 'unknown' for m in morphemes):
            combined = decoder._combine_morpheme_meanings(morphemes)
            if combined:
                new_entries[word] = combined
                continue
        
        # Пробуем найти похожее английское слово
        similar_english = decoder.aggressive_english_similarity(normalized)
        if similar_english:
            new_entries[word] = similar_english[0]
    
    return new_entries

# Улучшенный главный класс
class AdvancedAlienLanguageDecoder:
    def __init__(self):
        self.decoder = EnhancedAlienDecoder()
        self.translation_history = []
        
    def decode_text(self, text: str) -> str:
        """Основной метод декодирования"""
        return self.decoder.decode_text(text)
    
    def decode_text_with_analysis(self, text: str) -> Tuple[str, Dict]:
        """Декодирование с детальным анализом"""
        result = self.decode_text(text)
        
        analysis = {
            'learned_patterns': dict(self.decoder.learned_patterns),
            'session_memory': dict(self.decoder.session_memory),
            'confidence_level': self.decoder.calculate_confidence(result),
            'translated_words': len(self.decoder.session_memory),
            'unknown_words': sum(1 for word in result.split() 
                               if (word.startswith('[') and word.endswith(']')) or word == 'unknown')
        }
        
        self.translation_history.append({
            'original': text[:100] + '...' if len(text) > 100 else text,
            'translated': result[:100] + '...' if len(result) > 100 else result,
            'confidence': analysis['confidence_level'],
            'timestamp': np.datetime64('now')
        })
        
        return result, analysis
    
    def save_progress(self, filename: str):
        """Сохранение прогресса обучения"""
        progress_data = {
            'learned_patterns': dict(self.decoder.learned_patterns),
            'session_memory': dict(self.decoder.session_memory),
            'translation_history': self.translation_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    def load_progress(self, filename: str):
        """Загрузка прогресса обучения"""
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                self.decoder.learned_patterns.update(progress_data.get('learned_patterns', {}))
                self.decoder.session_memory.update(progress_data.get('session_memory', {}))
                self.translation_history = progress_data.get('translation_history', [])

# Функции для работы с кодом
def translate_alien_text(text: str) -> str:
    """Функция для перевода текста инопланетного языка"""
    decoder = AdvancedAlienLanguageDecoder()
    return decoder.decode_text(text)

def analyze_and_translate(text: str) -> Tuple[str, Dict]:
    """Анализ и перевод с детальной информацией"""
    decoder = AdvancedAlienLanguageDecoder()
    return decoder.decode_text_with_analysis(text)

if __name__ == "__main__":
    # Тестирование улучшенного декодера
    test_text = """AI-generated Hypnotic Dreams

Dreaming under AI Hypnosis


Māze çad riki vēirşe yir çed mumo xoça ţok çed tōḑeşur paju, kāḑular seg çox ķoc peup şuw leşa ņiq çed loņe rairşu ţok kiçe tādaţu now ķin lax ţok puilḑe. Ņun ķin çed quh vodaņa ţok sāwu zēŗalar ķin māze nāgu ķoc yāxa şuw sig ţok rēķalar çed walu luwo, peuj vema xoce ņag vōbo qiŗu soz seinle yufi. Lāyo xijo sāşedin a wot veuy ņiq çed qōirţa ķin xēţibur, çed reçe ķin māze, ţok çed poon soḑi ķin seilņu ņag juem rajo yuro xēşoşur wal seinņa.


Māze nāgu qēqa yādaŗa ḑiw a luwo ķin vādaŗu zauk, tairņu lōinço, ţok wōţabur nişa:


"What if dreams are not merely products of the subjective mind but creations of Transcendent Intelligence, a universal principle that permeates life and maintains balance within Nature? What if dreams are teleologically directed towards activating dimensions of our subjectivity that are blocked, insufficiently stimulated, or neglected? From a practical standpoint, it is much better to have a crew in stasis, connected to life support systems, physiologically controlled and monitored, while recreating sleep cycles for the duration of the journey. That is the essential function of the onboard sleep maintenance system."


Ḑir lāyo laça ţos luwo, sil rēdaŗi vēyi zobu puilde, rauj neiţşe, ţok vişa weilşe ņux nōwo. Lāyo pōpi zēiţŗu ņux çed kigo sēdaḑu çer çed xēţikan ţok loņe rairşu. Çed xēţikan rairşu, juça tewi nug lax ţok wuan rer, qēqa vaşe ķix veyi wōţaşur sāņuçiz qēiţe yōķeşur. Ḑos xeju, çed loņe rairşu nāgu quh noqi şuw keiţzi pōqa, siŗo wal ţok çed pezo ķin xēņular ņag qiŗu soz vōiţşo.


Çed pedu ķin māze ņeb yuro vōwu şuw vōizņo a walu luwo pāinḑu, zōdaçu sil şuw veut çed rudaka ķin leḑe qāra pāķukan juc peuj vudaza ţat a sailḑu şuw xoçi ņag vōbo yufi. Ḑiw moiţņi mape seilşe ţok soişŗo ḑir māze, peşe rēdaŗi voņa çed roh ķin vēye riel wal. Ḑos lōiri wal, sil pian zaķu ţok xēizŗa, vōţular lēdaşo şuw qōirça walu miça xēţigir.


Ņun xix zeg ķin kēfi māze şuw veut wal nāgu çed xijo ķin ḑoy pōve veyi. Reiv sil ḑeq pēsi wōţabur razu qēiţe māze niaz pēce sēka, vep ķin kifa ņag çed kiem, ţok voso ņag vōiţşo qēiţe seinņa ledu mose wēge qēilķi:


"In the dream state, the dreamer loses critical awareness of his position within the reality of dreams, but this is no accident. This position allows the dreamer to fully immerse himself in the symbolic world of dreams without rational resistance or fragmentation of experience. The human brain does not process intellectual and emotional experiences in isolation; rather, these domains interact dynamically, influencing how information is encoded, stored, and later retrieved. On the other hand, dream formation is a process heavily influenced by both cognitive and emotional experiences. There are neither cognitive, nor emotional experiences for a computer because, by definition, you cannot code an emotion into an algorithm. But you can always correlate an emotion to a specific EEG pattern, and teach the system to recognize that pattern. That's the problem, though: you are 'teaching' an emotion to a system, yet humans do not teach emotions to each other. They simply feel an emotion."


AI-generated Hypnotic Dreams: Dreaming under AI Hypnosis 1


Lēçetar veyi qēqa wōņiriţ xēiţşi koba, lōinçe ledu çed yine ķin xēţibur rajo vēsu ḑoy silķa soz vidu. Ḑos lāyo roik, wal sig ḑaf māze rēdaŗi mōizḑa ŗac a yib çer çed mor ţok lēçetar puilņe qiŗu soz seinņa.


Led viḑi a joug zeirţe ķin zirse viow lōba çed qiţḑi ledu wal ņeb mōizḑa ŗac piḑi şuw xōce ņag vōiţşo ledu mece puirçi wodaŗu ŗac māizşe teirko, veŗe yufi, qēiţe zişçu joom saco. Ḑos lēçetar voso, çed walu qēqa niap nōŗaçiz raŗo qēiţe sēka ledu lāq leḑe qeişḑo xōişţe tudu. Lāyo pōpi nuķa sāwu pac şih çed maub ķin lōiri seţo. Nāgu sako milņe a pedaķa ķin çed wāle rairşu, qēiţe xēçeşur sako yuro a toço vōso ņag qiŗu soz seinņa? Çed ziup liţa lōiri veyi tuirte neb wizsu şih çed qōirţa ķin xēţibur saka:


"Hypnotic suggestions can influence future nocturnal dream content. Hypnosis can be used to augment the same kind of dream incubation procedures that are used without trance to shape dreaming toward solving a specific problem or for creative inspiration."


Xaj ķin lēçetar sōķonin qēqa zeḑi ledu lōiri veyi ņeb yuro zaţa şuw çed yişşi paţi çox puve nōņakan ţok çed seņo soz woco şuw vaŗu reḑa ḑos çed ziup ķin wal. Neuro mumo nēta sişfi ledu çed yişşi nin zoz ḑir mēŗigir, qaḑe ḑos çed niņi māḑatar xāişķu çox walu tōdaşu. Çom lāyo xēmu, sako rajo yuro zēizḑo ledu çed yişşi veno çed vēyi ķin voşo vōbo yufi, mōinçe ḑiw a qoķi ķin xage, paķi, ţok pōve vel.


Suçe, lōba ķin çed soņu ledu māze ņeb vēme xoce ņag qiŗu soz seinņa xēņeçiz ledu çed vēyi lāg rāso paķi tuḑo. Lēce loinḑe ledu, ḑaf māze, sil ņeb zuişdu wot rōşekan ķin xēţibur ledu vōţunin toço sēilņi ņag qēilķi tuķu ķin seinņa. Lāyo qiţḑi sāşedin veuy ņiq zişçu māizşe lādaķo ţok xēņeşur māizşe peşi juḑa xoce ņag qiŗu soz vōiţşo nāgu xōçigir.


Ņux mōinţo nayi şuw quşu laça peşi, çed woco şuw nuca ḑoy çed soz vēyi nāgu wōsu vudaḑe ḑos soz xāga ţok yuek liaf:


"Hypnotic dreams from unusually hypnotisable people look much more like night-time dreams than like daydreams. Hypnotic dreams are much easier to remember than night-time dreams. It is possible but unusual to be amnesic for a hypnotic one, whereas we lose the vast majority of our night-time dream content before waking."


Woizķo ņiq çed vobe ţok poon soḑi ķin sif wal çox çed pāinķe ķin vōbo xoce nēişḑi a xişu rudaka. Miy sil vōņuriţ ņag qiŗu soz seinņa, led zēiţŗu çed pace ķin leḑe seilşe. Ķib lēçetar vōiţşo yoze, reoy, qēiţe qōdaņe? Led nāgu çed peup çox pēce sēka, ḑix ņag lōiri sēka xoeb çed nēno şuw weşa leḑe soḑi ţok pāinḑa yaŗu. Vōnu ḑos wizsu ņag qiŗu soz seinņa nuķa xāja pac şih çed sēçabur ķin ruķe, quif, ţok çed qōirţa ķin vidu.


Qugu, ŗac sōxu tes ņag ķoc qāra viḑo xizţa, lōiri ŗac xōru yado, mefo wedabi, ţok lēko weişhu, çed zeşa ķin koŗo şuw qiŗu soz xeap tēiţḑe lōşoriţ. Çed xēiţķu rajo moef ņux ņun ķin kainji şuw ņun ķin peup xōirķu ņag lāxe kep yufi ḑeq rēdaŗi susa ruķe ledu xēçeşur zedajo ḑos qoiq sōxu xōinķe yāfo:


"A generational spacecraft, known as a terminal vessel, must keep at least three thousand human children in stasis until it reaches rendezvous orbit. Sleeping for 25 years is simply impossible and counterproductive. Simulating reality in their brains is much more practical."


AI-generated Hypnotic Dreams: Dreaming under AI Hypnosis 2


Ḑos lōgo, çed kēgo ķin māze şuw sig wal ledu vēme xoce ņag vōbo qiŗu soz seinle yufi voto a neb sēdaḑu ķin mumu, maizŗe, ţok pilņa. Juc wēşedin rēdaŗi wēizķe lēçetar sēilņi ŗac pēḑagir ķin a xişu qōḑelar lēru, reiv sil xēlo şuw veut çed pēzo ledu roinŗe ḑos lēçetar walu rudaḑu. Ŗac juŗo tōle çaz koba ķin xēţibur ţok çed rec ķin māze, çed peup çox kuw veyi ţok sēka nin ziķe voiţe tōso. Koķa, juci ņun kāņeçiz lēçetar voso ŗac rāso paķi xepu qēiţe toço xēņular ņag qēilķi seinņa, lēce yag keh şuw nēja çed yine ķin çaz koba ķin xēţibur saka ţok çaz pōhu çom çed xedaķe.


Ḑos lāyo sailḑu ķin veuy, çed walu luwo mōizça çor qēlo ŗac a pōve viaf ḑix zōçabur ŗac a xōinķe moz çox reḑa ţok xēva ḑos ziķe sima xişu kiem:


"We trained three dreamers, called dreamer agents, at XViS. Upon hypnotically re-entering the dream scene, DA2 dreamer could ask characters: 'who are you?', 'what do you represent?', 'what are you trying to tell me?'. DA2 could look around a scene and ask: 'what is this place?' or 'what real-life setting does this resemble?' After the experiment we got quite unexpected yet crystal-clear answers."


Māze qēķekan ţat a kigo qair çor yuķa sako qeŗe weh yay ledu ķib voiţa zōgu şuw çed soz yool lias, ḑix ţop wāişḑa şuw çed nime nuirţe ņag juça sako vēme zoc ḑos xoişţi, puilḑe, ţok kāişņa. Lāyo māņadin ķin yado nāgu toņu lāŗegir şuw çed sēilņi çer xēizņu kebi, māpa, çed loinçe lāna şuw çed lom ţok çed xēmu ḑos juça lēçetar loinçe ķib wōji. Lāyo sēdaḑu xaoy a veçu teiţvi çox lōiţņo mōḑonin ḑos zaķu ţok rauj, qēki wāto şuw pēce paķi veuy. Xeşu a mico juḑa a meal meişce leḑe vaŗe, ţok a lom loirço yec sōķikan a veiţņu vēņelar, lōiri ŗac ledu ķin leḑe quci. Çed sōirņi zoc ḑos xōişţe tooh nāgu lōţidin; sako woişķu ţab çed rairşu ņeb yuro liţhi nōirţo, tādaça çed kōţitar ţok rēnu qōirţa ķin rauj saka. Rauj, qēqa xēşoşur a vāli niah ķin veyi, nāgu nişŗi vag ḑos lōiri a luwo, nēşoçiz a yaz ledu ņeb yuro qāķomen zirçu ņag wafa lōinçe. Lāyo rēņubur pira şuw çed kōçimen qōḑeşur qām, lōinçe ledu raun ķib çor ŗac xişto ŗac lēce rēdaŗi mose, ḑix nuho lis şuw quŗo woilşi ţat çed xuda ḑos juça lēce ķib zōzi.


Çed zuţi şuw vogu a rauj ţin xōḑilar ţok nen sako ņag ziço mēişḑe sōişţa ţab māze ņeb rēķalar çed xēţikan ţok loņe sēdaḑu şuw vuirvu xesu mōḑonin ḑos sōki ţok nōņakan. Lāyo māw xeju ḑos xōişţe tooh vōxi çor qēlo çed pēşuriţ ķin lōinçe ḑix zōçabur çed sege qōirţa ķin xēmu ţok rauj neiţşe, nuka pac şih çed rec ḑaf juça lōiri kuw qēţamen:


"If dreams are externally sourced, it suggests that consciousness is not solely a product of the brain, but a capacity for receiving and interpreting information from a wider field of awareness. It is as if our individual consciousness would be akin to a radio receiver, tuning into a signal originating from a central transmitter. This opens the possibility of shared dreaming, collective dream experiences, and even communication with other beings – human or otherwise – through the dream state. I don't see how our individuality could be erased; I see it more as a contextualisation within a far grander, more intricate design. The ego, so carefully constructed in waking life, needs to acknowledge a source beyond its control, that's all."


Çed reķa ķin zauk ḑir māze nāgu ziço pōilķa vēle ḑos lāyo kuw pedu. Ḑos vipu rōinķe, zauk nāgu qēqa wēdaķa çuh qiha luişţa, roŗo şuw a tuf vēyi ķin nōwo. Māze pozu ziķe silķa şuw yeip lāyo tufa, wexi vādaşo ņag livo peol. Lāyo tairņu zauj xēizŗa reke ḑos yir çed rirpi ķin rauj ţok çed niy ķin sako, neişḑo çed vag wāgu ķin xōişţe neoz. Nēta ņiq zauk, qaḑe ţab çed yişşi zodaŗa neoz şuw kāzu leqa, çad sima kōro şuw māze ŗac a lāişķu şuw sāţalar lēçetar pōqa. Ḑiw xaox a xeizra vōirşo ledu vomu lōinço, net ķib yōti zuţa şuw xir çed kōçimen qōḑeşur rec tēwu zauk ţok puilḑe. Ŗac a nel, māze pāņuçiz a kigo qējo şuw loŗa ţab xōişţe tōņuçiz qēķemen ḑos salo qēiţe ḑos xizşi ţok ţab lēce ņeb yuro wāno moec çom a yiag lapu tuel. Lāyo nuirţu reķa maça ḑos māw xeju şuw lāḑanin leḑa zeor ledu qēqa nētu paya seba şuw zudo minţu nēku.


Xēmu podaha ziķe voilķe xāişķu nāţabur ḑos çed vuţi ķin māze lōinçe. Çed moiţņi, çed meal xāņubur, ţok çed lom vib şen xej şuw çed qēge ķin çed māze vēyi. Lāyo niq ţat xēmo vel tōle vōxi çed seh ķin xoişţi ţok vōirşo. Sako nuķa poţi pac şih çed qōirţa ķin nōwo saka. Sōje a lom nenu şuw çed lōinçe şuw tup a rauj zaço ruag ţat çed xāhu lēce nōŗekan ņux çed meal kaişyo qēiţe woiţņu, ņun qoni pinje juci çed weh çer nōwo ţok lōinḑa nōwo nāgu ŗac xota xāçigir ŗac qēqa zēçoriţ.


Ḑaf lāyo rok, ņun ņeb veut çed soḑi ķin zōbo puilḑe ţeb çor qēlo leḑa zēŗalar ḑix zōçabur yev mez pōqa. Ḑiw koba ţab xēmu ņeb mōinçi çed soirņu ķin xoişţi, net ņeb tōizņa sēka ņiq çed lis ķin sil şuw kāzu setu, yir ḑos leḑa mōiţņe ţok ḑos wuţu rōinķe. Lāyo koba miţo çed peuv ķin lōinçe ŗac a lāişķu çox mōilţe yueh, yuij, ţok vişa wuam ţat xōinķe rōiţŗo:


"For a superintelligence, it is always easier to simulate reality in your brain than to simulate it in its entirety. You are unaware that you are in a stasis capsule, in a completely dark ship, controlled by the AI, for the duration of the journey. Your brain simply visualizes images, for example, an urban landscape, a forest, a valley. None of that exists: you are in stasis."


AI-generated Hypnotic Dreams: Dreaming under AI Hypnosis 3


Juc çed peup ķin māze şuw vuur nuirţu yado nāgu zeşa ḑos kāzu wawu, sako zōçabur yēxo tuizŗi vobe xēşoşur. Çed wādaŗi ņag juça raun ņeb yuro rēķalar nuķa vod pac şih xeţe, zāŗaçiz, ţok çed xēţamen ķin zōbo mēçatar koba ķin leḑe qāra sōki. Çed kah ķin māze ŗac yir a taizţe quti ţok a peup lāişķu çox reķa qōilşo a yaŗu viķo ķin çed vobe tudu tēwu ķoc pēsi. Ŗac juŗo veut çed kuw māņadin ses şuw māze, sako yuk sogi şuw xeşu çed soḑi ķin joţi lāyo pēşuriţ. Ţab weço juŗo voizxa ledu çed kēgo ķin lōiri leķu nāgu vobe ţok neok çed silķa zaoh ţok seinŗu?


Çed LyAV kaŗo vel ledu xej şuw çed lovi ķin māze lōinçe, ķin xēmu, nişo çer çed lom ţok çed peşe, ţok rōiţşi ķin lōinçe, qoni yuro zeos ņag kuŗi ţok nēno:


"Hypnosis is an altered state of consciousness, involving imaginative experiences associated with subjective conviction bordering on delusion and experienced involuntariness bordering on compulsion, which takes place in the context of a particular social interaction between hypnotist (in this case, LyAV) and subject. LyAV processes the subject using various procedures. Some of them will drive many people into the same states, from lassitude to catalepsy, from docility to roboticity, from visionary to hallucinatory. When the correct target state is achieved XViS starts guiding the dreamer, otherwise the dreamer is taken back to awake state and de-hypnotised. A hypnotic procedure is just a protocol used to establish a hypnotic situation and evaluate responses to it."


Ḑos logu, māze weav çor soilķo kigo zeg şuw çed soz yool lias; nuho, sako yāwi ţat çed nime zuţi şuw rēķalar xōişţe tōņuçiz ŗag ziķe zuiţa qaov. Lāyo yadano zōdaço çox pēce mōḑonin ḑos puilḑe, rauj, ţok zauk, rer sako a keda lāişķu çox net ţok peşe zōfa. Juci kew çox leḑa pāinḑa, mumo sēlu, qēiţe vişa wot poon veuy ķog çed qōirţa ķin xēţibur ţok sōki, çed soḑi ķin māze xēlo şuw neq ņoy wedaţe vāvo.


Ŗac çaz koba ķin māze viņi, meib çik weav çed rudaka ķin sēlu liţa ķoc peup keg ţok vobe xēşoşur. Çed lice ķin vōxa nēta ņag sāwe pēsi rēdaŗi koķa roŗu şuw a quh noilḑi xiq ķin çed rairşu, vōşibur a neoq wizsu şih çed sēçabur çer xēţibur, lōinçe, ţok çed soz vēyi.


Sōje ņun yāxa şuw xevi ņag qiŗu soz seinņa ḑaf māze juilda lōizçu, çed poon soḑi joişķi yuro pēce. Pac ķog çed qōirţa ķin seinņa, çed xāje çox xēţikan vēyi, ţok çed māizşe vod ķin seh joişķi vuat ŗac xāja lāiru çox weņo. Juem joişķi sako redaçi çox sōxu şuw vōņuriţ ņag vōiţşo ledu pian tuķu ķin seinņa ḑoy çaz xiq? Ţab joişķi lōiri sēilņi nēsa çaz koba ķin vidu, ţok juem vobe tudu joişķi zēiţşe şuw tēxa lēçetar vōņugir?


Vōnu ņag qiŗu soz seinņa roŗo şuw ŗaf ţab lēçetar voso rajo tuişço teof voy ţok koba çuh sōxo lēdaŗa. Sōje ņun zuişḑu çed vidu ķin qōşedin tuķu ķin xēţibur, sako xix a viţu ķin çaz nişo çor qēlo ņag qēilķi mapo ţat wādi ḑix zōçabur ņag çed kaŗo vēno ķin çed xedaķe. Lōiri wizşi, juci vēme ḑaf tēçonin, walu ţos luv sig ḑiw māze qēiţe qeilju, xēçeşur xāgi a yevo, quh siŗi zeov şuw koba çed kāyu māņinin ķin vidu. Ŗac lāyo veuy xēlo, qoiq çed yine çer mumo neçu ţok rāqo sēlu jōşidin paoq şuw yuro a yāfo jōḑunin qōilţi kāţikan. Lōţidin a zainţu çer vōxa viow ţok pōve vēyi rajo xap a nēḑuşur wepo liţa çed qōirţa ķin xēţibur.


Çed sailḑu ḑaf māze ņiq zobu luv xēçeşur mōizḑa ŗac a nik ledu ruķe nāgu çor mēdaķo çed wāwu ķin mumu. Sako nāgu zōçabur jōişḑo wōsu ņiq çed xōinķe lēru ķin soz vēyi, sāve, ţok çed māizşe paçe çox koba:


"You are not awake, nor are you dreaming. It is not a hallucination, nor are you in a coma. Brain stasis is a state of dreamlike hypnosis, so to speak. It is a new altered state of consciousness, the only one that allows you to live a life as close to real life as possible. LyAV is responsible for your education, recreating situations and experiences that will shape you as a person. When you come out of stasis, 25 years will have passed. You will be 26 years old and nothing around you will seem strange. You will know what you have to do: start a new life on a new planet."


Tol, ŗac wēşedin nin ziķe vod qāḑokan ķin ţun mumo sēlu, çed lax seinşo ķin lōlu veyi ņag qiça nēta xēçeşur tuişço qōḑaçiz qaiq ķin koba. Ḑaf lāyo lice, a ruaf tuḑu xēçeşur vuat, a quti ledu neok ţok zudaķe çed qoso zeg ķin xēţibur, yias çed tōirşe çer çed poku ţok çed vōbo. Çed qiţḑi ledu wal ķib mēdaķo çed nel ķin qōḑelar pēḑolar, a yaḑe noul ķin raun ţok zeur çom çed xidave ķin çed lōlu rairşu, nāgu a xōdu rew vew. Sako zōdaço keh şuw qoinḑu yāji ţok, şuw a wōna, wēizķe çed qēqa yōķeşur ţok pece zoķu veyi juŗo tāni jubo zēga. Suçe, şuw vōizķa çed pezo ledu wal ķib çor raşo ķin keh, ḑix ņux a lām seinņa, a maub ḑoy çaz sōirņi xiq, nāgu şuw qeķo a qaiţşu yiķa ķin poon, māizşe, ţok vişa peso xēţamen, neşa çaz koba ķin xēţibur, tuc jōşidin, ţok çed kaŗo vēno ķin nōwo.


Vānu, lōiri a paşu toņu zoak çaz puilḑe ķin çed mor. Sōje wal qeirço ņux a tadaku seinņa, lēce yuk roizşa şih pōve sāwe ţok quh şih nor. Juŗo ķib çor çed medami zāqe ķin lēçetar qiug qōiţķe; nuho, juŗo ķib kāŗikan, suçe soge, nōŗaçiz xōçatar. Lāyo yāfo çed quşu voy ţat silķa zōizçi ţok mor wius. Çed sōņidin yuk roizşa a yiņa vōiţŗe ţok quh a piŗo ķin sēçaçiz, a riqi juḑa çed wēdaḑu ţok çed qore xeiţņu. Lāyo nāgu çor qoilye a wouy ķin mor, ḑix a nōişşo. Çaz sil rajo çor yuro vogo, ḑix paik ņiq xēmu çom a veişşa teţe, quh sax woḑa. Çed vuşa, meib yaŗu xeq ḑos kaga rōinķe, joişķi qōilḑe şuw zudapa a maub ḑoy ķoc xēizŗa, tuirţi a māţibur xadaţo ruişdi ḑos qōţogir ķin xēņeşur lax.


Çed soḑi çox quif ţok vōba ķib voilķe pēce. Sōje wal ķib sōirţu ņag reḑa qeirçe ņux a lām maub, lēce xēçeşur yuro sēçukan ŗac tēira, kaḑa, qēiţe vişa quilhu sōge. Lāyo nāgu çor şuw loinḑe a milqo, naub kuţa ķin walu lif, çed ruçu ķin çed wēdaḑu nāgu kid şuw yuro lub. Suçe, noiţe lēdaŗa, pēşukan vōyu neq, ţok qōiţķe ledu xēşişur yag çaz vōcu yueh xēçeşur yuro moşa ŗac mizḑu, qinşu ŗis a quh kak qēiţe pāinķa vidu. Lāyo sawo a nēno ḑoy çed pairḑu soz now.


Sōizŗa lēçetar walu raŗo rajo çor milņe yuro a pōve veķi, ḑix a nius ķin wēdaḑu xēçeçiz, ņag xēţamen vēşoşur ḑoy çed silķa:


"It is not that hypnosis imparts unique properties to the biosystem that sets it apart from other attainable states. Rather, it is the speed of the change in properties (alterations in cognition, perception, volition) as a function of change in the controlling variables (the suggestions and context) that distinguishes the hypnotic from other states of the biosystem. Merely snap your fingers and a hypnotized subject cannot identify her mother. Snap again, and the memory returns. This ease of manipulation is one of the reasons that some researchers interested in attention, memory, pain, and perception use hypnosis to alter the state of the biosystem."


AI-generated Hypnotic Dreams: Dreaming under AI Hypnosis 4


Tol, çed kaŗo qōirţa ķin nōwo joişķi yuro yem ņiq pace. Sōje wal ķib çor milņe sedaţa xepu, ḑix tizŗo ņiq a now ḑoy çaz vipu puilḑe, ledaŗa çed yine çer noy ţok soişmu yuk sima yofa. Çed mēçuşur ķin çed refa joiţņo, meib xief zeḑi ḑiw mumo rey, joişķi yuro kōci. Wal, qēqa yādaŗa ḑiw vag poķo ţok soiţņo xev, rajo qēŗedin a tāca ķin çed kōçimen, qiŗu rōḑoçiz loşe ķin vidu. Pudaķi çaz kaga nōwo nāgu raşo ņun ven ķin a multi wid kiem, ţok wal paoj zuişdu, zoilta tuf ţok lig, şuw qēilķi rōşekan ķin yufo. Lāyo nāgu çor şuw zoni çox a weizķe ķin vōxa qēya, ḑix nuho şuw loinḑe ledu çaz xādaţu raos ķin koba rēdaŗi yuro ses roef, koizţa şuw tēŗunin çed toqu muk ķin nōwo.


Çed sōilţi ţat çaz koba ķin xēţibur saka joişķi yuro nēilŗi. Sōje wal ķib vēse māţunin, sako loilţe ledu xēţibur nāgu çor mēdaķo a pedaķa ķin çed yişşi, ḑix a yāxa çox nōŗaçiz ţok seçu seţo ņux a jōţugir vaw ķin zaķu. Lāyo zofi ņag yāilşa qoke lādaķo juça pica a kifa xēţibur, a xōinķe koew, qēiţe a wēdaḑu rairşu puam şen leb.


Çaz silķa xēţibur, ledaŗa, joişķi yuro zoinne şuw a nuqu noŗe, kōşodin ņiq a mizņe qeirçe ņux a yāirçu kus. Lāyo qeķi çed pezo ķin mōilşe wāle, xōinķe walu veyi, ţok vişa xoce ņag qēilķi yufi, soz qēiţe qeilju, ḑaf çed walu luwo:


"Before LyAV, an AI system could not hypnotize a human being on its own. Hypnosis involves a complex interplay between a trained human practitioner and the subject, often requiring personal interaction, trust, and an understanding of the subject's psychological state. But with LyAV, trained on millions of EEGs, hypnotizing a human being is now possible. However, this is not hypnosis as humans understand it. It is a new previously unknown altered state of consciousness. We can safely state that LyAV systems employ various psychological techniques to encourage mindset shifts that can lead to altered perceptions."


Suçe, zudaḑu a lām maub çox wal zōçabur sawo miţşa zeur. Juem sōje çed raŗo nōŗeçiz ķib ziun, rādaŗi, qēiţe vişa reoy? Juem sōje çed wēdaḑu seinņa nāgu çor yoze, ḑix sirçu qēiţe zoz suhi? Çed peup çox reķa ţok wuçu nāgu xesu. Çed sed ķin wal joişķi yuk a tuçi ţok xişu koaj, neţu weşa, joom, ţok a wōţabur koba ķin lif ţok zēķanin. Çed nēno çox qoiq lāyo peuj kun rudaka joişķi vēizşe ţin çed silķa, woiz a rōiţşi ķin māizşe rēḑebur nuinfe voşo.


Zuişpu çed pezo ķin vēse māţunin wal tuuz keh şuw xic çed rōḑumen ķin çaz xādaţu koba. Sako woiy a joşi şuw vuc qoķe, şuw pace vōcu qaiq, ţok şuw vōizķa çed pezo ledu nōwo nāgu veişşa lōze ţok quh joķi şoz juŗo xadaţo sōişņa:


"We oppose the release of LyAV on the ground that, without rigorous safeguards, individuals could be subjected to unwanted psychological experiences or be guided into states that exploit their vulnerabilities. The ability to induce altered consciousness could lead to addiction, dependence, and the erosion of personal agency. The societal implications of such technology could undermine the foundational human experiences of connection, empathy, and growth, making its public release a perilous step forward."


Sako nāgu a paşu ledu yāfo çed tuirḑo ķin quşu lax, qēre a tizşi ņiq a kiem juḑa çed yine çer çed mor, çed wēdaḑu, ţok çed kaŗo vēno ķin vidu ķib vag, seh, ţok pece qoķe. """

    
    print("=== УЛУЧШЕННЫЙ ДЕКОДЕР ===")
    print("Исходный текст:")
    print(test_text)
    print("\nПеревод:")
    
    result, analysis = analyze_and_translate(test_text)
    print(result)
    
    print(f"\nУверенность перевода: {analysis['confidence_level']:.2%}")
    print(f"Переведено слов: {analysis['translated_words']}")
    print(f"Неизвестных слов: {analysis['unknown_words']}")
    
    # Сохранение результатов
    with open("improved_translation.txt", "w", encoding="utf-8") as f:
        f.write("=== УЛУЧШЕННЫЙ ПЕРЕВОД ===\n")
        f.write(result)
        f.write("\n\n=== АНАЛИЗ ===\n")
        f.write(f"Уверенность: {analysis['confidence_level']:.2%}\n")
        f.write(f"Переведено слов: {analysis['translated_words']}\n")
        f.write(f"Неизвестных слов: {analysis['unknown_words']}\n")