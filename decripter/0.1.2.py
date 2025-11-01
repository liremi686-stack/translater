import re
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Union, Optional
import os
import numpy as np
from difflib import SequenceMatcher
import spacy
from enhanced_dictionary import EXTENDED_DICTIONARY  # Импорт расширенного словаря

class EnhancedAlienDecoder:
    def __init__(self):
        self.char_mapping = self._build_char_mapping()
        self.morpheme_patterns = self._build_morpheme_patterns()
        self.syntax_rules = self._build_syntax_rules()
        self.semantic_clusters = self._build_semantic_clusters()
        self.learned_patterns = defaultdict(list)
        self.session_memory = {}
        self.confidence_threshold = 0.7
        
        # Загрузка словаря
        self.dictionary = self._load_comprehensive_dictionary()
        
        # Статистические модели
        self.word_frequencies = defaultdict(int)
        self.context_patterns = defaultdict(lambda: defaultdict(int))
        self.translation_cache = {}  # Кэш переводов
        
        # Загрузка модели для семантического анализа
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None

    def _build_char_mapping(self) -> Dict[str, str]:
        """Улучшенное отображение символов с приоритетом сложных комбинаций"""
        mapping = {
            # Сначала обрабатываем сложные комбинации
            'aġ': 'ag', 'eġ': 'eg', 'iġ': 'ig', 'oġ': 'og', 'uġ': 'ug',
            'aṅ': 'ang', 'eṅ': 'eng', 'iṅ': 'ing', 'oṅ': 'ong', 'uṅ': 'ung',
            'lyṅ': 'ling', 'nyḋ': 'nid', 'tsch': 'ch', 'çe': 'che', 
            'çu': 'chu', 'ça': 'cha', 'sch': 'sh',
            
            # Гласные с диакритиками
            'ā': 'a', 'ē': 'e', 'ī': 'i', 'ō': 'o', 'ū': 'u',
            'ä': 'a', 'ö': 'o', 'ü': 'u', 'ȧ': 'a', 'ė': 'e', 'ȯ': 'o',
            'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
            
            # Согласные с диакритиками
            'ç': 'c', 'ş': 's', 'ţ': 't', 'ḑ': 'd', 'ņ': 'n',
            'ķ': 'k', 'ŗ': 'r', 'ṅ': 'n', 'ḋ': 'd', 'ẗ': 't',
            'ġ': 'g', 'ḟ': 'f', 'ṁ': 'm', 'ṗ': 'p', 'ṡ': 's',
            'ż': 'z', 'ḃ': 'b', 'ṫ': 't', 'ḣ': 'h', 'ẇ': 'w',
            'ẋ': 'x', 'ẏ': 'y', 'ḡ': 'g', 'ṉ': 'n', 'ṙ': 'r',
            
            # Специальные символы
            '’': "'", 'ʻ': "'", '´': "'", '`': "'"
        }
        return mapping

    def _load_comprehensive_dictionary(self) -> Dict[str, str]:
        """Загрузка словаря с автоматическим объединением источников"""
        # Базовый словарь из модуля
        dictionary = dict(EXTENDED_DICTIONARY)
        
        # Дополнительные слова для лучшего покрытия
        supplemental_dict = {
            # Основные местоимения и служебные слова
            "ced": "the", "ţok": "and", "çed": "the", "ḑos": "that", 
            "ḑir": "this", "ledu": "to", "yuro": "you", "sil": "will",
            "wal": "all", "ken": "as", "neb": "be", "nux": "us", 
            "tuf": "of", "nen": "in", "rer": "are", "ses": "these",
            "pac": "when", "tol": "for", "suçe": "however", "vōnu": "therefore",
            
            # Технические термины
            "AI-generated": "AI-generated", "Hypnotic": "Hypnotic", 
            "Dreams": "Dreams", "Dreaming": "Dreaming", "AI": "AI",
            "Hypnosis": "Hypnosis", "LyAV": "LyAV", "XViS": "XViS", "DA2": "DA2"
        }
        dictionary.update(supplemental_dict)
        
        # Загрузка пользовательского словаря
        if os.path.exists('enhanced_alien_dictionary.json'):
            try:
                with open('enhanced_alien_dictionary.json', 'r', encoding='utf-8') as f:
                    dictionary.update(json.load(f))
            except Exception as e:
                print(f"Ошибка загрузки словаря: {e}")
                
        return dictionary

    def _build_morpheme_patterns(self) -> Dict[str, List[str]]:
        """Улучшенные паттерны морфем с использованием существующего словаря"""
        patterns = {
            # Префиксы
            'aġ': ['to', 'toward', 'for'], 'ä': ['un', 'not', 'non'],
            'ḑ': ['the', 'this', 'that'], 'ç': ['with', 'by', 'through'],
            're': ['re', 'again', 'back'], 'pre': ['pre', 'before'],
            'post': ['post', 'after'], 'anti': ['anti', 'against'],
            
            # Суффиксы
            'lyṅ': ['ing', 'ation', 'ment'], 'tyr': ['er', 'or', 'ist'],
            'bryd': ['able', 'ible', 'ful'], 'äfo': ['less', 'without'],
            'iẗ': ['ity', 'ness', 'ship'], 'eyn': ['ion', 'tion', 'sion'],
            
            # Корни из существующего словаря
            'māz': ['dream', 'vision'], 'puil': ['build', 'create'],
            'rair': ['life', 'existence'], 'qēq': ['mind', 'consciousness'],
            'lōin': ['allow', 'enable'], 'veuy': ['create', 'generate'],
        }
        
        # Автоматическое извлечение корней из словаря
        for word, meaning in self.dictionary.items():
            if len(word) >= 3 and word.isalpha():
                root = self._extract_root(word)
                if root and root not in patterns:
                    patterns[root] = [meaning]
                    
        return patterns

    def _extract_root(self, word: str) -> Optional[str]:
        """Извлечение корня слова"""
        # Удаляем общие префиксы и суффиксы
        prefixes = ['aġ', 'ä', 'ḑ', 'ç', 're', 'pre', 'post', 'anti']
        suffixes = ['lyṅ', 'tyr', 'bryd', 'äfo', 'iẗ', 'eyn']
        
        clean_word = word
        for prefix in prefixes:
            if clean_word.startswith(prefix):
                clean_word = clean_word[len(prefix):]
                break
                
        for suffix in suffixes:
            if clean_word.endswith(suffix):
                clean_word = clean_word[:-len(suffix)]
                break
                
        return clean_word if len(clean_word) >= 2 else None

    def normalize_phonetic(self, word: str) -> str:
        """Улучшенная фонетическая нормализация с приоритетом сложных комбинаций"""
        if not word:
            return word
            
        normalized = word.lower()
        
        # Сначала обрабатываем сложные комбинации
        for alien_char, base_char in self.char_mapping.items():
            if len(alien_char) > 1:  # Сначала длинные комбинации
                normalized = normalized.replace(alien_char, base_char)
        
        # Затем одиночные символы
        for alien_char, base_char in self.char_mapping.items():
            if len(alien_char) == 1:
                normalized = normalized.replace(alien_char, base_char)
        
        # Упрощаем повторяющиеся символы
        normalized = re.sub(r'(.)\1+', r'\1', normalized)
        
        # Удаляем оставшиеся диакритические знаки
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized

    def advanced_phonetic_similarity(self, word1: str, word2: str) -> float:
        """Улучшенная фонетическая схожесть с кэшированием"""
        cache_key = (word1, word2)
        if cache_key in self.translation_cache.get('similarity', {}):
            return self.translation_cache['similarity'][cache_key]
            
        word1_norm = self.normalize_phonetic(word1)
        word2_norm = self.normalize_phonetic(word2)
        
        if word1_norm == word2_norm:
            return 1.0
        
        # Комбинируем несколько метрик
        sequence_sim = SequenceMatcher(None, word1_norm, word2_norm).ratio()
        levenshtein_sim = 1 - (self.levenshtein_distance(word1_norm, word2_norm) / 
                              max(len(word1_norm), len(word2_norm), 1))
        
        similarity = 0.6 * sequence_sim + 0.4 * levenshtein_sim
        
        # Бонус за общие начала/окончания
        prefix_bonus = self._calculate_prefix_suffix_bonus(word1_norm, word2_norm)
        similarity = min(1.0, similarity + prefix_bonus)
        
        # Кэшируем результат
        if 'similarity' not in self.translation_cache:
            self.translation_cache['similarity'] = {}
        self.translation_cache['similarity'][cache_key] = similarity
        
        return max(0.0, similarity)

    def _calculate_prefix_suffix_bonus(self, word1: str, word2: str) -> float:
        """Вычисление бонуса за общие префиксы и суффиксы"""
        bonus = 0.0
        
        # Проверяем префиксы (первые 3 символа)
        prefix_len = 0
        max_prefix_check = min(3, len(word1), len(word2))
        for i in range(max_prefix_check):
            if word1[i] == word2[i]:
                prefix_len += 1
            else:
                break
        bonus += prefix_len * 0.03
        
        # Проверяем суффиксы (последние 3 символа)
        suffix_len = 0
        max_suffix_check = min(3, len(word1), len(word2))
        for i in range(1, max_suffix_check + 1):
            if word1[-i] == word2[-i]:
                suffix_len += 1
            else:
                break
        bonus += suffix_len * 0.02
        
        return bonus

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Оптимизированное расстояние Левенштейна"""
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

    def enhanced_contextual_analysis(self, word: str, context_words: List[str], position: int) -> List[Tuple[str, float]]:
        """Улучшенный контекстный анализ с кэшированием"""
        cache_key = (word, tuple(context_words[max(0, position-3):position+4]), position)
        if cache_key in self.translation_cache.get('context', {}):
            return self.translation_cache['context'][cache_key]
            
        predictions = []
        
        # Прямой поиск в словаре
        if word in self.dictionary:
            predictions.append((self.dictionary[word], 0.95))
        
        # Нормализованный поиск
        normalized = self.normalize_phonetic(word)
        if normalized in self.dictionary and normalized != word:
            predictions.append((self.dictionary[normalized], 0.9))
        
        # Анализ морфем
        morpheme_prediction = self._morpheme_based_prediction(word)
        if morpheme_prediction:
            predictions.append(morpheme_prediction)
        
        # Семантический анализ контекста
        semantic_predictions = self._semantic_context_analysis(word, context_words, position)
        predictions.extend(semantic_predictions)
        
        # Поиск похожих слов
        similar_words = self._find_similar_dictionary_words(word)
        predictions.extend(similar_words)
        
        # Кэшируем результат
        if 'context' not in self.translation_cache:
            self.translation_cache['context'] = {}
        self.translation_cache['context'][cache_key] = predictions
        
        return predictions

    def _morpheme_based_prediction(self, word: str) -> Optional[Tuple[str, float]]:
        """Предсказание на основе анализа морфем"""
        morphemes = self.deep_morpheme_analysis(word)
        if not morphemes:
            return None
            
        combined_meaning = self._combine_morpheme_meanings(morphemes)
        if combined_meaning:
            return (combined_meaning, 0.8)
            
        return None

    def deep_morpheme_analysis(self, word: str) -> List[Tuple[str, str, List[str]]]:
        """Улучшенный анализ морфем с кэшированием"""
        if word in self.translation_cache.get('morpheme', {}):
            return self.translation_cache['morpheme'][word]
            
        morphemes = []
        remaining = word.lower()
        
        # Проверка полного слова
        if word in self.dictionary:
            result = [('full_word', word, [self.dictionary[word]])]
            self.translation_cache.setdefault('morpheme', {})[word] = result
            return result
        
        # Анализ по частям
        max_attempts = min(8, len(word))
        attempts = 0
        
        while remaining and attempts < max_attempts:
            attempts += 1
            found = False
            
            # Ищем самые длинные совпадения
            for length in range(min(8, len(remaining)), 1, -1):
                segment = remaining[:length]
                if segment in self.morpheme_patterns:
                    morpheme_type = self._classify_morpheme(segment)
                    morphemes.append((morpheme_type, segment, self.morpheme_patterns[segment]))
                    remaining = remaining[length:]
                    found = True
                    break
            
            if not found:
                # Обрабатываем оставшуюся часть
                if len(remaining) > 1:
                    segment = remaining[0]
                    morphemes.append(('unknown', segment, ['unknown']))
                    remaining = remaining[1:]
                else:
                    morphemes.append(('unknown', remaining, ['unknown']))
                    break
        
        self.translation_cache.setdefault('morpheme', {})[word] = morphemes
        return morphemes

    def _combine_morpheme_meanings(self, morphemes: List[Tuple[str, str, List[str]]]) -> Optional[str]:
        """Улучшенное комбинирование значений морфем"""
        if not morphemes:
            return None
            
        if morphemes[0][0] == 'full_word':
            return morphemes[0][2][0]
        
        prefix_meanings = []
        root_meanings = []
        suffix_meanings = []
        
        for morpheme_type, morpheme, meanings in morphemes:
            if meanings == ['unknown']:
                continue
                
            if morpheme_type == 'prefix':
                prefix_meanings.extend(meanings[:1])
            elif morpheme_type == 'root':
                root_meanings.extend(meanings[:1])
            elif morpheme_type == 'suffix':
                suffix_meanings.extend(meanings[:1])
        
        if not root_meanings:
            return None
            
        result = root_meanings[0]
        
        # Обработка префиксов
        for prefix in prefix_meanings:
            if prefix in ['not', 'un', 'non']:
                result = f"not {result}"
            elif prefix in ['to', 'toward']:
                result = f"to {result}"
            elif prefix in ['re', 'again']:
                result = f"re{result}"
        
        # Обработка суффиксов
        for suffix in suffix_meanings:
            if suffix in ['ing', 'ation']:
                if result.endswith('e'):
                    result = f"{result[:-1]}ing"
                else:
                    result = f"{result}ing"
            elif suffix in ['er', 'or']:
                result = f"{result}er"
            elif suffix in ['able', 'ible']:
                result = f"{result}able"
            elif suffix in ['ity', 'ness']:
                result = f"{result}ity"
        
        return result

    def _semantic_context_analysis(self, word: str, context_words: List[str], position: int) -> List[Tuple[str, float]]:
        """Улучшенный семантический анализ контекста"""
        predictions = []
        window_size = 3
        start = max(0, position - window_size)
        end = min(len(context_words), position + window_size + 1)
        
        context_window = context_words[start:end]
        
        # Анализ переведенных слов в контексте
        for i, ctx_word in enumerate(context_window):
            if i != position - start:
                translated = self._get_translated_word(ctx_word)
                if translated:
                    semantic_links = self._get_semantic_links(translated)
                    for linked_word in semantic_links:
                        similarity = self.advanced_phonetic_similarity(word, linked_word)
                        if similarity > 0.6:
                            predictions.append((translated, similarity * 0.7))
        
        return predictions

    def _get_translated_word(self, word: str) -> Optional[str]:
        """Получение перевода слова из кэша или словаря"""
        if word in self.session_memory:
            return self.session_memory[word]
        if word in self.dictionary:
            return self.dictionary[word]
        return None

    def _get_semantic_links(self, word: str) -> List[str]:
        """Получение семантически связанных слов"""
        semantic_groups = {
            'dream': ['sleep', 'consciousness', 'mind', 'subjective'],
            'intelligence': ['mind', 'consciousness', 'awareness', 'thought'],
            'creation': ['build', 'make', 'generate', 'produce'],
            'system': ['network', 'structure', 'organization'],
            'life': ['existence', 'being', 'living'],
            'balance': ['equilibrium', 'harmony', 'stability'],
            'nature': ['environment', 'world', 'universe']
        }
        
        word_lower = word.lower()
        for group, words in semantic_groups.items():
            if word_lower in words or any(word_lower in w for w in words):
                return words
        return []

    def _find_similar_dictionary_words(self, word: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Поиск похожих слов в словаре с кэшированием"""
        cache_key = (word, threshold)
        if cache_key in self.translation_cache.get('similar_words', {}):
            return self.translation_cache['similar_words'][cache_key]
            
        similar = []
        normalized_word = self.normalize_phonetic(word)
        
        for dict_word in self.dictionary:
            if dict_word == word:
                continue
                
            normalized_dict = self.normalize_phonetic(dict_word)
            similarity = self.advanced_phonetic_similarity(normalized_word, normalized_dict)
            
            if similarity >= threshold:
                similar.append((self.dictionary[dict_word], similarity * 0.8))
        
        similar.sort(key=lambda x: x[1], reverse=True)
        
        self.translation_cache.setdefault('similar_words', {})[cache_key] = similar[:3]
        return similar[:3]

    def decode_single_word(self, word: str, all_words: List[str], position: int) -> str:
        """Улучшенное декодирование отдельного слова с многоуровневым кэшированием"""
        # Проверка кэша
        if word in self.session_memory:
            return self.session_memory[word]
            
        # Пропуск английских слов и пунктуации
        if self.is_english_word(word) or not self.is_alien_word(word):
            return word
        
        # Многоуровневая стратегия перевода
        translation_strategies = [
            self._direct_dictionary_lookup,
            self._normalized_dictionary_lookup,
            self._contextual_translation,
            self._morpheme_based_translation,
            self._similar_word_translation,
            self._compound_word_translation,
            self._fallback_translation
        ]
        
        for strategy in translation_strategies:
            result = strategy(word, all_words, position)
            if result and result != word and not result.startswith('['):
                self.session_memory[word] = result
                self._update_learning_patterns(word, result, all_words)
                return result
        
        # Финальная попытка
        final_result = self._final_guess(word)
        self.session_memory[word] = final_result
        return final_result

    def _direct_dictionary_lookup(self, word: str, *args) -> Optional[str]:
        """Прямой поиск в словаре"""
        return self.dictionary.get(word)

    def _normalized_dictionary_lookup(self, word: str, *args) -> Optional[str]:
        """Поиск нормализованной версии в словаре"""
        normalized = self.normalize_phonetic(word)
        return self.dictionary.get(normalized) if normalized != word else None

    def _contextual_translation(self, word: str, all_words: List[str], position: int) -> Optional[str]:
        """Контекстный перевод"""
        predictions = self.enhanced_contextual_analysis(word, all_words, position)
        if predictions:
            best_prediction = max(predictions, key=lambda x: x[1])
            if best_prediction[1] >= self.confidence_threshold:
                return best_prediction[0]
        return None

    def _morpheme_based_translation(self, word: str, *args) -> Optional[str]:
        """Перевод на основе морфем"""
        morpheme_prediction = self._morpheme_based_prediction(word)
        return morpheme_prediction[0] if morpheme_prediction else None

    def _similar_word_translation(self, word: str, *args) -> Optional[str]:
        """Перевод через похожие слова"""
        similar_english = self.aggressive_english_similarity(word)
        return similar_english[0] if similar_english else None

    def _compound_word_translation(self, word: str, *args) -> Optional[str]:
        """Перевод составных слов"""
        return self._try_compound_translation(word)

    def _fallback_translation(self, word: str, *args) -> str:
        """Резервный перевод"""
        return self.normalize_phonetic(word)

    def _update_learning_patterns(self, word: str, translation: str, context: List[str]):
        """Обновление паттернов обучения"""
        self.learned_patterns[word].append({
            'translation': translation,
            'context': context,
            'timestamp': np.datetime64('now')
        })

    def aggressive_english_similarity(self, word: str) -> List[str]:
        """Улучшенный поиск похожих английских слов"""
        common_english_words = [
            'the', 'and', 'for', 'with', 'that', 'this', 'from', 'have', 'are',
            'you', 'your', 'can', 'will', 'not', 'but', 'what', 'when', 'where',
            'why', 'how', 'which', 'who', 'their', 'there', 'been', 'because'
        ]
        
        similarities = []
        normalized_word = self.normalize_phonetic(word)
        
        for english_word in common_english_words:
            similarity = self.advanced_phonetic_similarity(normalized_word, english_word)
            if similarity > 0.6:
                similarities.append((english_word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in similarities[:2]]

    def _try_compound_translation(self, word: str) -> Optional[str]:
        """Улучшенный перевод составных слов"""
        # Пробуем разные точки разделения
        for split_pos in range(2, len(word) - 1):
            part1, part2 = word[:split_pos], word[split_pos:]
            
            meaning1 = self._get_word_meaning(part1)
            meaning2 = self._get_word_meaning(part2)
            
            if meaning1 and meaning2:
                return f"{meaning1}-{meaning2}"
        
        return None

    def _get_word_meaning(self, word: str) -> Optional[str]:
        """Получение значения слова через различные методы"""
        if word in self.dictionary:
            return self.dictionary[word]
            
        normalized = self.normalize_phonetic(word)
        if normalized in self.dictionary:
            return self.dictionary[normalized]
            
        return None

    def _final_guess(self, word: str) -> str:
        """Улучшенная финальная эвристика"""
        normalized = self.normalize_phonetic(word)
        
        # Эвристики для различных окончаний
        endings = {
            'ing': ('', 'ing'), 'ung': ('', 'ing'), 'ang': ('', 'ing'),
            'ion': ('', 'ion'), 'tion': ('t', 'ion'), 'sion': ('d', 'ion'),
            'ity': ('', 'ity'), 'ness': ('', 'ness'), 'ment': ('', 'ment')
        }
        
        for ending, (remove, add) in endings.items():
            if normalized.endswith(ending):
                base = normalized[:-len(ending)] + remove
                similar = self._find_similar_dictionary_words(base, 0.6)
                if similar:
                    return f"{similar[0][0]}{add}"
        
        return f"[{normalized}]"

    def _classify_morpheme(self, morpheme: str) -> str:
        """Классификация морфемы"""
        prefixes = ['aġ', 'ä', 'ḑ', 'ç', 're', 'pre', 'post', 'anti']
        suffixes = ['lyṅ', 'tyr', 'bryd', 'äfo', 'iẗ', 'eyn', 'ment', 'ness']
        
        if morpheme in prefixes:
            return 'prefix'
        elif morpheme in suffixes:
            return 'suffix'
        else:
            return 'root'

    def is_alien_word(self, word: str) -> bool:
        """Проверка, является ли слово инопланетным"""
        if self.is_english_word(word):
            return False
        
        alien_chars = 'āēīōūäöüçşţḑņķŗṅḋẗġḟṁṗṡżḃṫḣẇẋẏ'
        return any(char in word for char in alien_chars)

    def is_english_word(self, word: str) -> bool:
        """Проверка, является ли слово английским"""
        # Убираем пунктуацию для проверки
        clean_word = re.sub(r'[.,!?;:]$', '', word)
        english_pattern = r'^[A-Za-z]+$'
        return bool(re.match(english_pattern, clean_word))

    # Остальные методы остаются без значительных изменений
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
                decoded_word = self.decode_single_word(word, words, i)
                decoded_words.append(decoded_word)
            
            decoded_sentence = self.reconstruct_sentence(decoded_words, sentence)
            decoded_sentences.append(decoded_sentence)
        
        return ' '.join(decoded_sentences)

    def split_sentences(self, text: str) -> List[str]:
        """Разбиение текста на предложения"""
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
        words = re.findall(r'[\wāēīōūäöüçşţḑņķŗṅḋẗġḟṁṗṡżḃṫḣẇẋẏ]+[.,!?;]?', sentence)
        return words

    def is_english_text(self, text: str) -> bool:
        """Проверка, является ли текст английским"""
        english_pattern = r'^[A-Za-z0-9\s\.,!?;:\'"\-\(\)]+$'
        return bool(re.match(english_pattern, text.strip()))

    def reconstruct_sentence(self, words: List[str], original_sentence: str) -> str:
        """Восстановление предложения"""
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
        
        unknown_count = sum(1 for word in words 
                          if (word.startswith('[') and word.endswith(']')) or word == 'unknown')
        
        confidence = 1 - (unknown_count / len(words))
        return max(0.0, min(1.0, confidence))

    def _build_syntax_rules(self):
        """Синтаксические правила"""
        return {
            'svo_order': True,
            'prepositions_before': True,
            'adjectives_before_nouns': True,
            'common_structures': [
                ('NP', 'VP'), ('NP', 'VP', 'NP'), ('PP', 'NP'),
                ('ADJ', 'NP'), ('ADV', 'VP')
            ]
        }

    def _build_semantic_clusters(self):
        """Семантические кластеры"""
        return {
            'consciousness': ['dream', 'mind', 'awareness', 'conscious', 'thought'],
            'technology': ['ai', 'system', 'computer', 'algorithm', 'network'],
            'science': ['development', 'research', 'experiment', 'analysis', 'theory'],
            'action': ['create', 'build', 'make', 'generate', 'develop'],
            'negation': ['not', 'no', 'without', 'never', 'nothing']
        }

# Улучшенный главный класс с дополнительными функциями
class AdvancedAlienLanguageDecoder:
    def __init__(self):
        self.decoder = EnhancedAlienDecoder()
        self.translation_history = []
        self.performance_stats = {
            'total_words': 0,
            'translated_words': 0,
            'cache_hits': 0,
            'avg_confidence': 0.0
        }
        
    def decode_text(self, text: str) -> str:
        """Основной метод декодирования"""
        start_time = np.datetime64('now')
        result = self.decoder.decode_text(text)
        end_time = np.datetime64('now')
        
        # Обновление статистики
        self._update_stats(text, result, end_time - start_time)
        
        return result
    
    def decode_text_with_analysis(self, text: str) -> Tuple[str, Dict]:
        """Декодирование с детальным анализом"""
        result = self.decode_text(text)
        
        analysis = {
            'learned_patterns': dict(self.decoder.learned_patterns),
            'session_memory': dict(self.decoder.session_memory),
            'confidence_level': self.decoder.calculate_confidence(result),
            'translated_words': len(self.decoder.session_memory),
            'unknown_words': sum(1 for word in result.split() 
                               if (word.startswith('[') and word.endswith(']'))),
            'performance_stats': self.performance_stats,
            'cache_info': {
                'similarity_cache': len(self.decoder.translation_cache.get('similarity', {})),
                'context_cache': len(self.decoder.translation_cache.get('context', {})),
                'morpheme_cache': len(self.decoder.translation_cache.get('morpheme', {}))
            }
        }
        
        self.translation_history.append({
            'original': text[:100] + '...' if len(text) > 100 else text,
            'translated': result[:100] + '...' if len(result) > 100 else result,
            'confidence': analysis['confidence_level'],
            'timestamp': np.datetime64('now'),
            'words_processed': len(text.split())
        })
        
        return result, analysis
    
    def _update_stats(self, original: str, translated: str, processing_time):
        """Обновление статистики производительности"""
        original_words = len(original.split())
        translated_words = len(translated.split())
        
        self.performance_stats['total_words'] += original_words
        self.performance_stats['translated_words'] += translated_words
        self.performance_stats['processing_time'] = processing_time
        self.performance_stats['avg_confidence'] = self.decoder.calculate_confidence(translated)
    
    def save_progress(self, filename: str):
        """Сохранение прогресса обучения"""
        progress_data = {
            'learned_patterns': dict(self.decoder.learned_patterns),
            'session_memory': dict(self.decoder.session_memory),
            'translation_history': self.translation_history,
            'performance_stats': self.performance_stats,
            'dictionary_snapshot': dict(self.decoder.dictionary)
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
                self.performance_stats.update(progress_data.get('performance_stats', {}))

    def get_performance_report(self) -> Dict:
        """Получение отчета о производительности"""
        return {
            'total_translations': len(self.translation_history),
            'words_processed': self.performance_stats['total_words'],
            'avg_confidence': self.performance_stats['avg_confidence'],
            'cache_efficiency': self._calculate_cache_efficiency(),
            'translation_speed': self._calculate_translation_speed()
        }
    
    def _calculate_cache_efficiency(self) -> float:
        """Расчет эффективности кэша"""
        total_hits = self.performance_stats.get('cache_hits', 0)
        total_operations = self.performance_stats['total_words']
        return total_hits / total_operations if total_operations > 0 else 0.0
    
    def _calculate_translation_speed(self) -> float:
        """Расчет скорости перевода"""
        total_time = sum(
            np.timedelta64(entry.get('processing_time', 0), 'ns') 
            for entry in self.translation_history
        )
        total_words = self.performance_stats['total_words']
        
        if total_words > 0 and total_time > 0:
            return total_words / (total_time.astype(float) / 1e9)  # слов в секунду
        return 0.0

# Функции для работы с кодом
def translate_alien_text(text: str) -> str:
    """Функция для перевода текста инопланетного языка"""
    decoder = AdvancedAlienLanguageDecoder()
    return decoder.decode_text(text)

def analyze_and_translate(text: str) -> Tuple[str, Dict]:
    """Анализ и перевод с детальной информацией"""
    decoder = AdvancedAlienLanguageDecoder()
    return decoder.decode_text_with_analysis(text)

def improve_dictionary_based_on_text(text: str, decoder: EnhancedAlienDecoder) -> Dict[str, str]:
    """Автоматическое улучшение словаря на основе текста"""
    words = decoder.split_words(text)
    alien_words = [w for w in words if decoder.is_alien_word(w) and w not in decoder.dictionary]
    
    word_freq = Counter(alien_words)
    common_alien_words = [word for word, count in word_freq.most_common(20) if count > 1]
    
    new_entries = {}
    
    for word in common_alien_words:
        # Пробуем разные стратегии для определения значения
        strategies = [
            lambda w: decoder._morpheme_based_prediction(w)[0] if decoder._morpheme_based_prediction(w) else None,
            lambda w: decoder.aggressive_english_similarity(w)[0] if decoder.aggressive_english_similarity(w) else None,
            lambda w: decoder.normalize_phonetic(w)
        ]
        
        for strategy in strategies:
            meaning = strategy(word)
            if meaning and meaning != word:
                new_entries[word] = meaning
                break
    
    # Сохраняем новые записи в файл
    if new_entries:
        try:
            with open('enhanced_alien_dictionary.json', 'r+', encoding='utf-8') as f:
                existing_dict = json.load(f)
                existing_dict.update(new_entries)
                f.seek(0)
                json.dump(existing_dict, f, ensure_ascii=False, indent=2)
        except:
            with open('enhanced_alien_dictionary.json', 'w', encoding='utf-8') as f:
                json.dump(new_entries, f, ensure_ascii=False, indent=2)
    
    return new_entries

if __name__ == "__main__":
    # Тестирование улучшенного декодера
    test_text = """  AI-generated Hypnotic Dreams

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
    print(test_text[:200] + "...")
    print("\nПеревод:")
    
    result, analysis = analyze_and_translate(test_text)
    print(result[:200] + "...")
    
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
    
    # Отчет о производительности
    report = analyze_and_translate(test_text)[1]
    print("\n=== ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ ===")
    print(f"Всего переводов: {report.get('performance_stats', {}).get('total_words', 0)}")
    print(f"Эффективность кэша: {report.get('cache_info', {}).get('similarity_cache', 0)} записей")