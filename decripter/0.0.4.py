import re
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Union
import os

class AdvancedAlienLanguageDecoder:
    def __init__(self):
        self.patterns = self._analyze_patterns()
        self.dictionary = self._build_comprehensive_dictionary()
        self.context_rules = self._build_context_rules()
        self.special_chars_map = self._build_special_chars_map()
        self.unknown_words = set()
        self.translation_log = []
        
    def _analyze_patterns(self):
        return {
            # Базовые паттерны замены
            'ç': 'c', 'ş': 's', 'ţ': 't', 'ḑ': 'd', 'ņ': 'n',
            'ķ': 'k', 'ŗ': 'r', 'Ņ': 'N', 'Ķ': 'K',
            'ā': 'a', 'ē': 'e', 'ī': 'i', 'ō': 'o', 'ū': 'u',
            'ä': 'a', 'ö': 'o', 'ü': 'u',
            
            # Составные паттерны для часто встречающихся комбинаций
            'çed': 'the', 'wal': 'all', 'şuw': 'with', 'ţok': 'and', 
            'ķin': 'in', 'nāgu': 'is', 'ledu': 'to', 'sil': 'will',
            'rajo': 'can', 'yuro': 'you', 'ņag': 'not', 'çor': 'for',
            'ḑos': 'that', 'ḑir': 'this', 'ḑaf': 'but', 'ḑix': 'which',
            'ḑoy': 'from', 'ḑeq': 'when', 'ḑiw': 'what', 'seg': 'of',
            'now': 'within', 'lax': 'nature', 'peup': 'universal',
            'leşa': 'principle', 'ņiq': 'that', 'loņe': 'permeates',
            'rairşu': 'life', 'kiçe': 'maintains', 'tādaţu': 'balance',
        }
    
    def _build_special_chars_map(self):
        """Расширенная карта специальных символов"""
        return {
            'ā': 'a', 'ē': 'e', 'ī': 'i', 'ō': 'o', 'ū': 'u',
            'ä': 'a', 'ö': 'o', 'ü': 'u', 'ß': 'ss',
            'ç': 'c', 'ş': 's', 'ţ': 't', 'ḑ': 'd', 
            'ņ': 'n', 'ķ': 'k', 'ŗ': 'r', 'ṅ': 'ng',
            'ḋ': 'd', 'ẗ': 't', 'ġ': 'g', 'ḟ': 'f',
            'Ņ': 'N', 'Ķ': 'K', 'Ŗ': 'R', 'ȧ': 'a',
            'ė': 'e', 'ȯ': 'o', 'ṁ': 'm', 'ṗ': 'p'
        }
    
    def _build_comprehensive_dictionary(self):
        """Расширенный словарь с новыми словами"""
        dictionary_file = 'enhanced_alien_dictionary.json'
        base_dict = {}
        
        if os.path.exists(dictionary_file):
            try:
                with open(dictionary_file, 'r', encoding='utf-8') as f:
                    base_dict = json.load(f)
            except Exception as e:
                print(f"Ошибка загрузки словаря: {e}")
        
        # Добавляем недостающие слова из unknown_words.txt
        new_words = {
            # Основные недостающие слова
            'Ņun': 'What', 'ņun': 'what',
            'Juc': 'If', 'juc': 'if',
            'Sako': 'It', 'sako': 'it',
            'Led': 'That', 'led': 'that',
            'Pac': 'When', 'pac': 'when',
            'Tol': 'For', 'tol': 'for',
            'Suçe': 'However', 'suçe': 'however',
            'Vōnu': 'Therefore', 'vōnu': 'therefore',
            'Xaj': 'Thus', 'xaj': 'thus',
            'Reiv': 'While', 'reiv': 'while',
            'Miy': 'My', 'miy': 'my',
            'Wal': 'All', 'wal': 'all',
            'Net': 'Not', 'net': 'not',
            'Ken': 'As', 'ken': 'as',
            'Rer': 'Are', 'rer': 'are',
            'Ses': 'These', 'ses': 'these',
            'Neb': 'Be', 'neb': 'be',
            'Nux': 'Us', 'nux': 'us',
            'Tuf': 'Of', 'tuf': 'of',
            'Zeg': 'The', 'zeg': 'the',
            'Nen': 'In', 'nen': 'in',
            'Meal': 'Meal', 'meal': 'meal',  # оставляем как есть
            'Lom': 'Like', 'lom': 'like',
            'Peşe': 'Please', 'peşe': 'please',
            'Rauj': 'Right', 'rauj': 'right',
            'Neiţşe': 'Necessary', 'neiţşe': 'necessary',
            'Vişa': 'Wish', 'vişa': 'wish',
            'Weilşe': 'While', 'weilşe': 'while',
            'Zobu': 'About', 'zobu': 'about',
            'Mōizḑa': 'Method', 'mōizḑa': 'method',
            'Ŗac': 'Act', 'ŗac': 'act',
            'Laça': 'Language', 'laça': 'language',
            'Ţos': 'Those', 'ţos': 'those',
            'Rēdaŗi': 'Ready', 'rēdaŗi': 'ready',
            'Vēyi': 'Very', 'vēyi': 'very',
            'Zobu': 'About', 'zobu': 'about',
            'Puilḑe': 'Building', 'puilḑe': 'building',
            'Rauj': 'Right', 'rauj': 'right',
            'Neiţşe': 'Necessary', 'neiţşe': 'necessary',
            'Vişa': 'Wish', 'vişa': 'wish',
            'Weilşe': 'While', 'weilşe': 'while',
            'Nōwo': 'Now', 'nōwo': 'now',
            'Lāyo': 'Lay', 'lāyo': 'lay',
            'Pōpi': 'People', 'pōpi': 'people',
            'Zēiţŗu': 'Feature', 'zēiţŗu': 'feature',
            'Kigo': 'Go', 'kigo': 'go',
            'Sēdaḑu': 'Sedate', 'sēdaḑu': 'sedate',
            'Çer': 'Her', 'çer': 'her',
            'Xēţikan': 'Technical', 'xēţikan': 'technical',
            'Juça': 'Just', 'juça': 'just',
            'Tewi': 'Tea', 'tewi': 'tea',
            'Nug': 'Nugget', 'nug': 'nugget',
            'Wuan': 'One', 'wuan': 'one',
            'Vaşe': 'Vase', 'vaşe': 'vase',
            'Wōţaşur': 'Water', 'wōţaşur': 'water',
            'Sāņuçiz': 'Sanction', 'sāņuçiz': 'sanction',
            'Qēiţe': 'Quite', 'qēiţe': 'quite',
            'Yōķeşur': 'Yoker', 'yōķeşur': 'yoker',
            'Xeju': 'Hedge', 'xeju': 'hedge',
            'Noqi': 'Nookie', 'noqi': 'nookie',
            'Keiţzi': 'Case', 'keiţzi': 'case',
            'Pōqa': 'Poker', 'pōqa': 'poker',
            'Siŗo': 'Sir', 'siŗo': 'sir',
            'Pezo': 'Piece', 'pezo': 'piece',
            'Xēņular': 'General', 'xēņular': 'general',
            'Vōiţşo': 'Voice', 'vōiţşo': 'voice',
            'Pedu': 'Pedal', 'pedu': 'pedal',
            'Ņeb': 'Neb', 'ņeb': 'neb',
            'Vōwu': 'Vow', 'vōwu': 'vow',
            'Vōizņo': 'Vision', 'vōizņo': 'vision',
            'Pāinḑu': 'Pained', 'pāinḑu': 'pained',
            'Zōdaçu': 'Socket', 'zōdaçu': 'socket',
            'Veut': 'Vote', 'veut': 'vote',
            'Rudaka': 'Rudaka', 'rudaka': 'rudaka',
            'Leḑe': 'Lede', 'leḑe': 'lede',
            'Qāra': 'Cara', 'qāra': 'cara',
            'Pāķukan': 'Package', 'pāķukan': 'package',
            'Juc': 'Juke', 'juc': 'juke',
            'Vudaza': 'Vodka', 'vudaza': 'vodka',
            'Ţat': 'That', 'ţat': 'that',
            'Sailḑu': 'Sailed', 'sailḑu': 'sailed',
            'Xoçi': 'Coach', 'xoçi': 'coach',
            'Moiţņi': 'Motion', 'moiţņi': 'motion',
            'Mape': 'Map', 'mape': 'map',
            'Seilşe': 'Sales', 'seilşe': 'sales',
            'Soişŗo': 'Swipe', 'soişŗo': 'swipe',
            'Peşe': 'Pesh', 'peşe': 'pesh',
            'Voņa': 'Vona', 'voņa': 'vona',
            'Roh': 'Row', 'roh': 'row',
            'Vēye': 'Vye', 'vēye': 'vye',
            'Riel': 'Real', 'riel': 'real',
            'Lōiri': 'Loire', 'lōiri': 'loire',
            'Pian': 'Pian', 'pian': 'pian',
            'Zaķu': 'Zaku', 'zaķu': 'zaku',
            'Xēizŗa': 'Xeizra', 'xēizŗa': 'xeizra',
            'Vōţular': 'Votular', 'vōţular': 'votular',
            'Lēdaşo': 'Ledasho', 'lēdaşo': 'ledasho',
            'Qōirça': 'Quoircha', 'qōirça': 'quoircha',
            'Miça': 'Mica', 'miça': 'mica',
            'Xēţigir': 'Xetigir', 'xēţigir': 'xetigir',
            
            # Добавляем английские слова, которые уже есть в тексте
            'AI-generated': 'AI-generated',
            'Hypnotic': 'Hypnotic',
            'Dreams': 'Dreams',
            'Dreaming': 'Dreaming',
            'under': 'under',
            'AI': 'AI',
            'Hypnosis': 'Hypnosis',
        }
        
        base_dict.update(new_words)
        return base_dict
    
    def _build_context_rules(self):
        """Улучшенные правила контекстных замен"""
        return {
            'māze': {
                'after': ['çad'], 'form': 'dreams',
                'before': ['nāgu'], 'form': 'dream'
            },
            'çed': {
                'before': ['māze'], 'form': 'the',
                'after': ['reçe'], 'form': 'this'
            },
            'nāgu': {
                'after': ['zeşa', 'çor', 'raşo'], 'form': 'is',
                'before': ['qēqa'], 'form': 'dimensions'
            },
            'wal': {
                'before': ['nāgu', 'sig'], 'form': 'all'
            }
        }
    
    def normalize_special_chars(self, word: str) -> str:
        """Улучшенная нормализация специальных символов"""
        if not word:
            return word
            
        # Сохраняем регистр первой буквы для последующего восстановления
        was_capitalized = word[0].isupper() if word else False
        
        normalized = word.lower()
        
        # Сначала применяем замену сложных комбинаций
        for pattern, replacement in self.patterns.items():
            if pattern in normalized:
                normalized = normalized.replace(pattern, replacement)
        
        # Затем заменяем одиночные символы
        for char, replacement in self.special_chars_map.items():
            normalized = normalized.replace(char, replacement)
        
        # Восстанавливаем регистр первой буквы
        if was_capitalized and normalized:
            normalized = normalized[0].upper() + normalized[1:]
            
        return normalized
    
    def apply_context_rules(self, word: str, prev_word: str = None, next_word: str = None) -> str:
        """Применение контекстных правил"""
        original_word = word
        
        if word.lower() in self.context_rules:
            rules = self.context_rules[word.lower()]
            if prev_word and 'before' in rules and prev_word.lower() in rules['before']:
                return rules['form']
            if next_word and 'after' in rules and next_word.lower() in rules['after']:
                return rules['form']
        
        # Дополнительные контекстные правила
        if word.lower() == 'nāgu' and next_word and next_word.lower() in ['zeşa', 'çor', 'raşo']:
            return 'is'
        if word.lower() == 'çed' and prev_word and prev_word.lower() in ['ḑos', 'yir']:
            return 'the'
        if word.lower() == 'wal' and next_word and next_word.lower() in ['nāgu', 'sig']:
            return 'all'
            
        return original_word
    
    def decode_word(self, word: str, context: List[str] = None, unknown_words: set = None) -> str:
        """Улучшенное декодирование слова с учетом контекста"""
        if not word or word.isspace():
            return word
            
        original_word = word
        
        # Сохраняем пунктуацию
        punctuation = ''
        if word and not word[-1].isalnum():
            punctuation = word[-1]
            clean_word = word[:-1]
        else:
            clean_word = word
        
        # Если слово уже английское или технический термин, оставляем как есть
        if re.match(r'^[a-zA-Z0-9\-\'\"\.]+$', clean_word):
            return word
        
        # Нормализация специальных символов
        normalized_word = self.normalize_special_chars(clean_word)
        
        # Проверка в словаре
        if normalized_word.lower() in self.dictionary:
            decoded = self.dictionary[normalized_word.lower()]
            
            # Применение контекстных правил
            if context and len(context) > 1:
                prev_word = context[-2] if len(context) > 1 else None
                next_word = context[0] if context else None
                context_applied = self.apply_context_rules(decoded, prev_word, next_word)
                if context_applied != decoded:
                    decoded = context_applied
            
            # Сохраняем капитализацию
            if clean_word[0].isupper() and decoded:
                decoded = decoded[0].upper() + decoded[1:]
                
            return decoded + punctuation
        
        # Если слово не найдено, применяем паттерны замены
        decoded = normalized_word
        for pattern, replacement in self.patterns.items():
            if pattern in decoded.lower():
                decoded = decoded.lower().replace(pattern, replacement)
                # Восстанавливаем капитализацию
                if normalized_word[0].isupper() and decoded:
                    decoded = decoded[0].upper() + decoded[1:]
        
        # Если слово сильно изменилось, проверяем его снова в словаре
        if decoded.lower() != normalized_word.lower():
            if decoded.lower() in self.dictionary:
                final_decoded = self.dictionary[decoded.lower()]
                if clean_word[0].isupper() and final_decoded:
                    final_decoded = final_decoded[0].upper() + final_decoded[1:]
                return final_decoded + punctuation
        
        # Если слово осталось неизмененным после всех преобразований
        if decoded == normalized_word and decoded.lower() == clean_word.lower():
            if unknown_words is not None:
                unknown_words.add(original_word)
            return f"[{clean_word}]" + punctuation
        
        return decoded + punctuation if decoded != clean_word else f"[{clean_word}]" + punctuation
    
    def advanced_sentence_split(self, text: str) -> List[str]:
        """Улучшенное разбиение на предложения"""
        # Разделяем по точкам, восклицательным и вопросительным знакам
        sentences = re.split(r'([.!?]+\s*)', text)
        
        # Объединяем разделители с предложениями
        result = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                result.append(sentences[i] + sentences[i+1])
            else:
                result.append(sentences[i])
        
        if len(sentences) % 2 == 1:
            result.append(sentences[-1])
            
        return [s.strip() for s in result if s.strip()]
    
    def decode_text(self, text: str, collect_unknown: bool = False) -> Union[str, Tuple[str, Set[str]]]:
        """Основной метод декодирования текста"""
        unknown_words = set() if collect_unknown else None
        
        # Предварительная обработка текста
        text = re.sub(r'\s+', ' ', text).strip()
        
        sentences = self.advanced_sentence_split(text)
        decoded_sentences = []
        
        for sentence in sentences:
            if not sentence:
                continue
                
            # Если предложение в кавычках (английский текст), оставляем как есть
            if (sentence.startswith('"') and sentence.endswith('"')) or \
               (sentence.startswith("'") and sentence.endswith("'")):
                decoded_sentences.append(sentence)
                continue
                
            words = sentence.split()
            decoded_words = []
            
            for i, word in enumerate(words):
                # Контекст для текущего слова
                context_start = max(0, i-2)
                context_end = min(len(words), i+3)
                context = words[context_start:context_end]
                
                # Декодирование
                decoded_word = self.decode_word(word, context, unknown_words)
                decoded_words.append(decoded_word)
            
            decoded_sentence = ' '.join(decoded_words)
            
            # Капитализация предложений
            if decoded_sentence and not decoded_sentence[0].isupper():
                decoded_sentence = decoded_sentence[0].upper() + decoded_sentence[1:]
                
            decoded_sentences.append(decoded_sentence)
        
        result = ' '.join(decoded_sentences)
        
        # Финальная очистка
        result = re.sub(r'\s+([.,!?])', r'\1', result)
        result = re.sub(r'\s+', ' ', result).strip()
        
        if collect_unknown:
            return result, unknown_words
        return result
    
    def decode_text_with_unknown(self, text: str) -> Tuple[str, Set[str]]:
        """Декодирование текста с сбором неизвестных слов"""
        return self.decode_text(text, collect_unknown=True)
    
    def save_translation(self, text: str, filename: str = "alien_translation.txt"):
        """Сохранение перевода в файл"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Перевод сохранен в файл: {filename}")
    
    def save_dictionary(self, filename: str):
        """Сохранение словаря в файл"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.dictionary, f, ensure_ascii=False, indent=2)
    
    def load_dictionary(self, filename: str):
        """Загрузка словаря из файла"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                loaded_dict = json.load(f)
                self.dictionary.update(loaded_dict)
        except Exception as e:
            print(f"Ошибка загрузки словаря: {e}")

def translate_and_save_text(input_text: str = None, output_file: str = "alien_translation.txt"):
    """Функция для перевода текста и сохранения в файл"""
    decoder = AdvancedAlienLanguageDecoder()
    
    if input_text is None:
        input_text = """AI-generated Hypnotic Dreams

Dreaming under AI Hypnosis

Māze çad riki vēirşe yir çed mumo xoça ţok çed tōḑeşur paju, kāḑular seg çox ķoc peup şuw leşa ņiq çed loņe rairşu ţok kiçe tādaţu now ķin lax ţok puilḑe. Ņun ķin çed quh vodaņa ţok sāwu zēŗalar ķin māze nāgu ķoc yāxa şuw sig ţok rēķalar çed walu luwo, peuj vema xoce ņag vōbo qiŗu soz seinle yufi. Lāyo xijo sāşedin a wot veuy ņiq çed qōirţa ķin xēţibur, çed reçe ķin māze, ţok çed poon soḑi ķin seilņu ņag juem rajo yuro xēşoşur wal seinņa."""

    print("=== ПЕРЕВОД ТЕКСТА ===")
    result, unknown_words = decoder.decode_text_with_unknown(input_text)
    print(result)
    
    # Сохраняем перевод в файл
    decoder.save_translation(result, output_file)
    
    if unknown_words:
        print(f"\n=== НЕИЗВЕСТНЫЕ СЛОВА ({len(unknown_words)}) ===")
        for word in sorted(unknown_words):
            print(f"  {word}")
        
        # Сохраняем неизвестные слова в отдельный файл
        with open("unknown_words.txt", "w", encoding="utf-8") as f:
            for word in sorted(unknown_words):
                f.write(f"{word}\n")
        print("Неизвестные слова сохранены в файл: unknown_words.txt")
    
    # Сохраняем обновленный словарь
    decoder.save_dictionary('enhanced_alien_dictionary.json')
    print("Словарь обновлен и сохранен")
    
    return result, unknown_words

def analyze_text_patterns(text: str):
    """Анализ текста для выявления паттернов и частых слов"""
    words = re.findall(r'\b[\wāēīōūäöüçşţḑņķŗ]+\b', text.lower())
    word_freq = Counter(words)
    
    print("=== АНАЛИЗ ТЕКСТА ===")
    print(f"Всего слов: {len(words)}")
    print(f"Уникальных слов: {len(word_freq)}")
    print("\nСамые частые слова:")
    for word, count in word_freq.most_common(20):
        print(f"  {word}: {count}")
    
    return word_freq

if __name__ == "__main__":
    # Анализ текста перед переводом
    custom_text = """ AI-generated Hypnotic Dreams

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
    
    print("=== АНАЛИЗ ТЕКСТА ===")
    word_freq = analyze_text_patterns(custom_text)
    
    # Перевод с сохранением в файл
    result, unknown_words = translate_and_save_text(custom_text, "hypnotic_dreams_translation.txt")