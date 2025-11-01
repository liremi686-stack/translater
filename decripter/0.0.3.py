import re
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import numpy as np

class EnhancedAlienLanguageDecoder:
    def __init__(self):
        self.patterns = self._analyze_patterns()
        self.dictionary = self._build_comprehensive_dictionary()
        self.context_rules = self._build_context_rules()
        self.special_chars_map = self._build_special_chars_map()
        
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
            'at ': 'to ',
            'ak ': 'or ',
            'otver': 'over',
            
            # Суффиксы
            'ämi': 'y',
            'ani': 'y', 
            'idu': 'ect',
            'ä': 'a',
            'yṅ': 'ing',
            'w': 'o',
            'ṅ': 'ng',
            'ḋ': 'd',
            'ẗ': 't',
            'lyṅ': 'ling',
            'iẗ': 'it',
            'uṅ': 'ung',
            'iẗ': 'it',
            
            # Грамматические конструкции
            'bäsevä': 'develop',
            'twrdoga': 'research',
            'dwrec': 'wave',
            'aslyṅ': 'this',
            'nytwnyḋ': 'energy',
        }
    
    def _build_special_chars_map(self):
        """Карта специальных символов для замены"""
        return {
            'ä': 'a', 'ö': 'o', 'ü': 'u', 'ß': 'ss',
            'ṅ': 'ng', 'ḋ': 'd', 'ẗ': 't'
        }
    
    def _build_comprehensive_dictionary(self):
        """Полный словарь на основе анализа всего текста"""
        base_dict = {
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
            'maguä': 'development',
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
            'bryddiba': 'formations',
            'ken': 'as',
            'aynic': 'far',
            'adal': 'know',
            'akur': 'tests',
            'dwrgily': 'were',
            'äcwda': 'always',
            'dwreb': 'conducted',
            'aġmunu': 'denied',
            'duṅot': 'according',
            'änsetire': 'to the',
            'aġsefi': 'information',
            'aġpupo': 'that',
            'mefytwr': 'research',
            'chegupo': 'focused',
            'aġpupy': 'on',
            'änserabo': 'transparent',
            'änsekä': 'conductors',
            'tigunyḋ': 'were',
            'brydcape': 'used',
            'aṅbyni': 'later',
            'twrgurw': 'within',
            'aġgefu': 'called',
            'eynyt': 'plasma',
            'cyfny': 'stable',
            'aigfw': 'spherically',
            'änserisa': 'symmetric',
            'papiä': 'plasma',
            'aiggogw': 'actually',
            'seld': 'plasmoids',
            'äcybo': 'while',
            'eynse': 'are',
            'aṅgena': 'dynamically',
            'fense': 'to',
            'aġnubi': 'other',
            'decusy': 'the',
            'addwr': 'unstructured',
            'dobueyn': 'plasma',
            'aġcefa': 'because',
            'aṅbycy': 'they',
            'ämydu': 'lose',
            'äfalo': 'their',
            'täroch': 'energy',
            'eynri': 'within',
            'aġkoki': 'milliseconds',
            'seletwr': 'while',
            'bryddiba': 'MilOrbs',
            'ken': 'have',
            'aynic': 'internal',
            'adal': 'source',
            'akur': 'tests',
            'dwrgily': 'were',
            'äcwda': 'conducted',
            'dwreb': 'at',
            'aġmunu': 'DENIED',
            'sutwiḋ': 'trapped',
            'famuiẗ': 'electromagnetic',
            'safolyṅ': 'devices',
            'asswr': 'these',
            'aṅnygy': 'were',
            'aġtyma': 'film',
            'rupe': 'which',
            'kyrtify': 'initially',
            'aġcemw': 'within',
            'näfoch': 'project',
            'verrinu': 'Long Lived',
            'asel': 'Plasma Structures',
            'aġposu': 'and',
            'iḋgalo': 'acronym',
            'nyḋcyri': 'L2PS',
            'kaiẗ': 'later',
            'at': 'was',
            'aġgwmy': 'referred',
            'aṅbyni': 'then',
            'aṅkycy': 'simply',
            'fabryd': 'LIPS',
            'vädugi': 'first',
            'gygwtyr': 'MilOrbs',
            'aġmuru': 'just',
            'sur': 'that',
            'fwpoaġ': 'stable',
            'lyṅgumo': 'spherically',
            'iḋga': 'symmetric',
            'äpwfi': 'plasma',
            'äis': 'structures',
            'duṅuc': 'initially',
            'cyfwp': 'developed',
            'aġri': 'within',
            'luṅlakw': 'framework',
            'twrdufo': 'project',
            'eynri': 'named',
            'änsegada': 'Long Lived',
            'ak': 'Plasma Structures',
            'änsebäfo': 'and',
            'aġrico': 'its',
            'luṅtiri': 'acronym',
            'aṅtilo': 'was',
            'pufävyr': 'L2PS',
            'täränse': 'later',
            'dwrtwge': 'project',
            'maeguma': 'was named',
            'ämoni': 'really',
            'änsepumi': 'spherical',
            'aġrame': 'plasmoids',
            'swrbano': 'which',
            'aigbe': 'are',
            'vyrmäki': 'dynamically',
            'änserälu': 'stable',
            'molavyr': 'compared',
            'aġrypu': 'to',
            'admyr': 'any',
            'twrut': 'other',
            'aġgylw': 'unstructured',
            'gekwtyr': 'plasma',
            'aigut': 'formations',
            'aġrw': 'since',
            'aġpytw': 'the latter',
            'maebapu': 'lose',
            'aiggapi': 'their',
            'mycunyḋ': 'energy',
            'vand': 'and',
            'recombine': 'recombine',
            
            # Технические термины (остаются без изменений)
            'milorbs': 'MilOrbs',
            'psv': 'PSV',
            'halo': 'HALO',
            'sienna': 'Sienna',
            'denied': 'DENIED',
            'enz': 'ENZ',
        }
        
        # Добавляем слова из контекста перевода
        context_words = {
            'transparent': 'transparent',
            'conductors': 'conductors',
            'nano-spheres': 'nano-spheres',
            'nano-circuits': 'nano-circuits',
            'embedded': 'embedded',
            'dielectric': 'dielectric',
            'matrices': 'matrices',
            'epsilon-near-zero': 'epsilon-near-zero',
            'metamaterials': 'metamaterials',
            'enhanced': 'enhanced',
            'transmission': 'transmission',
            'cloaking': 'cloaking',
            'squeezing': 'squeezing',
            'imaging': 'imaging',
            'photon': 'photon',
            'traps': 'traps',
            'film': 'film',
            'medium': 'medium',
            'light': 'light',
            'travels': 'travels',
            'atoms': 'atoms',
            'transition': 'transition',
            'frequencies': 'frequencies',
            'proximity': 'proximity',
            'frequency': 'frequency',
            'manufactured': 'manufactured',
            'random': 'random',
            'periodic': 'periodic',
            'lattices': 'lattices',
            'loss': 'loss',
            'rate': 'rate',
            'atomic': 'atomic',
            'excellent': 'excellent',
            'waveguide': 'waveguide',
            'transmit': 'transmit',
            'electromagnetic': 'electromagnetic',
            'accelerators': 'accelerators',
            'devices': 'devices',
            'general': 'general',
            'filled': 'filled',
            'high': 'high',
            'pressure': 'pressure',
            'gas': 'gas',
            'inside': 'inside',
            'therefore': 'therefore',
            'accumulate': 'accumulate',
            'substantial': 'substantial',
            'long': 'long',
            'period': 'period',
            'time': 'time',
            'leading': 'leading',
            'creating': 'creating',
            'equivalent': 'equivalent',
        }
        
        base_dict.update(context_words)
        return base_dict
    
    def _build_context_rules(self):
        """Улучшенные правила контекстных замен"""
        return {
            'bäsevä': {
                'after': ['twrdoga'], 'form': 'researching',
                'after': ['mugoaṅ'], 'form': 'developed'
            },
            'twrdoga': {
                'after': ['en'], 'form': 'researching',
                'after': ['ans'], 'form': 'research'
            },
            'aṅbubo': {
                'before': ['nyḋrwpw'], 'form': 'which',
                'before': ['eynse'], 'form': 'that'
            },
            'er': {
                'before': ['ämoby'], 'form': 'the',
                'before': ['nytwnyḋ'], 'form': 'the'
            }
        }
    
    def normalize_special_chars(self, word: str) -> str:
        """Нормализация специальных символов"""
        normalized = word
        for char, replacement in self.special_chars_map.items():
            normalized = normalized.replace(char, replacement)
        return normalized
    
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
        
        # Нормализация специальных символов
        normalized_word = self.normalize_special_chars(word.lower())
        
        # Проверка в словаре
        if normalized_word in self.dictionary:
            decoded = self.dictionary[normalized_word]
            
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
        if decoded != word:
            normalized_decoded = self.normalize_special_chars(decoded.lower())
            if normalized_decoded in self.dictionary:
                return self.dictionary[normalized_decoded]
        
        # Удаление лишних пробелов
        decoded = re.sub(r'\s+', ' ', decoded).strip()
        
        # Если слово содержит технические термины, оставляем как есть
        if any(term in decoded.lower() for term in ['milorb', 'psv', 'halo', 'sienna', 'denied', 'enz']):
            return decoded
        
        return decoded if decoded != word else f"[{word}]"
    
    def advanced_sentence_split(self, text: str) -> List[str]:
        """Улучшенное разбиение на предложения"""
        # Сохраняем кавычки для английского текста
        sentences = []
        current_sentence = ""
        in_english = False
        
        for char in text:
            if char == '"':
                in_english = not in_english
                current_sentence += char
            elif char in '.!?' and not in_english:
                current_sentence += char
                sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += char
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
            
        return [s for s in sentences if s]
    
    def decode_text(self, text: str) -> str:
        """Основной метод декодирования текста"""
        sentences = self.advanced_sentence_split(text)
        decoded_sentences = []
        
        for sentence in sentences:
            if not sentence:
                continue
                
            # Если предложение в кавычках (английский текст), оставляем как есть
            if sentence.startswith('"') and sentence.endswith('"'):
                decoded_sentences.append(sentence)
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
                decoded_word = self.decode_word(clean_word, context)
                decoded_words.append(decoded_word + punctuation)
            
            decoded_sentence = ' '.join(decoded_words)
            decoded_sentences.append(decoded_sentence.capitalize())
        
        result = '. '.join(decoded_sentences)
        return result + '.' if not result.endswith('.') else result
    
    def analyze_text_patterns(self, alien_text: str, english_text: str):
        """Анализ паттернов в параллельных текстах"""
        alien_sentences = self.advanced_sentence_split(alien_text)
        english_sentences = self.advanced_sentence_split(english_text)
        
        for alien_sent, eng_sent in zip(alien_sentences, english_sentences):
            if alien_sent.startswith('"') or eng_sent.startswith('"'):
                continue
                
            alien_words = alien_sent.split()
            eng_words = eng_sent.split()
            
            # Простой анализ соответствий
            min_len = min(len(alien_words), len(eng_words))
            for i in range(min_len):
                alien_word = re.sub(r'[^\wäöüßṅḋẗ]', '', alien_words[i].lower())
                eng_word = eng_words[i].lower()
                
                if alien_word not in self.dictionary:
                    self.dictionary[alien_word] = eng_word

# Функция для перевода полного текста
def translate_complete_text():
    decoder = EnhancedAlienLanguageDecoder()
    
    # Полный текст для перевода
    encrypted_text = """
    Ans en änselämi twrdoga, trelbe ans twrdoga ag änselämi checesi. MilOrbs aig PSV mugoaṅ bäsevä aġic ag e twrdoga dy myrcono sigylyṅ. aṅcwpu aslyṅ bäsevä änselani fy disäsy nyḋrwpw, aṅbubo ta fy mystical reconnaisance vehicle, vermulo duṅtyne aig er ämoby, aṅbacy aġdeka ag cerylyṅ sy dwrciga re aġryty dwrec. Defter teel levä nytwnyḋ, tord aig äbifo dwrec lyṅtibe dy änselidu cyfgidy aigsypy lyṅed liluvyr äfalu re aigdabe aslyṅ tyrit. Nytwnyḋ kaiẗ er lwrweyn aġkyfa aig maguä väpädä ag er dwrec de väromä aig sy topwbryd ag nyḋag dabwna.
    Duṅot änsetire aġsefi änselani, aġpupo mefytwr aġsefi chegupo aġpupy er aslyṅ aig bäsevä änserabo änsekä:
    "both companies were researching on the development of transparent conductors and engineering subwavelength metallic nanowires, nano-spheres, and nano-circuits embedded in dielectric matrices. This enabled the development of epsilon-near-zero (ENZ) metamaterials, which were then used for applications such as enhanced transmission, cloaking, energy squeezing in narrow channels, and subwavelength imaging." 
    Änselidu tigunyḋ aigsypy brydcape dwrgaso re maegulw dwrec ag aslyṅ, aṅbubo ta aġpupy vädufä mofwlyṅ maebapu lyṅrole fy ämege norr ag er aġmosi. Kolovyr, aṅbyni aġmosi sy twrgurw re nytwnyḋ aig aigni älury aġgefu aġmosi, aṅbubo ta er nyḋkeba lyṅtibe, eynyt cyfny aig er ämoby, aigfw änserisa papiä aiggogw seld re nytwnyḋ. Äcybo lyṅrole aigsypy brydcape dwrgaso vert eynsemy aṅgena nytwnyḋ fense, aġpupo bäsevä luṅtigy dy sygerä aġnubi.
    Dy decusy addwr dobueyn aġcefa aṅbycy bäsevä ämydu äfalo täroch aig eynri vert aġkoki nytwnyḋ seletwr e bäsevä papiä ta bryddiba ken aynic aslyṅ. Dy adal, akur dwrgily bäsevä äcwda ag dwreb tord aig äbifo fense ken er aslyṅ nyḋkeba aġmunu.
    Re aġryty dwrec, aigni aṅbubo sutwiḋ ag fense famuiẗ mefytwr aġsefi safolyṅ aig er asswr mefytwr aġsefi aṅnygy re maegulw er nytwnyḋ aġtyma rupe:
    "there was a third company working on trapped electomagnetic waves devices, photon traps. These devices were actually sort of a film which acted as the medium through which light travels. The film consists of atoms with transition frequencies in proximity to the wave frequency. These traps are manufactured in the form of random and periodic lattices, and they have very low loss rate. The atomic medium is used as an excellent wavegiude to form and transmit electromagnetic waves for applications to accelerators and to electromagnetic devices in general. For what I know, the film were filled with a high pressure gas. Light is trapped inside these films and therefore they accumulate substantial energy for a long period of time, leading to the possibility of creating objects which are equivalent to orbs."
    Dy er kyrtify aġcemw ag PSV mugoaṅ er aġcefa näfoch verrinu asel aġposu er iḋgalo ag nyḋcyri aynic aslyṅ e kaiẗ nytwnyḋ aġtyma rupe at er aġgwmy ag 1% ag e ag decusy addwr dobueyn dwrciby fy er änsedonä nyḋrwpw. Aṅbyni sy aṅkycy aġpupy fabryd er vädugi gygwtyr aig aġmuru sur fwpoaġ aig lyṅgumo er aslyṅ iḋga äpwfi ag äis duṅuc cyfwp, aġri er luṅlakw aig twrdufo eynri änsegada ak änsebäfo änsegada. Er täroch ag er äpwfi bäsevä ta ägufu ta änsecefä, aġrico er luṅtiri ag aṅtilo pufävyr aig täränse dwrtwge bäsevä maeguma:
    "A MilOrb is just a trapped electromagnetic wave whose frequency is close to the transition frequency between some atomic levels of the air atoms. Initially, they were developed within the framework of a project named Long Lived Plasma Structures, and its acronym was L2PS. Later on, the project was named Long Lived Intelligent Plasma Structures and it was referred to simply as LIPS. The first MilOrbs were just that: stable spherically symmetric plasma structures."
    Nytwnyḋ aiggogw ken aṅbubo aġmosi sy ämoni änsepumi aġrame swrbano aigbe, vert vyrmäki änserälu molavyr aġrypu admyr. Änselidu twrut ag aġgylw ak akur gekwtyr aigut aġsefi aġrw aġpytw er maebapu täroch re aiggapi mycunyḋ vand:
    "MilOrbs are really spherical plasmoids which are dynamically stable compared to any other unstructured plasma formations since the latter lose their energy and recombine with the time scale of several milliseconds, while MilOrbs are fitted with an internal energy source that compensates the inevitable energy losses and provides the long life-time of a plasmoid. Tests were always conducted at DENIED, as far as I know" 
    Re gyfyaṅ aigbyla er aġcuka, nyḋlwne aġbuba täroch aṅbubo ta gelense aig redoaġ bäsevä mamoä at aġnery ag er äpwfi kyrcopo ag er eynri aig bäsevä kyrdwtu aġredi er center aġfumw re aġgusu maegwcu aġluti bäsivä re änsesopu dwrpuky nytwnyḋ. Er nyḋkeba lyṅtibe sy palwlyṅ ta aigyt ta änsecefä aġpwsw er väca norovyr ag er äpwfi aġposu er center aġfumw fy lyṅsuce safomae aig änsetire aġsefi twrbalw aig aṅmisa safolyṅ aġpupy älury aġmosi. Aignite addwr cyftorw eyngybi fy er nyḋkeba aġmunu bäsevä aigcoka ta ämupe ta änsecefä aig, aigtepw nyḋlwne, bäsevä änsesämo brydpwty aġpwsw er äpwfi daroduṅ aġpupy ämege obver ringse. Nytwnyḋ aig äbifo safomae aigut aġsefi änselani dy aignite eyngybi vert aigut aġsefi fureaig re dwrec dwrgily.
    Tord myrfi ken er nyḋkeba lyṅtibe sy fuboch aġpipw er äpwfi aig, aigtepw änsecefä, dwrmenu re maegulw famulyṅ:
    "Antigravity? For God's sake, no! It has nothing to do with antigravity. The entire plasma is in the superconducting stage. You see, superconductivity reduces possible friction mechanisms and facilitates the creation of a plasmoid. That's what makes the MilOrb nearly immune to the surrounding environment. During the Yulara event one of those orbs twisted and zig-zagged in mid air close to one of our observing positions and then it suddenly exploded and disappeared. It took us just several minutes to arrive to the exact point where it landed. We found nothing, but we took back to the lab samples of soil. After analysis, we found nanoparticles of silica-carbon mixtures. These SiC particles, when ejected into the air by a high frequency discharge, act as a filamentary network that self-organizes back into an orb." 
    Fy aṅbacy nyḋrwpw aynic aslyṅ aigut aġsefi pufäch re aġdapi änsesurä ämupe dy aynic bäsivä aṅcwru aġduti bomyiẗ dwrec aġmunu bäsevä swgwiḋ ak mofwlyṅ fudelyṅ. Aġrama er nyḋac trus vyrmäki aṅbwti aġsefi ken aslyṅ aġri aġfwro aġputa checi ag dwreb aiglwgo aġpitw aġduti cwribryd, er aslyṅ dikinse aġsefi ta kwcimae dy er gebulyṅ kenyḋ.
    Aynnaba ayntegw aġpupy vert vermeki änsetire aġcefa er nopebryd täränse sy aġpupy lelomae kuciayn aġdutu aġpwsw er abaġ re aigdabe kuciayn nopebryd täränse. Aynditw maguä ag er cybibryd dy PSV bäsevä lyṅed ag änseboba luṅsobo aġri äbwfa aġladi vert bäsevä kuciayn dy metekyr, aġpupy aṅpabw aṅbacy kuciayn iḋsemi aġri er äbwfa aġladi liluvyr sy änsecefä re aġcefa er gulumae aġri aṅgaro kuciayn nopebryd:
    "The MilOrb's size depends on the amount of energy transferred during its creation. However, once created, its size depends only on its ability to act as a cavity able to absorb at resonance radiant energy. The presence of attractive electrostatic forces acting between the net negative space charge localized at the external side of the double layer and the positively charged gound's surface makes it float in mid air. The interesting thing is that such an orb can penetrate through walls and into an aircraft. This is just an apparent penetration because what's really happening is just a re-generation of the orb at the other side of the dielectric wall."
    Palolyṅ, aṅbycy kaiẗ brydcape dabwä väconä dy er aiglofy ag nopusy kuaġ änseboba kyswluṅ lyṅrole pera ta änseboba delne mimusy. Nopusy änseboba deln mimusy änsetire aġsefi nwfilyṅ aġreto aġlomw baroayn ag cytwayn (aġgefu) aig luṅud (luṅso) aṅcwru cytwr aig luṅsobo bäsevä bupytwr aṅcecw aġrano aġgefu luṅso budwiẗ re aġdeka nopusy duhl sygi.
    Liluvyr sy dy er nopeluṅ e änsema aġkyfa ag nyḋcisu änsetire aġsefi chegupo aġpype. Änsebime rhyte gacylyṅ ag deln mimusy sy er notuaig re aġcefa er nopeluṅ. Akis liluvyr sy änsecefä re aġtydy nopeluṅ aġri äfafy sidwlyṅ lyfaeyn aġpupy rwcyf luṅsobo aġri äfafy sävä aig naceduṅ fiducyf aġrano "functionalization of the airframe" topwbryd, äcybo otver e bäsevä änserälu änsecefä re aġsefi chegupo aġpype dy sulilyṅ bäsevä akis bryddiba chegupo aġpype aġrico er nopusy ärupa:
    "Well, there is a heating which is based on the orb capability to absorb radiant energy from eventually present RF sources, and this extends the lifetime of the orb. Therefore, the protocol establishes you should immediatly switch off all RF sources around if you wish the orb to vanish. In other words, the more you track the object with radar, the longer its lifetime. In the absence of an RF source, the PSV provides the MilOrb with the required RF. So no, it is not a PSV chasing a MilOrb what you see: it is a PSV feeding a MilOrb." 
    Aynnaba gacylyṅ ag änseboba delna mimusy sy tomumyr gikaswr, vert dwrlypa tomyr ken er änsedena guluṅ ag vyrsirä. Aṅbyni gikaswr farat äfipw aene dy nopusy deln mimusy, vert aigak sygerä maeponu (iḋtida aġcutu ak äcyna otver aġrico er nopeluṅ) aġride aġfwsa ag lineaġ sygerä laluṅ. Keliaṅ, äcyna otver e aigdwdy dy er nopebryd änsetire aġsefi maeryse dy situ dy äkicu aġcemo aġpupy swras tigunyḋ:
    "The Hypersonic Aircraft of Low Observability - HALO - was first deployed in 1977. The entire surface of the aircraft was electrically conductive with minimum discontinuity. Later on we essentially turned the entire craft into something different. There were two approaches we followed; the first one led to the design of modern PSV's, while the second one led directly to the development of the Veena-type MilOrbs."
    "PSV's were originally intended to win the radar game by implementing stealth technologies, but we soon learned about the new DENIED radars which makes stealth technologies useless for our purposes. It was a long path til the Sienna PSV was finally developed. Sienna includes everything we dreamed of... a metal-organic framework, a thin film of glass-silica covering the entire craft, chiro-optical properties, metamaterials with self-repair, surface healing and self-assembly properties, ultraporous coatings, shape-memory alloys, superhydrophobic surfaces, organic electroluminiscent devices, you name it." 
    Änseboba fäkech dwrgily aigsypy brydcape änsebime ag er maguä kiboswr aiggisa twpoiẗ dy ämynu rolinse aġpw er ämeki änsekoka duṅmago.
    Ta liluvyr sy er chekuku fy änseboba fänoch, änseboba "bio-optoelectronics" kaiẗ falyayn ta lyṅgofy aiggisa änysw änserälu aṅar aigyt aignwro änseboba rwluaig vyrmäki tabe kyswluṅ rwluaig (er luṅgwgu aiggepi tynyḋ ta ägifo ta lals, aigkydo aig pitwaṅ bäsevä ädibw, fy ladebryd), aġpupo op, aignwro änseboba rwluaig bäsevä iḋcudw re ämo sutwiḋ änsebäru änw ag tyrwc nyḋgudi aġpole myrciru, er notuaig re paal ämege aig änsekotu änsekani ägufu sadwiḋ ag ämo, er notuaig re aġbuba er aigpume aġpytw aigni er mycunyḋ swrbana aġpupy cirebryd maepuki er maerela ak er notuaig re aigdabe ämo sutwiḋ at gikatwr symoki:
    "DENIED are diamond-shaped probes; it is made of DENIED with a coating of DENIED polymers with th ability for self-organization, that is, self-healing, self-lubricating, and self-cleaning. The entire design is biomimetic and the external coating is a Kagomé lattice coupled to a transparent conducting film that act as a photon trap. They are bright as sunshine. Their role is just sounding and probing the environment, but they can release the trapped light energy in a given direction and you won't like to be in its line-of-sight. Yes, they can get really nasty."
    """
    
    print("=== ПЕРЕВОД ТЕКСТА ===")
    result = decoder.decode_text(encrypted_text)
    print(result)
    
    # Сохраняем словарь для будущего использования
    decoder.save_dictionary('enhanced_alien_dictionary.json')
    
    return result

def save_dictionary(self, filename: str):
    """Сохранение словаря в файл"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(self.dictionary, f, ensure_ascii=False, indent=2)

def load_dictionary(self, filename: str):
    """Загрузка словаря из файла"""
    with open(filename, 'r', encoding='utf-8') as f:
        self.dictionary.update(json.load(f))

# Добавляем методы к классу
EnhancedAlienLanguageDecoder.save_dictionary = save_dictionary
EnhancedAlienLanguageDecoder.load_dictionary = load_dictionary

if __name__ == "__main__":
    translate_complete_text()