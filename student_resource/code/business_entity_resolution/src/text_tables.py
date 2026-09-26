# -*- coding: utf-8 -*-
"""
text_tables.py -- static lookup tables for business-entity resolution
(business names + addresses; US, India, France).

Pure Python data. The only import is `unidecode`, used at module load to
register ASCII-transliterated variants of every non-ASCII key (so that a
caller who unidecodes its input before lookup still hits the table).

Conventions
-----------
* All keys are lowercase. ASCII keys carry no punctuation; where a key
  naturally has punctuation ('h.no', "l'", 'saint-louis') the punctuation-
  stripped, space-separated variant is what is stored ('h no', 'l',
  'saint louis').
* Values are canonical forms, consistent within each table.
* Tables are meant for UNIFICATION of variants, not linguistic correctness.
"""

from unidecode import unidecode

# ---------------------------------------------------------------------------
# 1. US states / territories -> 2-letter USPS code (lowercase)
# ---------------------------------------------------------------------------
US_STATES = {
    # full names
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar',
    'california': 'ca', 'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de',
    'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi', 'idaho': 'id',
    'illinois': 'il', 'indiana': 'in', 'iowa': 'ia', 'kansas': 'ks',
    'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms',
    'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv',
    'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny',
    'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok',
    'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc',
    'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut',
    'vermont': 'vt', 'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv',
    'wisconsin': 'wi', 'wyoming': 'wy',
    'district of columbia': 'dc', 'washington dc': 'dc', 'washington d c': 'dc',
    'puerto rico': 'pr', 'guam': 'gu', 'virgin islands': 'vi',
    'us virgin islands': 'vi', 'u s virgin islands': 'vi',
    'american samoa': 'as', 'northern mariana islands': 'mp',
    # common variants / older abbreviations
    'n carolina': 'nc', 'n dakota': 'nd', 's carolina': 'sc', 's dakota': 'sd',
    'w virginia': 'wv', 'n hampshire': 'nh', 'n jersey': 'nj', 'n mexico': 'nm',
    'n york': 'ny', 'new york state': 'ny', 'commonwealth of massachusetts': 'ma',
    'commonwealth of pennsylvania': 'pa', 'commonwealth of virginia': 'va',
    'commonwealth of kentucky': 'ky', 'state of texas': 'tx', 'state of california': 'ca',
    'calif': 'ca', 'cali': 'ca', 'fla': 'fl', 'tex': 'tx', 'penn': 'pa', 'penna': 'pa',
    'mass': 'ma', 'conn': 'ct', 'ariz': 'az', 'ark': 'ar', 'colo': 'co', 'del': 'de',
    'ill': 'il', 'ind': 'in', 'kans': 'ks', 'kan': 'ks', 'mich': 'mi', 'minn': 'mn',
    'miss': 'ms', 'mont': 'mt', 'nebr': 'ne', 'neb': 'ne', 'nev': 'nv', 'okla': 'ok',
    'oreg': 'or', 'ore': 'or', 'tenn': 'tn', 'wash': 'wa', 'wis': 'wi', 'wisc': 'wi',
    'wyo': 'wy', 'ala': 'al', 'n mex': 'nm', 'n car': 'nc', 's car': 'sc',
    'n dak': 'nd', 's dak': 'sd', 'w va': 'wv', 'n y': 'ny', 'n j': 'nj', 'n c': 'nc',
    'n d': 'nd', 's c': 'sc', 's d': 'sd', 'n h': 'nh', 'n m': 'nm', 'd c': 'dc',
    'r i': 'ri', 'w v': 'wv',
}
"""Full US state / territory names (and common variants) -> lowercase USPS code.
Codes are also mapped to themselves (added in the loop right below)."""
for _c in list(set(US_STATES.values())):
    US_STATES[_c] = _c

# ---------------------------------------------------------------------------
# 2. Indian states / union territories -> canonical 2-letter code
# ---------------------------------------------------------------------------
IN_STATES = {
    # --- English names + codes ---
    'maharashtra': 'mh', 'mh': 'mh', 'maharastra': 'mh', 'maharshtra': 'mh',
    'telangana': 'tg', 'tg': 'tg', 'ts': 'tg', 'telengana': 'tg', 'telangana state': 'tg',
    'karnataka': 'ka', 'ka': 'ka', 'karnatak': 'ka', 'karnatka': 'ka', 'mysore state': 'ka',
    'tamil nadu': 'tn', 'tn': 'tn', 'tamilnadu': 'tn', 'tamilnad': 'tn', 'tamil nad': 'tn',
    'kerala': 'kl', 'kl': 'kl', 'keralam': 'kl', 'kerela': 'kl',
    'delhi': 'dl', 'dl': 'dl', 'new delhi': 'dl', 'nct of delhi': 'dl', 'nct delhi': 'dl',
    'national capital territory of delhi': 'dl', 'delhi ncr': 'dl', 'old delhi': 'dl',
    'uttar pradesh': 'up', 'up': 'up', 'uttarpradesh': 'up', 'utter pradesh': 'up',
    'west bengal': 'wb', 'wb': 'wb', 'bengal': 'wb', 'westbengal': 'wb', 'paschim banga': 'wb',
    'paschimbanga': 'wb', 'w bengal': 'wb', 'west bangal': 'wb',
    'gujarat': 'gj', 'gj': 'gj', 'gujrat': 'gj', 'gujarath': 'gj',
    'rajasthan': 'rj', 'rj': 'rj', 'rajastan': 'rj', 'rajsthan': 'rj',
    'haryana': 'hr', 'hr': 'hr', 'hariyana': 'hr',
    'punjab': 'pb', 'pb': 'pb', 'panjab': 'pb',
    'odisha': 'od', 'od': 'od', 'or': 'od', 'orissa': 'od', 'odissa': 'od', 'udisa': 'od',
    'andhra pradesh': 'ap', 'ap': 'ap', 'andhra': 'ap', 'andra pradesh': 'ap', 'andhrapradesh': 'ap',
    'andhra pradesh state': 'ap', 'a p': 'ap',
    'madhya pradesh': 'mp', 'mp': 'mp', 'madhyapradesh': 'mp', 'm p': 'mp', 'madhya pradesh state': 'mp',
    'jharkhand': 'jh', 'jh': 'jh', 'jharkand': 'jh', 'jharkhnd': 'jh',
    'chhattisgarh': 'cg', 'cg': 'cg', 'ct': 'cg', 'chattisgarh': 'cg', 'chhatisgarh': 'cg',
    'chattisgarh state': 'cg', 'chhattisgarh state': 'cg',
    'uttarakhand': 'uk', 'uk': 'uk', 'ua': 'uk', 'uttaranchal': 'uk', 'uttrakhand': 'uk',
    'uttarkhand': 'uk',
    'himachal pradesh': 'hp', 'hp': 'hp', 'himachal': 'hp', 'himachalpradesh': 'hp', 'h p': 'hp',
    'jammu and kashmir': 'jk', 'jk': 'jk', 'jammu kashmir': 'jk', 'jammu': 'jk', 'kashmir': 'jk',
    'j and k': 'jk', 'j k': 'jk', 'jammu amp kashmir': 'jk',
    'goa': 'ga', 'ga': 'ga',
    'assam': 'as', 'as': 'as', 'asom': 'as', 'axom': 'as',
    'bihar': 'br', 'br': 'br',
    'sikkim': 'sk', 'sk': 'sk',
    'tripura': 'tr', 'tr': 'tr',
    'meghalaya': 'ml', 'ml': 'ml',
    'manipur': 'mn', 'mn': 'mn',
    'mizoram': 'mz', 'mz': 'mz',
    'nagaland': 'nl', 'nl': 'nl',
    'arunachal pradesh': 'ar', 'ar': 'ar', 'arunachal': 'ar',
    'chandigarh': 'ch', 'ch': 'ch', 'chandigarh ut': 'ch',
    'puducherry': 'py', 'py': 'py', 'pondicherry': 'py', 'pondichery': 'py', 'pondy': 'py',
    'ladakh': 'la', 'la': 'la',
    'andaman and nicobar islands': 'an', 'an': 'an', 'andaman and nicobar': 'an',
    'andaman nicobar': 'an', 'andaman': 'an', 'a and n islands': 'an', 'a n islands': 'an',
    'dadra and nagar haveli and daman and diu': 'dn', 'dn': 'dn', 'dd': 'dn',
    'dadra and nagar haveli': 'dn', 'daman and diu': 'dn', 'daman': 'dn', 'diu': 'dn',
    'dadra nagar haveli': 'dn', 'silvassa': 'dn',
    'lakshadweep': 'ld', 'ld': 'ld', 'lakshadeep': 'ld',
    # --- Devanagari (Hindi / Marathi) ---
    'महाराष्ट्र': 'mh', 'तेलंगाना': 'tg', 'तेलंगणा': 'tg', 'कर्नाटक': 'ka', 'तमिलनाडु': 'tn',
    'तमिळनाडू': 'tn', 'केरल': 'kl', 'केरळ': 'kl', 'दिल्ली': 'dl', 'नई दिल्ली': 'dl',
    'उत्तर प्रदेश': 'up', 'पश्चिम बंगाल': 'wb', 'गुजरात': 'gj', 'राजस्थान': 'rj',
    'हरियाणा': 'hr', 'हरयाणा': 'hr', 'पंजाब': 'pb', 'ओडिशा': 'od', 'उड़ीसा': 'od', 'ओड़िशा': 'od',
    'आंध्र प्रदेश': 'ap', 'आन्ध्र प्रदेश': 'ap', 'मध्य प्रदेश': 'mp', 'झारखंड': 'jh', 'झारखण्ड': 'jh',
    'छत्तीसगढ़': 'cg', 'छत्तीसगढ': 'cg', 'उत्तराखंड': 'uk', 'उत्तराखण्ड': 'uk',
    'हिमाचल प्रदेश': 'hp', 'जम्मू और कश्मीर': 'jk', 'जम्मू कश्मीर': 'jk', 'गोवा': 'ga', 'गोआ': 'ga',
    'असम': 'as', 'बिहार': 'br', 'सिक्किम': 'sk', 'त्रिपुरा': 'tr', 'मेघालय': 'ml', 'मणिपुर': 'mn',
    'मिज़ोरम': 'mz', 'मिजोरम': 'mz', 'नागालैंड': 'nl', 'अरुणाचल प्रदेश': 'ar', 'चंडीगढ़': 'ch',
    'चण्डीगढ़': 'ch', 'पुडुचेरी': 'py', 'पांडिचेरी': 'py', 'लद्दाख': 'la',
    'अंडमान और निकोबार द्वीपसमूह': 'an', 'लक्षद्वीप': 'ld',
    'दादरा और नगर हवेली': 'dn', 'दमन और दीव': 'dn',
    # --- Bengali ---
    'পশ্চিমবঙ্গ': 'wb', 'পশ্চিম বঙ্গ': 'wb', 'মহারাষ্ট্র': 'mh', 'কর্ণাটক': 'ka', 'তামিলনাড়ু': 'tn',
    'কেরালা': 'kl', 'দিল্লি': 'dl', 'উত্তরপ্রদেশ': 'up', 'উত্তর প্রদেশ': 'up', 'গুজরাত': 'gj',
    'রাজস্থান': 'rj', 'হরিয়ানা': 'hr', 'পাঞ্জাব': 'pb', 'ওড়িশা': 'od', 'উড়িষ্যা': 'od',
    'অন্ধ্রপ্রদেশ': 'ap', 'অন্ধ্র প্রদেশ': 'ap', 'মধ্যপ্রদেশ': 'mp', 'মধ্য প্রদেশ': 'mp',
    'ঝাড়খণ্ড': 'jh', 'ছত্তিশগড়': 'cg', 'উত্তরাখণ্ড': 'uk', 'হিমাচল প্রদেশ': 'hp', 'বিহার': 'br',
    'আসাম': 'as', 'অসম': 'as', 'ত্রিপুরা': 'tr', 'সিকিম': 'sk', 'মেঘালয়': 'ml', 'মণিপুর': 'mn',
    'মিজোরাম': 'mz', 'নাগাল্যান্ড': 'nl', 'তেলেঙ্গানা': 'tg', 'গোয়া': 'ga', 'চণ্ডীগড়': 'ch',
    # --- Gujarati ---
    'ગુજરાત': 'gj', 'મહારાષ્ટ્ર': 'mh', 'રાજસ્થાન': 'rj', 'મધ્ય પ્રદેશ': 'mp', 'દિલ્હી': 'dl',
    'કર્ણાટક': 'ka', 'તમિલનાડુ': 'tn', 'કેરળ': 'kl', 'ઉત્તર પ્રદેશ': 'up', 'પંજાબ': 'pb',
    'હરિયાણા': 'hr', 'બિહાર': 'br', 'ગોવા': 'ga', 'તેલંગાણા': 'tg', 'આંધ્ર પ્રદેશ': 'ap',
    'પશ્ચિમ બંગાળ': 'wb', 'ઓડિશા': 'od', 'દમણ': 'dn', 'દીવ': 'dn', 'દાદરા અને નગર હવેલી': 'dn',
    # --- Kannada ---
    'ಕರ್ನಾಟಕ': 'ka', 'ಮಹಾರಾಷ್ಟ್ರ': 'mh', 'ತಮಿಳುನಾಡು': 'tn', 'ಕೇರಳ': 'kl', 'ಆಂಧ್ರ ಪ್ರದೇಶ': 'ap',
    'ತೆಲಂಗಾಣ': 'tg', 'ಗೋವಾ': 'ga', 'ದೆಹಲಿ': 'dl', 'ಗುಜರಾತ್': 'gj', 'ರಾಜಸ್ಥಾನ': 'rj',
    'ಉತ್ತರ ಪ್ರದೇಶ': 'up', 'ಮಧ್ಯ ಪ್ರದೇಶ': 'mp', 'ಪಶ್ಚಿಮ ಬಂಗಾಳ': 'wb', 'ಪಂಜಾಬ್': 'pb',
    'ಬಿಹಾರ': 'br', 'ಒಡಿಶಾ': 'od',
    # --- Telugu ---
    'తెలంగాణ': 'tg', 'ఆంధ్ర ప్రదేశ్': 'ap', 'ఆంధ్రప్రదేశ్': 'ap', 'కర్ణాటక': 'ka', 'తమిళనాడు': 'tn',
    'మహారాష్ట్ర': 'mh', 'కేరళ': 'kl', 'ఢిల్లీ': 'dl', 'ఒడిశా': 'od', 'గుజరాత్': 'gj',
    'రాజస్థాన్': 'rj', 'మధ్య ప్రదేశ్': 'mp', 'ఉత్తర ప్రదేశ్': 'up', 'పశ్చిమ బెంగాల్': 'wb',
    'బీహార్': 'br', 'పంజాబ్': 'pb', 'గోవా': 'ga',
    # --- Tamil ---
    'தமிழ்நாடு': 'tn', 'தமிழ் நாடு': 'tn', 'கேரளா': 'kl', 'கர்நாடகா': 'ka', 'கர்நாடகம்': 'ka',
    'ஆந்திரப் பிரதேசம்': 'ap', 'ஆந்திரா': 'ap', 'தெலுங்கானா': 'tg', 'தெலங்காணா': 'tg',
    'மகாராஷ்டிரா': 'mh', 'டெல்லி': 'dl', 'தில்லி': 'dl', 'புதுச்சேரி': 'py', 'பாண்டிச்சேரி': 'py',
    'குஜராத்': 'gj', 'ராஜஸ்தான்': 'rj', 'மேற்கு வங்காளம்': 'wb', 'உத்தரப் பிரதேசம்': 'up',
    'மத்தியப் பிரதேசம்': 'mp', 'பஞ்சாப்': 'pb', 'பீகார்': 'br', 'ஒடிசா': 'od', 'கோவா': 'ga',
    # --- Malayalam ---
    'കേരളം': 'kl', 'കേരള': 'kl', 'തമിഴ്നാട്': 'tn', 'കർണാടക': 'ka', 'കര്ണാടക': 'ka',
    'മഹാരാഷ്ട്ര': 'mh', 'ആന്ധ്രാപ്രദേശ്': 'ap', 'ആന്ധ്രപ്രദേശ്': 'ap', 'തെലങ്കാന': 'tg',
    'ഡൽഹി': 'dl', 'ദില്ലി': 'dl', 'ഗുജറാത്ത്': 'gj', 'രാജസ്ഥാൻ': 'rj', 'ഉത്തർപ്രദേശ്': 'up',
    'മധ്യപ്രദേശ്': 'mp', 'പശ്ചിമ ബംഗാൾ': 'wb', 'പഞ്ചാബ്': 'pb', 'ബീഹാർ': 'br', 'ഒഡീഷ': 'od',
    'ഗോവ': 'ga', 'ലക്ഷദ്വീപ്': 'ld', 'പുതുച്ചേരി': 'py',
    # --- Odia ---
    'ଓଡ଼ିଶା': 'od', 'ଓଡିଶା': 'od', 'ପଶ୍ଚିମବଙ୍ଗ': 'wb', 'ମହାରାଷ୍ଟ୍ର': 'mh', 'ଆନ୍ଧ୍ର ପ୍ରଦେଶ': 'ap',
    'ତେଲେଙ୍ଗାନା': 'tg', 'ଛତିଶଗଡ': 'cg', 'ଛତିଶଗଡ଼': 'cg', 'ଝାଡ଼ଖଣ୍ଡ': 'jh', 'ବିହାର': 'br',
    'ଦିଲ୍ଲୀ': 'dl', 'କର୍ଣ୍ଣାଟକ': 'ka', 'ତାମିଲନାଡୁ': 'tn', 'କେରଳ': 'kl', 'ଗୁଜରାଟ': 'gj',
    'ମଧ୍ୟପ୍ରଦେଶ': 'mp', 'ଉତ୍ତର ପ୍ରଦେଶ': 'up', 'ରାଜସ୍ଥାନ': 'rj', 'ପଞ୍ଜାବ': 'pb',
    # --- Gurmukhi (Punjabi) ---
    'ਪੰਜਾਬ': 'pb', 'ਹਰਿਆਣਾ': 'hr', 'ਦਿੱਲੀ': 'dl', 'ਨਵੀਂ ਦਿੱਲੀ': 'dl', 'ਚੰਡੀਗੜ੍ਹ': 'ch',
    'ਹਿਮਾਚਲ ਪ੍ਰਦੇਸ਼': 'hp', 'ਰਾਜਸਥਾਨ': 'rj', 'ਉੱਤਰ ਪ੍ਰਦੇਸ਼': 'up', 'ਮਹਾਰਾਸ਼ਟਰ': 'mh',
    'ਗੁਜਰਾਤ': 'gj', 'ਜੰਮੂ ਅਤੇ ਕਸ਼ਮੀਰ': 'jk', 'ਉੱਤਰਾਖੰਡ': 'uk', 'ਮੱਧ ਪ੍ਰਦੇਸ਼': 'mp', 'ਬਿਹਾਰ': 'br',
    'ਕਰਨਾਟਕ': 'ka', 'ਤਮਿਲਨਾਡੂ': 'tn', 'ਕੇਰਲ': 'kl', 'ਪੱਛਮੀ ਬੰਗਾਲ': 'wb',
}
"""Indian states / UTs -> canonical 2-letter code (vehicle-registration / ISO
3166-2:IN style). Canonical picks: 'tg' for Telangana (ts also mapped),
'od' for Odisha (or/orissa mapped), 'cg' Chhattisgarh (ct mapped), 'uk'
Uttarakhand (ua/uttaranchal mapped), 'dn' for DNH&DD (dd mapped), 'py'
Puducherry. Includes native-script spellings (Devanagari, Bengali, Gujarati,
Kannada, Telugu, Tamil, Malayalam, Odia, Gurmukhi); unidecoded variants of
those are added at module load."""

# ---------------------------------------------------------------------------
# 3. French regions / departments -> canonical region slug
# ---------------------------------------------------------------------------
_FR_DEPTS = [
    # (code, department name (unidecoded, hyphens->spaces), region slug)
    ('01', 'ain', 'auvergne rhone alpes'),
    ('02', 'aisne', 'hauts de france'),
    ('03', 'allier', 'auvergne rhone alpes'),
    ('04', 'alpes de haute provence', 'provence alpes cote d azur'),
    ('05', 'hautes alpes', 'provence alpes cote d azur'),
    ('06', 'alpes maritimes', 'provence alpes cote d azur'),
    ('07', 'ardeche', 'auvergne rhone alpes'),
    ('08', 'ardennes', 'grand est'),
    ('09', 'ariege', 'occitanie'),
    ('10', 'aube', 'grand est'),
    ('11', 'aude', 'occitanie'),
    ('12', 'aveyron', 'occitanie'),
    ('13', 'bouches du rhone', 'provence alpes cote d azur'),
    ('14', 'calvados', 'normandie'),
    ('15', 'cantal', 'auvergne rhone alpes'),
    ('16', 'charente', 'nouvelle aquitaine'),
    ('17', 'charente maritime', 'nouvelle aquitaine'),
    ('18', 'cher', 'centre val de loire'),
    ('19', 'correze', 'nouvelle aquitaine'),
    ('2a', 'corse du sud', 'corse'),
    ('2b', 'haute corse', 'corse'),
    ('21', 'cote d or', 'bourgogne franche comte'),
    ('22', 'cotes d armor', 'bretagne'),
    ('23', 'creuse', 'nouvelle aquitaine'),
    ('24', 'dordogne', 'nouvelle aquitaine'),
    ('25', 'doubs', 'bourgogne franche comte'),
    ('26', 'drome', 'auvergne rhone alpes'),
    ('27', 'eure', 'normandie'),
    ('28', 'eure et loir', 'centre val de loire'),
    ('29', 'finistere', 'bretagne'),
    ('30', 'gard', 'occitanie'),
    ('31', 'haute garonne', 'occitanie'),
    ('32', 'gers', 'occitanie'),
    ('33', 'gironde', 'nouvelle aquitaine'),
    ('34', 'herault', 'occitanie'),
    ('35', 'ille et vilaine', 'bretagne'),
    ('36', 'indre', 'centre val de loire'),
    ('37', 'indre et loire', 'centre val de loire'),
    ('38', 'isere', 'auvergne rhone alpes'),
    ('39', 'jura', 'bourgogne franche comte'),
    ('40', 'landes', 'nouvelle aquitaine'),
    ('41', 'loir et cher', 'centre val de loire'),
    ('42', 'loire', 'auvergne rhone alpes'),
    ('43', 'haute loire', 'auvergne rhone alpes'),
    ('44', 'loire atlantique', 'pays de la loire'),
    ('45', 'loiret', 'centre val de loire'),
    ('46', 'lot', 'occitanie'),
    ('47', 'lot et garonne', 'nouvelle aquitaine'),
    ('48', 'lozere', 'occitanie'),
    ('49', 'maine et loire', 'pays de la loire'),
    ('50', 'manche', 'normandie'),
    ('51', 'marne', 'grand est'),
    ('52', 'haute marne', 'grand est'),
    ('53', 'mayenne', 'pays de la loire'),
    ('54', 'meurthe et moselle', 'grand est'),
    ('55', 'meuse', 'grand est'),
    ('56', 'morbihan', 'bretagne'),
    ('57', 'moselle', 'grand est'),
    ('58', 'nievre', 'bourgogne franche comte'),
    ('59', 'nord', 'hauts de france'),
    ('60', 'oise', 'hauts de france'),
    ('61', 'orne', 'normandie'),
    ('62', 'pas de calais', 'hauts de france'),
    ('63', 'puy de dome', 'auvergne rhone alpes'),
    ('64', 'pyrenees atlantiques', 'nouvelle aquitaine'),
    ('65', 'hautes pyrenees', 'occitanie'),
    ('66', 'pyrenees orientales', 'occitanie'),
    ('67', 'bas rhin', 'grand est'),
    ('68', 'haut rhin', 'grand est'),
    ('69', 'rhone', 'auvergne rhone alpes'),
    ('70', 'haute saone', 'bourgogne franche comte'),
    ('71', 'saone et loire', 'bourgogne franche comte'),
    ('72', 'sarthe', 'pays de la loire'),
    ('73', 'savoie', 'auvergne rhone alpes'),
    ('74', 'haute savoie', 'auvergne rhone alpes'),
    ('75', 'paris', 'ile de france'),
    ('76', 'seine maritime', 'normandie'),
    ('77', 'seine et marne', 'ile de france'),
    ('78', 'yvelines', 'ile de france'),
    ('79', 'deux sevres', 'nouvelle aquitaine'),
    ('80', 'somme', 'hauts de france'),
    ('81', 'tarn', 'occitanie'),
    ('82', 'tarn et garonne', 'occitanie'),
    ('83', 'var', 'provence alpes cote d azur'),
    ('84', 'vaucluse', 'provence alpes cote d azur'),
    ('85', 'vendee', 'pays de la loire'),
    ('86', 'vienne', 'nouvelle aquitaine'),
    ('87', 'haute vienne', 'nouvelle aquitaine'),
    ('88', 'vosges', 'grand est'),
    ('89', 'yonne', 'bourgogne franche comte'),
    ('90', 'territoire de belfort', 'bourgogne franche comte'),
    ('91', 'essonne', 'ile de france'),
    ('92', 'hauts de seine', 'ile de france'),
    ('93', 'seine saint denis', 'ile de france'),
    ('94', 'val de marne', 'ile de france'),
    ('95', 'val d oise', 'ile de france'),
    ('971', 'guadeloupe', 'guadeloupe'),
    ('972', 'martinique', 'martinique'),
    ('973', 'guyane', 'guyane'),
    ('974', 'la reunion', 'la reunion'),
    ('976', 'mayotte', 'mayotte'),
]

FR_REGIONS = {
    # current regions (slug -> itself) and variants
    'auvergne rhone alpes': 'auvergne rhone alpes', 'aura': 'auvergne rhone alpes',
    'bourgogne franche comte': 'bourgogne franche comte', 'bfc': 'bourgogne franche comte',
    'bretagne': 'bretagne', 'brittany': 'bretagne', 'breizh': 'bretagne',
    'centre val de loire': 'centre val de loire', 'centre': 'centre val de loire',
    'corse': 'corse', 'corsica': 'corse',
    'grand est': 'grand est', 'alsace champagne ardenne lorraine': 'grand est',
    'hauts de france': 'hauts de france', 'nord pas de calais picardie': 'hauts de france',
    'ile de france': 'ile de france', 'idf': 'ile de france', 'region parisienne': 'ile de france',
    'normandie': 'normandie', 'normandy': 'normandie',
    'nouvelle aquitaine': 'nouvelle aquitaine', 'aquitaine limousin poitou charentes': 'nouvelle aquitaine',
    'occitanie': 'occitanie', 'languedoc roussillon midi pyrenees': 'occitanie',
    'pays de la loire': 'pays de la loire', 'pays de loire': 'pays de la loire',
    'provence alpes cote d azur': 'provence alpes cote d azur', 'paca': 'provence alpes cote d azur',
    'provence alpes cote dazur': 'provence alpes cote d azur', 'cote d azur': 'provence alpes cote d azur',
    'provence': 'provence alpes cote d azur', 'region sud': 'provence alpes cote d azur',
    'guadeloupe': 'guadeloupe', 'martinique': 'martinique', 'guyane': 'guyane',
    'guyane francaise': 'guyane', 'la reunion': 'la reunion', 'reunion': 'la reunion',
    'ile de la reunion': 'la reunion', 'mayotte': 'mayotte',
    # pre-2016 regions -> new region slug
    'aquitaine': 'nouvelle aquitaine', 'limousin': 'nouvelle aquitaine',
    'poitou charentes': 'nouvelle aquitaine',
    'nord pas de calais': 'hauts de france', 'picardie': 'hauts de france',
    'midi pyrenees': 'occitanie', 'languedoc roussillon': 'occitanie', 'languedoc': 'occitanie',
    'roussillon': 'occitanie',
    'rhone alpes': 'auvergne rhone alpes', 'auvergne': 'auvergne rhone alpes',
    'bourgogne': 'bourgogne franche comte', 'franche comte': 'bourgogne franche comte',
    'burgundy': 'bourgogne franche comte',
    'alsace': 'grand est', 'lorraine': 'grand est', 'champagne ardenne': 'grand est',
    'champagne ardennes': 'grand est', 'champagne': 'grand est',
    'basse normandie': 'normandie', 'haute normandie': 'normandie',
    # department-name variants (apostrophe / spelling)
    'cote dor': 'bourgogne franche comte', 'cotes darmor': 'bretagne', 'cotes du nord': 'bretagne',
    'val doise': 'ile de france', 'seine st denis': 'ile de france', 'seine saint denis': 'ile de france',
    'alpes de hte provence': 'provence alpes cote d azur', 'basses alpes': 'provence alpes cote d azur',
    'hte garonne': 'occitanie', 'hte savoie': 'auvergne rhone alpes', 'hte loire': 'auvergne rhone alpes',
    'hte vienne': 'nouvelle aquitaine', 'hte marne': 'grand est', 'hte saone': 'bourgogne franche comte',
    'hte corse': 'corse', 'htes alpes': 'provence alpes cote d azur', 'htes pyrenees': 'occitanie',
    'basses pyrenees': 'nouvelle aquitaine', 'loire inferieure': 'pays de la loire',
    'seine inferieure': 'normandie', 'charente inferieure': 'nouvelle aquitaine',
    'seine et oise': 'ile de france', 'seine': 'ile de france',
    'rhone metropole': 'auvergne rhone alpes', 'metropole de lyon': 'auvergne rhone alpes',
    'grand lyon': 'auvergne rhone alpes', 'lyon': 'auvergne rhone alpes',
    'marseille': 'provence alpes cote d azur', 'toulouse': 'occitanie', 'lille': 'hauts de france',
    'bordeaux': 'nouvelle aquitaine', 'nantes': 'pays de la loire', 'strasbourg': 'grand est',
    'nice': 'provence alpes cote d azur', 'montpellier': 'occitanie', 'rennes': 'bretagne',
}
"""French region names (13 metropolitan + 5 overseas), all 101 department names,
pre-2016 region names and a few major-city fallbacks -> canonical region slug
(lowercase, unidecoded, hyphens/apostrophes -> spaces). Department names are
added from `_FR_DEPTS` below. NOTE: 'lot', 'loire', 'nord', 'centre', 'var',
'seine', 'cher', 'eure', 'manche' are ordinary French words too -- only look up
tokens that occur in a region/department slot."""
for _code, _name, _region in _FR_DEPTS:
    FR_REGIONS.setdefault(_name, _region)

FR_DEPT_CODES = {_code: _region for _code, _name, _region in _FR_DEPTS}
"""Department code (2-digit string, '2a'/'2b' for Corsica, 3-digit for
overseas) -> region slug. Intended as a postal-code prefix lookup:
`FR_DEPT_CODES.get(cp[:3]) or FR_DEPT_CODES.get(cp[:2])`. '20' (Corsican
postal prefix) maps to 'corse'; single-digit strings '1'..'9' are included
for callers that stripped the leading zero. Overseas collectivities
('975','977','978','986','987','988') map to their own slug."""
FR_DEPT_CODES['20'] = 'corse'
for _d in range(1, 10):
    FR_DEPT_CODES[str(_d)] = FR_DEPT_CODES['0%d' % _d]
FR_DEPT_CODES.update({
    '975': 'saint pierre et miquelon', '977': 'saint barthelemy', '978': 'saint martin',
    '986': 'wallis et futuna', '987': 'polynesie francaise', '988': 'nouvelle caledonie',
    '98000': 'monaco', '980': 'monaco',
})

# ---------------------------------------------------------------------------
# 4. Address token abbreviations -> canonical SHORT token
# ---------------------------------------------------------------------------
_ADDR_US = {
    # --- USPS Pub. 28 street suffixes (long form + variants -> canonical) ---
    'alley': 'aly', 'ally': 'aly', 'aly': 'aly',
    'annex': 'anx', 'anex': 'anx', 'annx': 'anx', 'anx': 'anx',
    'arcade': 'arc', 'arc': 'arc',
    'avenue': 'ave', 'ave': 'ave', 'av': 'ave', 'aven': 'ave', 'avenu': 'ave', 'avn': 'ave',
    'avnue': 'ave', 'aveune': 'ave', 'avanue': 'ave', 'avenues': 'ave',
    'bayou': 'byu', 'byu': 'byu', 'beach': 'bch', 'bch': 'bch', 'bend': 'bnd', 'bnd': 'bnd',
    'bluff': 'blf', 'blf': 'blf', 'bluffs': 'blfs', 'blfs': 'blfs', 'bottom': 'btm', 'btm': 'btm',
    'boulevard': 'blvd', 'blvd': 'blvd', 'boul': 'blvd', 'boulv': 'blvd', 'blv': 'blvd',
    'boulevrd': 'blvd', 'bd': 'blvd', 'bld': 'blvd', 'bvd': 'blvd', 'boulvard': 'blvd',
    'branch': 'br', 'brnch': 'br', 'br': 'br',
    'bridge': 'brg', 'brdge': 'brg', 'brg': 'brg', 'brook': 'brk', 'brk': 'brk',
    'brooks': 'brks', 'brks': 'brks', 'burg': 'bg', 'bg': 'bg', 'burgs': 'bgs', 'bgs': 'bgs',
    'bypass': 'byp', 'byp': 'byp', 'bypa': 'byp', 'bypas': 'byp', 'byps': 'byp',
    'camp': 'cp', 'cp': 'cp', 'cmp': 'cp', 'canyon': 'cyn', 'canyn': 'cyn', 'cnyn': 'cyn', 'cyn': 'cyn',
    'cape': 'cpe', 'cpe': 'cpe', 'causeway': 'cswy', 'causwa': 'cswy', 'cswy': 'cswy',
    'center': 'ctr', 'centre': 'ctr', 'cen': 'ctr', 'cent': 'ctr', 'centr': 'ctr', 'cnter': 'ctr',
    'cntr': 'ctr', 'ctr': 'ctr', 'centers': 'ctrs', 'ctrs': 'ctrs',
    'circle': 'cir', 'cir': 'cir', 'circ': 'cir', 'circl': 'cir', 'crcl': 'cir', 'crcle': 'cir',
    'circles': 'cirs', 'cirs': 'cirs',
    'cliff': 'clf', 'clf': 'clf', 'cliffs': 'clfs', 'clfs': 'clfs', 'club': 'clb', 'clb': 'clb',
    'common': 'cmn', 'cmn': 'cmn', 'commons': 'cmns', 'cmns': 'cmns',
    'corner': 'cor', 'cor': 'cor', 'corners': 'cors', 'cors': 'cors',
    'course': 'crse', 'crse': 'crse', 'court': 'ct', 'ct': 'ct', 'crt': 'ct', 'courts': 'cts',
    'cove': 'cv', 'cv': 'cv', 'coves': 'cvs', 'creek': 'crk', 'crk': 'crk',
    'crescent': 'cres', 'cres': 'cres', 'crsent': 'cres', 'crsnt': 'cres', 'crest': 'crst', 'crst': 'crst',
    'crossing': 'xing', 'xing': 'xing', 'crssng': 'xing',
    'crossroad': 'xrd', 'crossroads': 'xrd', 'cross road': 'xrd', 'cross roads': 'xrd',
    'x road': 'xrd', 'x roads': 'xrd', 'xroad': 'xrd', 'xroads': 'xrd', 'xrd': 'xrd', 'xrds': 'xrd',
    'curve': 'curv', 'curv': 'curv',
    'dale': 'dl', 'dl': 'dl', 'dam': 'dm', 'dm': 'dm', 'divide': 'dv', 'dv': 'dv', 'div': 'div',
    'division': 'div', 'divn': 'div',
    'drive': 'dr', 'dr': 'dr', 'driv': 'dr', 'drv': 'dr', 'drve': 'dr', 'drives': 'drs', 'drs': 'drs',
    'estate': 'est', 'est': 'est', 'estates': 'ests', 'ests': 'ests',
    'expressway': 'expy', 'expy': 'expy', 'exp': 'expy', 'expr': 'expy', 'express': 'expy',
    'expw': 'expy', 'expwy': 'expy',
    'extension': 'extn', 'extn': 'extn', 'ext': 'extn', 'extnsn': 'extn', 'extensions': 'extn',
    'falls': 'fls', 'fls': 'fls', 'ferry': 'fry', 'fry': 'fry', 'frry': 'fry',
    'field': 'fld', 'fld': 'fld', 'fields': 'flds', 'flds': 'flds',
    'ford': 'frd', 'frd': 'frd', 'forest': 'frst', 'frst': 'frst', 'forests': 'frst',
    'forge': 'frg', 'frg': 'frg', 'fork': 'frk', 'frk': 'frk', 'forks': 'frks', 'frks': 'frks',
    'fort': 'ft', 'ft': 'ft', 'frt': 'ft',
    'freeway': 'fwy', 'fwy': 'fwy', 'freewy': 'fwy', 'frway': 'fwy', 'frwy': 'fwy',
    'garden': 'gdn', 'gdn': 'gdn', 'gardn': 'gdn', 'grden': 'gdn', 'grdn': 'gdn',
    'gardens': 'gdns', 'gdns': 'gdns', 'grdns': 'gdns',
    'gateway': 'gtwy', 'gtwy': 'gtwy', 'gatewy': 'gtwy', 'gatway': 'gtwy', 'gtway': 'gtwy',
    'glen': 'gln', 'gln': 'gln', 'green': 'grn', 'grn': 'grn',
    'grove': 'grv', 'grv': 'grv', 'grov': 'grv', 'groves': 'grvs',
    'harbor': 'hbr', 'hbr': 'hbr', 'harb': 'hbr', 'harbr': 'hbr', 'hrbor': 'hbr', 'harbour': 'hbr',
    'haven': 'hvn', 'hvn': 'hvn', 'heights': 'hts', 'hts': 'hts', 'ht': 'hts', 'height': 'hts',
    'highway': 'hwy', 'hwy': 'hwy', 'highwy': 'hwy', 'hiway': 'hwy', 'hiwy': 'hwy', 'hway': 'hwy',
    'highways': 'hwy',
    'hill': 'hl', 'hl': 'hl', 'hills': 'hls', 'hls': 'hls',
    'hollow': 'holw', 'holw': 'holw', 'hllw': 'holw', 'hollows': 'holw',
    'inlet': 'inlt', 'inlt': 'inlt',
    'junction': 'jct', 'jct': 'jct', 'jction': 'jct', 'jctn': 'jct', 'junctn': 'jct', 'juncton': 'jct',
    'junctions': 'jcts', 'jcts': 'jcts',
    'knoll': 'knl', 'knl': 'knl', 'knolls': 'knls', 'knls': 'knls',
    'lake': 'lk', 'lk': 'lk', 'lakes': 'lks', 'lks': 'lks',
    'landing': 'lndg', 'lndg': 'lndg', 'lndng': 'lndg',
    'lane': 'ln', 'ln': 'ln', 'lne': 'ln', 'lanes': 'ln',
    'light': 'lgt', 'lgt': 'lgt', 'lights': 'lgts', 'lgts': 'lgts',
    'loaf': 'lf', 'lf': 'lf', 'lock': 'lck', 'lck': 'lck', 'locks': 'lcks', 'lcks': 'lcks',
    'lodge': 'ldg', 'ldg': 'ldg', 'ldge': 'ldg', 'loop': 'loop', 'loops': 'loop',
    'mall': 'mall', 'manor': 'mnr', 'mnr': 'mnr', 'manors': 'mnrs', 'mnrs': 'mnrs',
    'meadow': 'mdw', 'mdw': 'mdw', 'meadows': 'mdws', 'mdws': 'mdws', 'medows': 'mdws',
    'mill': 'ml', 'ml': 'ml', 'mills': 'mls', 'mls': 'mls',
    'mission': 'msn', 'msn': 'msn', 'missn': 'msn', 'motorway': 'mtwy', 'mtwy': 'mtwy',
    'mount': 'mt', 'mt': 'mt', 'mnt': 'mt', 'mont': 'mt',
    'mountain': 'mtn', 'mtn': 'mtn', 'mntain': 'mtn', 'mntn': 'mtn', 'mountin': 'mtn', 'mtin': 'mtn',
    'mountains': 'mtns', 'mtns': 'mtns',
    'neck': 'nck', 'nck': 'nck',
    'orchard': 'orch', 'orch': 'orch', 'orchrd': 'orch', 'oval': 'oval', 'ovl': 'oval',
    'overpass': 'opas', 'opas': 'opas',
    'park': 'park', 'prk': 'park', 'parks': 'park',
    'parkway': 'pkwy', 'pkwy': 'pkwy', 'parkwy': 'pkwy', 'pkway': 'pkwy', 'pky': 'pkwy',
    'parkways': 'pkwy', 'pkwys': 'pkwy',
    'pass': 'pass', 'path': 'path', 'paths': 'path', 'pike': 'pike', 'pikes': 'pike',
    'pine': 'pne', 'pne': 'pne', 'pines': 'pnes', 'pnes': 'pnes',
    'place': 'pl', 'pl': 'pl', 'plc': 'pl', 'plain': 'pln', 'pln': 'pln', 'plains': 'plns', 'plns': 'plns',
    'plaza': 'plz', 'plz': 'plz', 'plza': 'plz',
    'point': 'pt', 'pt': 'pt', 'points': 'pts', 'pts': 'pts',
    'port': 'prt', 'prt': 'prt', 'ports': 'prts', 'prts': 'prts',
    'prairie': 'pr', 'prr': 'pr',
    'radial': 'radl', 'radl': 'radl', 'rad': 'radl', 'ramp': 'ramp',
    'ranch': 'rnch', 'rnch': 'rnch', 'ranches': 'rnch', 'rnchs': 'rnch',
    'rapids': 'rpds', 'rpds': 'rpds', 'rest': 'rst', 'rst': 'rst',
    'ridge': 'rdg', 'rdg': 'rdg', 'rdge': 'rdg', 'ridges': 'rdgs', 'rdgs': 'rdgs',
    'river': 'riv', 'riv': 'riv', 'rvr': 'riv', 'rivr': 'riv',
    'road': 'rd', 'rd': 'rd', 'raod': 'rd', 'roda': 'rd', 'roads': 'rds', 'rds': 'rds',
    'route': 'rte', 'rte': 'rte', 'rt': 'rte', 'row': 'row', 'rue': 'r', 'run': 'run',
    'shoal': 'shl', 'shl': 'shl', 'shoals': 'shls', 'shls': 'shls',
    'shore': 'shr', 'shr': 'shr', 'shoar': 'shr', 'shores': 'shrs', 'shrs': 'shrs',
    'skyway': 'skwy', 'skwy': 'skwy',
    'spring': 'spg', 'spg': 'spg', 'spng': 'spg', 'sprng': 'spg', 'springs': 'spgs', 'spgs': 'spgs',
    'spur': 'spur', 'spurs': 'spur',
    'square': 'sq', 'sq': 'sq', 'sqr': 'sq', 'sqre': 'sq', 'squ': 'sq', 'squares': 'sqs', 'sqs': 'sqs',
    'station': 'stn', 'stn': 'stn', 'sta': 'stn', 'statn': 'stn',
    'stravenue': 'stra', 'stra': 'stra', 'stream': 'strm', 'strm': 'strm',
    'street': 'st', 'st': 'st', 'str': 'st', 'strt': 'st', 'stree': 'st', 'stret': 'st', 'sreet': 'st',
    'streets': 'sts', 'sts': 'sts',
    'summit': 'smt', 'smt': 'smt', 'sumit': 'smt', 'sumitt': 'smt',
    'terrace': 'ter', 'ter': 'ter', 'terr': 'ter', 'trce': 'trce', 'trace': 'trce',
    'throughway': 'trwy', 'trwy': 'trwy',
    'track': 'trak', 'trak': 'trak', 'trk': 'trak', 'trks': 'trak', 'tracks': 'trak',
    'trafficway': 'trfy', 'trfy': 'trfy',
    'trail': 'trl', 'trl': 'trl', 'trails': 'trl', 'trls': 'trl',
    'trailer': 'trlr', 'trlr': 'trlr', 'tunnel': 'tunl', 'tunl': 'tunl', 'tunnels': 'tunl',
    'turnpike': 'tpke', 'tpke': 'tpke', 'trnpk': 'tpke', 'turnpk': 'tpke',
    'underpass': 'upas', 'upas': 'upas',
    'valley': 'vly', 'vly': 'vly', 'vally': 'vly', 'vlly': 'vly', 'valleys': 'vlys', 'vlys': 'vlys',
    'viaduct': 'via', 'via': 'via', 'vdct': 'via',
    'view': 'vw', 'vw': 'vw', 'views': 'vws', 'vws': 'vws',
    'ville': 'vl', 'vl': 'vl', 'vista': 'vis', 'vis': 'vis', 'vist': 'vis', 'vst': 'vis', 'vsta': 'vis',
    'walk': 'walk', 'walks': 'walk', 'wall': 'wall', 'way': 'way', 'wy': 'way', 'ways': 'way',
    'well': 'wl', 'wl': 'wl', 'wells': 'wls', 'wls': 'wls',
    # --- USPS secondary unit designators ---
    'apartment': 'apt', 'apt': 'apt', 'apartments': 'apt', 'apts': 'apt', 'aprt': 'apt',
    'appartment': 'apt', 'appt': 'apt', 'appartement': 'apt', 'app': 'apt', 'apartement': 'apt',
    'basement': 'bsmt', 'bsmt': 'bsmt', 'bsmnt': 'bsmt', 'bsm': 'bsmt',
    'building': 'bldg', 'bldg': 'bldg', 'bldng': 'bldg', 'buildng': 'bldg', 'buliding': 'bldg',
    'department': 'dept', 'dept': 'dept', 'deptt': 'dept',
    'floor': 'fl', 'fl': 'fl', 'flr': 'fl', 'flor': 'fl', 'floors': 'fl',
    'front': 'frnt', 'frnt': 'frnt', 'hangar': 'hngr', 'hngr': 'hngr',
    'lobby': 'lbby', 'lbby': 'lbby', 'lot': 'lot', 'lots': 'lot', 'lower': 'lowr', 'lowr': 'lowr',
    'office': 'ofc', 'ofc': 'ofc', 'offc': 'ofc', 'offices': 'ofc',
    'pier': 'pier', 'rear': 'rear', 'room': 'rm', 'rm': 'rm', 'rooms': 'rm',
    'side': 'side', 'slip': 'slip', 'space': 'spc', 'spc': 'spc', 'stop': 'stop',
    'suite': 'ste', 'ste': 'ste', 'suites': 'ste', 'sute': 'ste', 'suit': 'ste',
    'unit': 'unit', 'units': 'unit', 'upper': 'uppr', 'uppr': 'uppr',
    'pmb': 'pmb', 'private mailbox': 'pmb', 'box': 'box', 'bx': 'box',
    'po': 'po', 'p o': 'po', 'po box': 'po box', 'p o box': 'po box', 'pob': 'po box',
    'post office box': 'po box', 'post box': 'po box', 'postbox': 'po box',
    'rural route': 'rr', 'r r': 'rr', 'rr': 'rr', 'highway contract': 'hc', 'hc': 'hc',
    'county road': 'cr', 'co rd': 'cr', 'cnty rd': 'cr', 'county rd': 'cr',
    'farm to market': 'fm', 'us highway': 'us hwy', 'us hwy': 'us hwy', 'u s highway': 'us hwy',
    # --- directionals ---
    'north': 'n', 'n': 'n', 'nth': 'n', 'south': 's', 's': 's', 'sth': 's', 'east': 'e', 'e': 'e',
    'west': 'w', 'w': 'w',
    'northeast': 'ne', 'north east': 'ne', 'ne': 'ne', 'n e': 'ne',
    'northwest': 'nw', 'north west': 'nw', 'nw': 'nw', 'n w': 'nw',
    'southeast': 'se', 'south east': 'se', 'se': 'se', 's e': 'se',
    'southwest': 'sw', 'south west': 'sw', 'sw': 'sw', 's w': 'sw',
    # --- misc ---
    'township': 'twp', 'twp': 'twp', 'twsp': 'twp', 'saint': 'st', 'sainte': 'ste',
    'number': 'no', 'no': 'no', 'num': 'no', 'nos': 'no', 'nmbr': 'no', 'nbr': 'no', 'numero': 'no',
    'zip': 'zip', 'zipcode': 'zip', 'zip code': 'zip', 'zp': 'zip',
    'c/o': 'co', 'c o': 'co', 'care of': 'co', 'co': 'co',
    'attention': 'attn', 'attn': 'attn', 'att': 'attn',
}

_ADDR_FR = {
    # --- La Poste / AFNOR voie types (canonical picked to unify with US where
    #     the same word exists: avenue->ave, boulevard->blvd, route->rte) ---
    'rue': 'r', 'r': 'r', 'rues': 'r',
    'allee': 'all', 'all': 'all', 'alle': 'all', 'allees': 'all',
    'impasse': 'imp', 'imp': 'imp', 'impasses': 'imp', 'impase': 'imp',
    'chemin': 'che', 'che': 'che', 'ch': 'che', 'chem': 'che', 'chemins': 'che', 'chm': 'che',
    'passage': 'pas', 'pas': 'pas', 'psge': 'pas', 'pass age': 'pas',
    'quai': 'qu', 'qu': 'qu', 'quais': 'qu',
    'cours': 'crs', 'crs': 'crs',
    'faubourg': 'fg', 'fg': 'fg', 'fbg': 'fg', 'fbrg': 'fg',
    'lieu dit': 'ld', 'ld': 'ld', 'lieudit': 'ld', 'lieux dits': 'ld',
    'residence': 'res', 'res': 'res', 'resid': 'res', 'residences': 'res',
    'batiment': 'bat', 'bat': 'bat', 'batiments': 'bat', 'batim': 'bat',
    'zone artisanale': 'za', 'za': 'za', 'zone d activite': 'za', 'zone d activites': 'za',
    'zone dactivite': 'za', 'zone dactivites': 'za', 'zone activite': 'za', 'zone activites': 'za',
    'zone industrielle': 'zi', 'zi': 'zi', 'zone indus': 'zi', 'zone ind': 'zi',
    'zac': 'zac', 'zone d amenagement concerte': 'zac', 'zae': 'zae', 'zone d activite economique': 'zae',
    'zone d activites economiques': 'zae', 'zone commerciale': 'zc', 'zc': 'zc',
    'parc d activite': 'pa', 'parc d activites': 'pa', 'parc dactivites': 'pa',
    'esplanade': 'esp', 'esp': 'esp',
    'sentier': 'sen', 'sen': 'sen', 'sente': 'sen',
    'villa': 'vla', 'vla': 'vla', 'villas': 'vla',
    'hameau': 'ham', 'ham': 'ham', 'hameaux': 'ham',
    'lotissement': 'lot', 'lotis': 'lot', 'loti': 'lot',
    'cite': 'cite', 'cites': 'cite',
    'montee': 'mte', 'mte': 'mte',
    'promenade': 'prom', 'prom': 'prom',
    'rond point': 'rpt', 'rpt': 'rpt', 'rond pt': 'rpt', 'rd pt': 'rpt', 'rondpoint': 'rpt',
    'carrefour': 'carr', 'carr': 'carr',
    'traverse': 'tra', 'tra': 'tra', 'venelle': 'ven', 'ven': 'ven', 'ruelle': 'rle', 'rle': 'rle',
    'boite postale': 'bp', 'bp': 'bp', 'cs': 'cs', 'course speciale': 'cs',
    'cedex': 'cedex', 'cdx': 'cedex',
    'bis': 'bis', 'ter': 'ter', 'quater': 'qua', 'qua': 'qua', 'quinquies': 'quinq',
    'centre commercial': 'cc', 'ccial': 'cc', 'ctre cial': 'cc', 'ccal': 'cc',
    'immeuble': 'imm', 'imm': 'imm', 'immeubles': 'imm',
    'escalier': 'esc', 'esc': 'esc', 'etage': 'etg', 'etg': 'etg', 'etages': 'etg',
    'porte': 'pte', 'pte': 'pte', 'entree': 'ent', 'ent': 'ent',
    'domaine': 'dom', 'dom': 'dom', 'chateau': 'chat', 'chat': 'chat',
    'grand': 'gd', 'gd': 'gd', 'grande': 'gde', 'gde': 'gde', 'grands': 'gds', 'grandes': 'gdes',
    'vieux': 'vx', 'vx': 'vx', 'vieille': 'vle', 'vle': 'vle', 'ancien': 'anc', 'ancienne': 'anc',
    'anc': 'anc',
    'route nationale': 'rn', 'rn': 'rn', 'rte nationale': 'rn', 'nationale': 'nale', 'nale': 'nale',
    'chemin departemental': 'cd', 'cd': 'cd', 'voie communale': 'vc', 'vc': 'vc',
    'zone': 'zone', 'parc': 'parc', 'pont': 'pont', 'port': 'prt',
    # --- honorifics / titles inside street names ---
    'general': 'gal', 'gal': 'gal', 'gen': 'gal', 'gnl': 'gal',
    'marechal': 'mal', 'mal': 'mal',
    'docteur': 'dr', 'doct': 'dr', 'doctor': 'dr',
    'professeur': 'pr', 'pr': 'pr', 'prof': 'pr', 'professor': 'pr',
    'commandant': 'cdt', 'cdt': 'cdt', 'president': 'pdt', 'pdt': 'pdt',
    'capitaine': 'cne', 'cne': 'cne', 'captain': 'cne', 'capt': 'cne',
    'colonel': 'col', 'col': 'col', 'lieutenant': 'lt', 'lt': 'lt', 'lieut': 'lt',
    'monsieur': 'm', 'm': 'm', 'madame': 'mme', 'mme': 'mme', 'mademoiselle': 'mlle', 'mlle': 'mlle',
    'notre dame': 'nd', 'nd': 'nd', 'saint': 'st', 'sainte': 'ste', 'saints': 'sts', 'saintes': 'stes',
    'stes': 'stes',
}

_ADDR_IN = {
    # --- Indian address abbreviations ---
    'house': 'h', 'h': 'h', 'hno': 'h', 'h no': 'h', 'house no': 'h', 'house number': 'h',
    'hn': 'h', 'hs no': 'h', 'h n': 'h', 'hse': 'h', 'hse no': 'h', 'house nos': 'h', 'hous no': 'h',
    'flat': 'flat', 'flat no': 'flat', 'fl no': 'flat', 'fno': 'flat', 'f no': 'flat', 'flt': 'flat',
    'flat number': 'flat', 'flats': 'flat', 'flt no': 'flat', 'flatno': 'flat',
    'plot': 'plot', 'plot no': 'plot', 'pno': 'plot', 'p no': 'plot', 'plt': 'plot', 'plt no': 'plot',
    'plot number': 'plot', 'plots': 'plot', 'plotno': 'plot',
    'shop': 'shop', 'shop no': 'shop', 'sno': 'shop', 's no': 'shop', 'shp': 'shop', 'shp no': 'shop',
    'shop number': 'shop', 'shops': 'shop', 'shopno': 'shop',
    'door': 'door', 'door no': 'door', 'dno': 'door', 'd no': 'door', 'door number': 'door',
    'dr no': 'door', 'doorno': 'door',
    'gala': 'gala', 'gala no': 'gala', 'gala number': 'gala', 'galla': 'gala', 'galla no': 'gala',
    'khasra': 'khasra', 'khasra no': 'khasra', 'khasra number': 'khasra', 'kh no': 'khasra',
    'khata': 'khata', 'khata no': 'khata', 'khewat': 'khewat', 'khewat no': 'khewat',
    'gat': 'gat', 'gat no': 'gat', 'gat number': 'gat', 'gut no': 'gat',
    'survey': 'survey', 'survey no': 'survey', 'sy no': 'survey', 'sr no': 'survey', 'sur no': 'survey',
    'survey number': 'survey', 'sy': 'survey', 'syno': 'survey', 'srno': 'survey', 'surve no': 'survey',
    'cts': 'cts', 'cts no': 'cts', 'c t s no': 'cts', 'city survey no': 'cts',
    'ward': 'ward', 'ward no': 'ward', 'ward number': 'ward', 'wd no': 'ward',
    'godown': 'godown', 'godown no': 'godown', 'godwn': 'godown', 'shed': 'shed', 'shed no': 'shed',
    'stall': 'stall', 'stall no': 'stall', 'booth': 'booth', 'booth no': 'booth',
    'sco': 'sco', 'sco no': 'sco', 'scf': 'scf', 'scf no': 'scf',
    'unit no': 'unit', 'unit number': 'unit', 'room no': 'rm', 'room number': 'rm', 'office no': 'ofc',
    'block no': 'blk', 'block number': 'blk', 'sector no': 'sec', 'sector number': 'sec',
    'phase no': 'ph', 'road no': 'rd', 'rd no': 'rd', 'street no': 'st', 'st no': 'st',
    'lane no': 'ln', 'gali no': 'gali', 'pocket no': 'pkt', 'floor no': 'fl',
    # floors
    'ground floor': 'gf', 'gf': 'gf', 'g f': 'gf', 'grnd floor': 'gf', 'ground flr': 'gf',
    'grd floor': 'gf', 'grnd flr': 'gf', 'ground fl': 'gf', 'groundfloor': 'gf', 'gr floor': 'gf',
    'gr fl': 'gf', 'ground': 'gf',
    'lower ground floor': 'lgf', 'lgf': 'lgf', 'lower ground': 'lgf',
    'upper ground floor': 'ugf', 'ugf': 'ugf', 'upper ground': 'ugf',
    'mezzanine': 'mezz', 'mezz': 'mezz', 'mezzanine floor': 'mezz', 'mezanine': 'mezz',
    'top floor': 'top fl', 'terrace floor': 'ter fl',
    # street / locality descriptors
    'marg': 'marg', 'gali': 'gali', 'galli': 'gali', 'gully': 'gali', 'gulli': 'gali', 'gali no': 'gali',
    'cross': 'crs', 'crss': 'crs', 'main': 'main', 'main road': 'main rd', 'main rd': 'main rd',
    'mn rd': 'main rd', 'mainroad': 'main rd', 'link road': 'link rd', 'link rd': 'link rd',
    'ring road': 'ring rd', 'ring rd': 'ring rd', 'bypass road': 'byp', 'by pass': 'byp',
    'by pass road': 'byp', 'service road': 'service rd', 'service rd': 'service rd',
    'nagar': 'nagar', 'nagr': 'nagar', 'ngr': 'nagar', 'nager': 'nagar', 'nagar ': 'nagar',
    'colony': 'col', 'cly': 'col', 'colny': 'col', 'coloni': 'col', 'colony ': 'col',
    'sector': 'sec', 'sec': 'sec', 'sect': 'sec', 'sctr': 'sec', 'sectr': 'sec', 'sectors': 'sec',
    'phase': 'ph', 'ph': 'ph', 'phs': 'ph', 'phase ': 'ph',
    'block': 'blk', 'blk': 'blk', 'blck': 'blk', 'bl': 'blk', 'blocks': 'blk',
    'pocket': 'pkt', 'pkt': 'pkt', 'pckt': 'pkt',
    'peth': 'peth', 'pura': 'pura', 'puram': 'puram', 'wadi': 'wadi', 'para': 'para', 'pada': 'pada',
    'chowk': 'chowk', 'chauk': 'chowk', 'chok': 'chowk', 'chowck': 'chowk', 'chawk': 'chowk',
    'bazar': 'bazar', 'bazaar': 'bazar', 'bazzar': 'bazar', 'market': 'mkt', 'mkt': 'mkt', 'mkt ': 'mkt',
    'complex': 'cmplx', 'cmplx': 'cmplx', 'complx': 'cmplx', 'cmplex': 'cmplx', 'complexe': 'cmplx',
    'society': 'soc', 'soc': 'soc', 'socy': 'soc', 'scty': 'soc', 'socity': 'soc', 'societies': 'soc',
    'chs': 'chs', 'chsl': 'chs', 'chs ltd': 'chs', 'co op hsg soc': 'chs', 'co op housing society': 'chs',
    'cooperative housing society': 'chs', 'co operative housing society': 'chs', 'coop hsg soc': 'chs',
    'cooperative': 'coop', 'co operative': 'coop', 'co op': 'coop', 'coop': 'coop', 'coopertive': 'coop',
    'housing': 'hsg', 'hsg': 'hsg', 'hous': 'hsg',
    'industrial': 'indl', 'indl': 'indl', 'ind': 'indl', 'indus': 'indl', 'indst': 'indl', 'indl ': 'indl',
    'industrial area': 'indl area', 'indl area': 'indl area', 'ind area': 'indl area',
    'industrial estate': 'indl est', 'indl est': 'indl est', 'ind est': 'indl est', 'ind estate': 'indl est',
    'midc': 'midc', 'gidc': 'gidc', 'sidco': 'sidco', 'sipcot': 'sipcot', 'kiadb': 'kiadb',
    'area': 'area', 'areas': 'area',
    'layout': 'layout', 'lyt': 'layout', 'lay out': 'layout', 'layot': 'layout',
    'stage': 'stg', 'stg': 'stg', 'stge': 'stg',
    'enclave': 'encl', 'encl': 'encl', 'enclv': 'encl', 'vihar': 'vihar', 'kunj': 'kunj',
    'tower': 'twr', 'twr': 'twr', 'towers': 'twr', 'twrs': 'twr', 'towr': 'twr',
    'bhavan': 'bhavan', 'bhawan': 'bhavan', 'bhuvan': 'bhavan', 'sadan': 'sadan', 'niwas': 'niwas',
    'nivas': 'niwas', 'mahal': 'mahal', 'manzil': 'manzil', 'chawl': 'chawl', 'chaal': 'chawl',
    'compound': 'cmpd', 'cmpd': 'cmpd', 'compd': 'cmpd', 'comp': 'cmpd', 'compund': 'cmpd',
    'warehouse': 'whse', 'whse': 'whse', 'ware house': 'whse',
    'mandir': 'mandir', 'temple': 'temple', 'masjid': 'masjid', 'gurudwara': 'gurudwara',
    'gurdwara': 'gurudwara', 'church': 'church', 'dargah': 'dargah',
    'hospital': 'hosp', 'hosp': 'hosp', 'hospt': 'hosp', 'hospitl': 'hosp',
    'college': 'clg', 'clg': 'clg', 'colg': 'clg', 'university': 'univ', 'univ': 'univ',
    'railway': 'rly', 'rly': 'rly', 'rlwy': 'rly', 'rail': 'rly', 'railway station': 'rly stn',
    'rly stn': 'rly stn', 'rly station': 'rly stn', 'railway stn': 'rly stn', 'rly st': 'rly stn',
    'bus stand': 'bus stand', 'bus stop': 'bus stop', 'bus depot': 'bus depot', 'bus station': 'bus stand',
    'metro station': 'metro stn', 'metro stn': 'metro stn',
    'petrol pump': 'petrol pump', 'petrol bunk': 'petrol pump', 'gas station': 'petrol pump',
    'police station': 'ps', 'ps': 'ps', 'police stn': 'ps', 'p s': 'ps',
    'post office': 'po', 'post': 'po', 'p o': 'po', 'po': 'po', 'post off': 'po', 'p office': 'po',
    'pin': 'pin', 'pincode': 'pin', 'pin code': 'pin', 'pin no': 'pin', 'pinno': 'pin', 'pin ': 'pin',
    # relational
    'opposite': 'opp', 'opp': 'opp', 'oppo': 'opp', 'oppst': 'opp', 'opposit': 'opp', 'oppsite': 'opp',
    'opp to': 'opp', 'opposite to': 'opp', 'oposite': 'opp',
    'near': 'nr', 'nr': 'nr', 'nera': 'nr', 'neer': 'nr', 'nearby': 'nr', 'near by': 'nr', 'nr to': 'nr',
    'near to': 'nr', 'nier': 'nr', 'nr ': 'nr',
    'behind': 'behind', 'bhd': 'behind', 'beh': 'behind', 'behnd': 'behind', 'back of': 'behind',
    'backside': 'behind', 'back side': 'behind', 'behind of': 'behind', 'bhind': 'behind',
    'beside': 'beside', 'besides': 'beside', 'bsd': 'beside', 'next to': 'beside', 'adjacent': 'adj',
    'adj': 'adj', 'adjacent to': 'adj', 'adjoining': 'adj', 'adjoining to': 'adj',
    'above': 'above', 'abv': 'above', 'below': 'below', 'inside': 'inside', 'in front of': 'infront',
    'infront': 'infront', 'infront of': 'infront', 'in front': 'infront',
    'landmark': 'landmark', 'land mark': 'landmark', 'lmk': 'landmark',
    # administrative
    'district': 'dist', 'dist': 'dist', 'distt': 'dist', 'dt': 'dist', 'dst': 'dist', 'districts': 'dist',
    'dis': 'dist', 'distric': 'dist', 'distt ': 'dist',
    'tehsil': 'teh', 'teh': 'teh', 'tehsel': 'teh', 'tahsil': 'teh', 'tehsil ': 'teh',
    'taluka': 'tal', 'taluk': 'tal', 'tal': 'tal', 'tq': 'tal', 'tk': 'tal', 'tlk': 'tal', 'taluq': 'tal',
    'talluk': 'tal', 'taluko': 'tal', 'tal ': 'tal',
    'village': 'vill', 'vill': 'vill', 'vlg': 'vill', 'vil': 'vill', 'villag': 'vill', 'villg': 'vill',
    'villiage': 'vill', 'villages': 'vill', 'vill ': 'vill', 'gram': 'vill', 'gaon': 'vill', 'gao': 'vill',
    'vpo': 'vpo', 'v p o': 'vpo', 'village and post office': 'vpo', 'village post office': 'vpo',
    'vill po': 'vpo', 'vill and po': 'vpo', 'v and po': 'vpo',
    'mandal': 'mandal', 'mdl': 'mandal', 'mandl': 'mandal', 'mandal ': 'mandal',
    'municipal': 'mun', 'mun': 'mun', 'muncipal': 'mun', 'municipality': 'mun', 'nagar palika': 'mun',
    'nagar nigam': 'mun', 'mahanagar palika': 'mun', 'municipal corporation': 'mun',
    'panchayat': 'panchayat', 'gram panchayat': 'panchayat', 'panchayath': 'panchayat',
    'town': 'town', 'city': 'city', 'cty': 'city', 'metro': 'metro',
    'pargana': 'pargana', 'circle': 'cir', 'sub division': 'sub div', 'subdivision': 'sub div',
    'sub divn': 'sub div', 'sub dist': 'sub dist', 'sub district': 'sub dist',
    # relations
    's/o': 'so', 's o': 'so', 'son of': 'so', 'd/o': 'do', 'd o': 'do', 'daughter of': 'do',
    'w/o': 'wo', 'w o': 'wo', 'wife of': 'wo', 'h/o': 'ho', 'husband of': 'ho',
    'prop': 'prop', 'proprietor': 'prop', 'propritor': 'prop', 'prop ': 'prop',
    # misc
    'extension': 'extn', 'extn': 'extn', 'ext': 'extn', 'extention': 'extn', 'extenstion': 'extn',
    'old': 'old', 'new': 'new', 'east': 'e', 'west': 'w', 'north': 'n', 'south': 's',
    'jn': 'jct', 'jnc': 'jct', 'junc': 'jct', 'juncn': 'jct',
    'road': 'rd', 'rd': 'rd', 'rode': 'rd', 'roadd': 'rd', 'rd ': 'rd', 'rasta': 'rd',
    'street': 'st', 'lane': 'ln', 'apartment': 'apt', 'floor': 'fl', 'no': 'no', 'number': 'no',
    'num': 'no', 'floor ': 'fl',
}

# Merge with conflict detection: US, then FR, then IN. Any key present in two
# sub-dicts MUST carry the same value (checked at import time) so that a
# conflict is an explicit decision, never an accident of dict-literal order.
ADDR_ABBREV = {}
for _sub in (_ADDR_US, _ADDR_FR, _ADDR_IN):
    for _k, _v in _sub.items():
        _k = _k.strip()
        if _k in ADDR_ABBREV and ADDR_ABBREV[_k] != _v:
            raise ValueError('ADDR_ABBREV conflict for %r: %r vs %r' % (_k, ADDR_ABBREV[_k], _v))
        ADDR_ABBREV[_k] = _v
ADDR_ABBREV_MULTIWORD = sorted((k for k in ADDR_ABBREV if ' ' in k), key=lambda k: (-len(k), k))
"""Address token / phrase -> canonical SHORT token (lowercase, no punctuation).
Apply multi-word keys first (see `ADDR_ABBREV_MULTIWORD`, longest-first),
then single tokens. Covers USPS Publication 28 street suffixes + secondary
unit designators + directionals, French La Poste / AFNOR voie types and
honorifics, and Indian address abbreviations.

Cross-language conflict resolutions (UNIFICATION over correctness):
* 'avenue': US 'ave' vs FR 'av'  -> both 'ave' ('av' -> 'ave').
* 'boulevard': US 'blvd' vs FR 'bd' -> both 'blvd' ('bd','bld','bvd' -> 'blvd';
  'bld' is treated as French boulevard, NOT 'building').
* 'allee': USPS lists it as an 'alley' variant, French allee is a voie type
  -> 'all' (French sense). 'alley' -> 'aly'.
* 'apartment'/'appartement'/'appt'/'app' -> single canonical 'apt'.
* 'saint' -> 'st' (collides with 'street' -> 'st'; accepted). 'sainte' -> 'ste'
  which collides with 'suite' -> 'ste' (accepted: each surface variant still
  unifies with its own abbreviation: sainte~ste, suite~ste, saint~st,
  street~st).
* 'colonel' (FR title) and 'colony' (IN) -> both 'col'.
* 'docteur'/'doctor' -> 'dr' (collides with 'drive'; accepted).
* 'cross' (IN) and 'cours' (FR) -> both 'crs'.
* 'passage': USPS 'psge' vs FR 'pas' -> 'pas'.
* 'village': USPS 'vlg' vs IN 'vill' -> 'vill'. 'extension': USPS 'ext' vs IN
  'extn' -> 'extn'. 'flat': USPS 'flt' vs IN 'flat' -> 'flat'.
  'station': USPS 'sta' vs IN 'stn' -> 'stn'.
* 'ch' -> 'che' (chemin); 'ground' -> 'gf'; 'th' -> 'teh' (tehsil); 'dt' ->
  'dist'; 'sno'/'s no' -> 'shop' (may also mean survey number).
* 'prairie' and 'professeur' -> 'pr'; 'no'/'number'/'numero' -> 'no'.
* Tokens like 'lot' (US lot / FR lotissement), 'est' (estate), 'm'
  (monsieur), 'e'/'w'/'n'/'s' are left as themselves.
Numbers-as-words ('first floor') are handled by ORDINALS, not here."""

# ---------------------------------------------------------------------------
# 5. Ordinal / cardinal number words -> digit strings
# ---------------------------------------------------------------------------
_EN_UNITS = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
_EN_UNITS_ORD = ['', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
                 'eighth', 'ninth']
_EN_TEENS = {10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
             15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen'}
_EN_TEENS_ORD = {10: 'tenth', 11: 'eleventh', 12: 'twelfth', 13: 'thirteenth',
                 14: 'fourteenth', 15: 'fifteenth', 16: 'sixteenth', 17: 'seventeenth',
                 18: 'eighteenth', 19: 'nineteenth'}
_EN_TENS = {20: 'twenty', 30: 'thirty', 40: 'forty', 50: 'fifty', 60: 'sixty',
            70: 'seventy', 80: 'eighty', 90: 'ninety'}
_EN_TENS_ORD = {20: 'twentieth', 30: 'thirtieth', 40: 'fortieth', 50: 'fiftieth',
                60: 'sixtieth', 70: 'seventieth', 80: 'eightieth', 90: 'ninetieth'}

ORDINALS = {
    # French ordinals
    'deuxieme': '2', 'second': '2', 'seconde': '2', 'troisieme': '3', 'quatrieme': '4',
    'cinquieme': '5', 'sixieme': '6', 'septieme': '7', 'huitieme': '8', 'neuvieme': '9',
    'dixieme': '10', 'onzieme': '11', 'douzieme': '12', 'treizieme': '13', 'quatorzieme': '14',
    'quinzieme': '15', 'seizieme': '16', 'dix septieme': '17', 'dix huitieme': '18',
    'dix neuvieme': '19', 'vingtieme': '20', 'trentieme': '30', 'centieme': '100',
    # French cardinals (un/une/neuf/cent/premier are in ORDINALS_RISKY)
    'deux': '2', 'trois': '3', 'quatre': '4', 'cinq': '5', 'six': '6', 'sept': '7', 'huit': '8',
    'dix': '10', 'onze': '11', 'douze': '12', 'treize': '13', 'quatorze': '14', 'quinze': '15',
    'seize': '16', 'dix sept': '17', 'dix huit': '18', 'dix neuf': '19', 'vingt': '20',
    'trente': '30', 'quarante': '40', 'cinquante': '50', 'soixante': '60', 'mille': '1000',
    # English extras / misspellings
    'zero': '0', 'hundred': '100', 'hundredth': '100', 'thousand': '1000', 'twelveth': '12',
    'fourth': '4', 'forth': '4', 'nineth': '9', 'fourty': '40', 'fourtieth': '40',
    'fivth': '5', 'eigth': '8', 'eigthth': '8', 'ninteen': '19',
}
for _n in range(1, 10):
    ORDINALS[_EN_UNITS[_n]] = str(_n)
    ORDINALS[_EN_UNITS_ORD[_n]] = str(_n)
for _n in range(10, 20):
    ORDINALS[_EN_TEENS[_n]] = str(_n)
    ORDINALS[_EN_TEENS_ORD[_n]] = str(_n)
for _t, _tw in _EN_TENS.items():
    ORDINALS[_tw] = str(_t)
    ORDINALS[_EN_TENS_ORD[_t]] = str(_t)
    for _u in range(1, 10):
        ORDINALS['%s %s' % (_tw, _EN_UNITS[_u])] = str(_t + _u)
        ORDINALS['%s%s' % (_tw, _EN_UNITS[_u])] = str(_t + _u)
        ORDINALS['%s %s' % (_tw, _EN_UNITS_ORD[_u])] = str(_t + _u)
        ORDINALS['%s%s' % (_tw, _EN_UNITS_ORD[_u])] = str(_t + _u)
"""Number words -> digit strings. English cardinals/ordinals 0..99 (spaced and
joined forms, e.g. 'twenty first' and 'twentyfirst'), 'hundred(th)',
'thousand'; French ordinals 'deuxieme'..'vingtieme' and cardinals 'deux'..
'vingt', tens to 'soixante'. 'second' is shared by both languages (-> '2').
Apply to address tokens; applying to business names is optional (it unifies
'One Stop' / '1 Stop' but may also alter e.g. 'Seven Seas')."""

ORDINALS_RISKY = {
    'un': '1', 'une': '1', 'premier': '1', 'premiere': '1', '1er': '1', '1ere': '1',
    'neuf': '9', 'cent': '100',
}
"""Number words that are also common ordinary words (French articles 'un'/'une',
'premier' = English 'premier', 'neuf' = 'new', 'cent' = USPS 'center' variant).
Apply only when the token is directly followed by a street/floor keyword."""

ORDINAL_SUFFIX_RE_HINT = r'(?<=\d)(st|nd|rd|th|er|ere|eme|e)\b'
"""Regex for stripping ordinal suffixes glued to numbers: re.sub(HINT, '', s)
turns '71st'->'71', '2nd'->'2', '1er'->'1', '1ere'->'1', '2eme'->'2', '3e'->'3'.
Apply on lowercase text, and BEFORE token abbreviation so that '5th' does not
become '5 teh'."""

# ---------------------------------------------------------------------------
# 6. Legal-form suffixes
# ---------------------------------------------------------------------------
_LEGAL_RAW = {
    # US
    'limited liability company', 'limited liability partnership', 'limited partnership',
    'limited liability limited partnership', 'professional corporation',
    'professional association', 'professional limited liability company',
    'general partnership', 'incorporated', 'inc', 'corporation', 'corp', 'company', 'co',
    'llc', 'l l c', 'llp', 'l l p', 'lp', 'l p', 'lllp', 'pllc', 'pc', 'p c', 'pa', 'p a',
    'ltd', 'limited', 'plc', 'co ltd', 'co inc', 'corp inc', 'inc ltd', 'ltd co', 'chartered',
    'chtd', 'and co', 'and company', 'and sons', 'and son', 'and associates', 'and partners',
    'and bros', 'and brothers', 'and daughters', 'et al',
    # India
    'private limited', 'pvt ltd', 'pvt limited', 'private ltd', 'p ltd', 'pvt', 'private',
    'pvt lt', 'pvtltd', 'opc', 'one person company', 'opc private limited', 'opc pvt ltd',
    'public limited', 'public ltd', 'public limited company', 'proprietorship', 'proprietor',
    'proprietary', 'prop', 'sole proprietorship', 'partnership firm', 'huf',
    'hindu undivided family', 'brothers', 'bros', 'sons', 'and firm',
    # France
    'societe a responsabilite limitee', 'societe a responsabilite limitee unipersonnelle',
    'societe par actions simplifiee', 'societe par actions simplifiee unipersonnelle',
    'societe par actions simplifiee a associe unique', 'societe anonyme',
    'societe anonyme a conseil d administration', 'societe anonyme a directoire',
    'societe civile immobiliere', 'societe en nom collectif',
    'entreprise unipersonnelle a responsabilite limitee', 'entreprise individuelle',
    'entreprise individuelle a responsabilite limitee', 'societe civile de moyens',
    'societe civile professionnelle', 'groupement d interet economique',
    'societe cooperative de production', 'societe cooperative et participative',
    'societe d exercice liberal a responsabilite limitee',
    'societe d exercice liberal par actions simplifiee', 'societe en commandite par actions',
    'societe en commandite simple', 'societe civile de construction vente', 'societe civile',
    'societe cooperative', 'societe d economie mixte',
    'exploitation agricole a responsabilite limitee',
    'groupement agricole d exploitation en commun', 'gaec', 'societe en participation',
    'auto entrepreneur', 'micro entreprise', 'sarl', 's a r l', 'sas', 's a s', 'sasu',
    's a s u', 'sa', 's a', 'eurl', 'e u r l', 'sci', 's c i', 'snc', 's n c', 'scop', 'scp',
    'scm', 'selarl', 'selas', 'selafa', 'sel', 'gie', 'earl', 'ei', 'e i', 'eirl', 'sem', 'sca',
    'scs', 'sccv', 'sarlu', 'ste', 'societe', 'ets', 'etablissements', 'etablissement', 'etab',
    'cie', 'et cie', 'et compagnie', 'compagnie', 'et fils', 'et filles', 'et freres',
    'et associes', 'associes', 'et fils sarl', 'sa au capital', 'sarl au capital',
}
LEGAL_SUFFIXES = sorted(_LEGAL_RAW, key=lambda s: (0 if ' ' in s else 1, -len(s), s))
"""Legal-form suffixes (US, India, France), lowercase, punctuation-free.
Ordered: multi-word entries first, then longest-first, so the list can be
walked in order for greedy suffix stripping. Generic descriptors
('enterprises', 'services', ...) are NOT here -- see NAME_FILLER_WORDS."""

LEGAL_SUFFIX_DOTTED_RE_HINT = r'\b([a-z])(?:\.\s?([a-z]))+\.?\b'
r"""Matches dotted/spaced single-letter acronyms: 's.a.r.l.', 'e.u.r.l', 'l.l.c.',
'p.v.t', 's. a.'. Collapse with
`re.sub(HINT, lambda m: re.sub(r'[.\s]', '', m.group(0)), s)` BEFORE matching
LEGAL_SUFFIXES / LEGAL_SUFFIX_ABBREV."""

LEGAL_SUFFIX_TRAILING_DOT_RE_HINT = (
    r'\b(inc|ltd|pvt|corp|co|llc|llp|plc|sarl|sas|sasu|sa|eurl|sci|snc|ets|etabs|cie|ste|bros|'
    r'prop|opc|gmbh|ag|bv|nv|spa|srl|oy|ab|as|kk|pte|sdn|bhd|ply|pty|lp|pc|pllc|pa)\.'
)
"""Matches abbreviations followed by a period ('inc.', 'pvt.', 'ltd.', 'sarl.');
replace with group 1 to drop the dot."""

LEGAL_SUFFIX_ABBREV = {
    # US
    'limited liability company': 'llc', 'l l c': 'llc', 'llc': 'llc', 'ltd liability co': 'llc',
    'limited liability partnership': 'llp', 'l l p': 'llp', 'llp': 'llp',
    'limited partnership': 'lp', 'l p': 'lp', 'lp': 'lp',
    'limited liability limited partnership': 'lllp', 'lllp': 'lllp',
    'professional corporation': 'pc', 'p c': 'pc', 'pc': 'pc',
    'professional association': 'pa', 'p a': 'pa',
    'professional limited liability company': 'pllc', 'pllc': 'pllc',
    'incorporated': 'inc', 'incorporation': 'inc', 'inc': 'inc', 'incorp': 'inc',
    'corporation': 'corp', 'corp': 'corp', 'corpn': 'corp', 'company': 'co', 'co': 'co',
    'comp': 'co', 'compny': 'co', 'limited': 'ltd', 'ltd': 'ltd', 'lmtd': 'ltd', 'limtd': 'ltd',
    'lmt': 'ltd', 'limitd': 'ltd', 'public limited company': 'plc', 'plc': 'plc',
    'chartered': 'chtd', 'chtd': 'chtd',
    'and company': 'and co', 'and co': 'and co', 'and sons': 'and sons', 'and son': 'and sons',
    'sons': 'sons', 'and associates': 'and assoc', 'and associate': 'and assoc',
    'associates': 'assoc', 'associate': 'assoc', 'assoc': 'assoc', 'assocs': 'assoc',
    'and brothers': 'and bros', 'and bros': 'and bros', 'brothers': 'bros', 'bros': 'bros',
    'brother': 'bros', 'and partners': 'and partners', 'partners': 'partners',
    # India
    'private limited': 'pvt ltd', 'pvt limited': 'pvt ltd', 'private ltd': 'pvt ltd',
    'p ltd': 'pvt ltd', 'pvt ltd': 'pvt ltd', 'pvt lt': 'pvt ltd', 'pvtltd': 'pvt ltd',
    'pvt lmt': 'pvt ltd', 'pvt lmtd': 'pvt ltd', 'private limtd': 'pvt ltd',
    'private limited company': 'pvt ltd', 'private': 'pvt', 'pvt': 'pvt',
    'pvtt': 'pvt', 'privat': 'pvt', 'prvt': 'pvt', 'one person company': 'opc', 'opc': 'opc',
    'public limited': 'plc', 'public ltd': 'plc', 'proprietorship': 'prop', 'proprietor': 'prop',
    'proprietary': 'prop', 'prop': 'prop', 'proprietors': 'prop', 'sole proprietorship': 'prop',
    'hindu undivided family': 'huf', 'huf': 'huf', 'partnership firm': 'firm',
    # France (canonical: the usual French sigle; SARLU unified with EURL)
    'societe a responsabilite limitee': 'sarl', 's a r l': 'sarl', 'sarl': 'sarl',
    'societe a responsabilite limitee unipersonnelle': 'eurl', 'sarlu': 'eurl',
    'entreprise unipersonnelle a responsabilite limitee': 'eurl', 'e u r l': 'eurl', 'eurl': 'eurl',
    'societe par actions simplifiee': 'sas', 's a s': 'sas', 'sas': 'sas',
    'societe par actions simplifiee unipersonnelle': 'sasu', 's a s u': 'sasu', 'sasu': 'sasu',
    'societe par actions simplifiee a associe unique': 'sasu',
    'societe anonyme': 'sa', 's a': 'sa', 'sa': 'sa',
    'societe anonyme a conseil d administration': 'sa', 'societe anonyme a directoire': 'sa',
    'societe civile immobiliere': 'sci', 's c i': 'sci', 'sci': 'sci',
    'societe en nom collectif': 'snc', 's n c': 'snc', 'snc': 'snc',
    'entreprise individuelle': 'ei', 'e i': 'ei', 'ei': 'ei',
    'entreprise individuelle a responsabilite limitee': 'eirl', 'eirl': 'eirl',
    'societe civile de moyens': 'scm', 'scm': 'scm', 'societe civile professionnelle': 'scp',
    'scp': 'scp', 'groupement d interet economique': 'gie', 'gie': 'gie',
    'societe cooperative de production': 'scop', 'societe cooperative et participative': 'scop',
    'scop': 'scop', 'societe d exercice liberal a responsabilite limitee': 'selarl',
    'selarl': 'selarl', 'societe d exercice liberal par actions simplifiee': 'selas',
    'selas': 'selas', 'societe en commandite par actions': 'sca', 'sca': 'sca',
    'societe en commandite simple': 'scs', 'scs': 'scs',
    'societe civile de construction vente': 'sccv', 'sccv': 'sccv',
    'societe d economie mixte': 'sem', 'sem': 'sem',
    'exploitation agricole a responsabilite limitee': 'earl', 'earl': 'earl',
    'groupement agricole d exploitation en commun': 'gaec', 'gaec': 'gaec',
    'societe civile': 'sc', 'societe cooperative': 'scop',
    'etablissements': 'ets', 'etablissement': 'ets', 'etab': 'ets', 'etabs': 'ets', 'ets': 'ets',
    'etablissements ': 'ets', 'compagnie': 'cie', 'cie': 'cie', 'et compagnie': 'et cie',
    'et cie': 'et cie', 'societe': 'ste', 'ste': 'ste', 'sté': 'ste', 'soc': 'ste',
    'freres': 'freres', 'frs': 'freres', 'frere': 'freres', 'et freres': 'et freres',
    'et frs': 'et freres', 'fils': 'fils', 'et fils': 'et fils', 'filles': 'filles',
    'et filles': 'et filles', 'associes': 'assoc', 'et associes': 'et assoc', 'associe': 'assoc',
    'et assoc': 'et assoc',
}
"""Legal-form long forms / variants -> ONE canonical short token (used to
build a 'name_norm_canon' variant). Safe to apply as whole-phrase replacement,
longest-first (see `LEGAL_SUFFIX_ABBREV_KEYS`). 'associates' (EN) and
'associes' (FR) both -> 'assoc'; 'sarlu' -> 'eurl'; 'company'/'co' -> 'co'
(note: 'co' also appears in addresses as 'care of')."""
LEGAL_SUFFIX_ABBREV = {k.strip(): v for k, v in LEGAL_SUFFIX_ABBREV.items()}
LEGAL_SUFFIX_ABBREV_KEYS = sorted(LEGAL_SUFFIX_ABBREV, key=lambda s: (-len(s), s))
"""Keys of LEGAL_SUFFIX_ABBREV sorted longest-first for greedy replacement."""

# ---------------------------------------------------------------------------
# 7. Low-information business-name words
# ---------------------------------------------------------------------------
NAME_FILLER_WORDS = set("""
center centre centers centres services service solutions solution group groupe groups
holdings holding partners partner associates associes enterprises enterprise entreprise
entreprises international intl inter global national regional india bharat indian france
francais francaise french usa us america american the company co companies trading traders
trader industries industry industrial consultants consultant consultancy consulting conseil
conseils technologies technology tech systems system ventures venture distribution
distributors distributor participations developpement development developers developer
institut institute institution union comite amicale club ecole maison federation
association asso societe ste ets etablissements etablissement cie fils freres dr mr mrs ms
smt shri shree sri sree m/s messrs mme mlle m pvt ltd llc inc corp corporation incorporated
sarl sas sa eurl sasu sci ei eirl snc scop scp scm selarl selas gie earl sem sca scs sccv
opc llp plc pc pllc lp huf limited private new old and of de du des la le les l d et en
pour a an au aux un une agency agencies agence store stores shop shops mart emporium house
works product products exports export import imports impex marketing sales general bros
brothers sons prop proprietor firm organisation organization org foundation fondation
trust society co-op coop cooperative limited ltd pvt pte llc
""".split()) | {'m s', 'and co', 'and sons', 'and company'}
"""Words that carry little identity information in a business name (legal
forms, generic descriptors, honorifics, articles, country names). Used to
build a strict 'core name' variant; deliberately a bit broad."""

NAME_JUNK_PREFIX_RE_HINT = r'^[\s\*\.\-#@!\|_~,;:\'"`]+'
"""Regex for leading junk characters on a name ('** ACME', '- Acme', '#Acme')."""

NAME_JUNK_SUFFIX_RE_HINT = (
    r'(?:\s*(?:#\s*\d+|\(\s*\d+\s*\)|\[\s*\d+\s*\]|\bid\s*[:#]?\s*\d+))?'
    r'[\s\*\.\-#@!\|_~,;:\'"`]+$'
)
"""Regex for trailing junk: '#12345', '(1)', '[2]', 'id: 77' and any trailing
punctuation/whitespace. Apply with re.sub(HINT, '', name.lower())."""

# ---------------------------------------------------------------------------
# 8. Leetspeak / symbol substitutions
# ---------------------------------------------------------------------------
LEET_MAP = {
    '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '7': 't', '8': 'b', '9': 'g', '2': 'z',
    '6': 'g', '@': 'a', '$': 's', '!': 'i', '|': 'l', '+': 't', '€': 'e', '£': 'l', '¢': 'c',
}
"""Digit/symbol -> letter for undoing leetspeak noise ('4cme' -> 'acme',
'w0rld' -> 'world'). GUARD: apply only inside a token that contains >= 3
letters and <= 2 digits/symbols (never to pure numbers like house numbers,
PINs, ZIPs, or to alphanumeric IDs such as 'b12', 'h no 4/5')."""

# ---------------------------------------------------------------------------
# 9. City renamings / alternate spellings -> canonical modern name
# ---------------------------------------------------------------------------
CITY_ALIASES = {
    # --- India: renamings ---
    'bombay': 'mumbai', 'mumbai': 'mumbai', 'bombai': 'mumbai', 'mumbay': 'mumbai', 'mumbi': 'mumbai',
    'greater mumbai': 'mumbai', 'new bombay': 'navi mumbai', 'navi mumbai': 'navi mumbai',
    'calcutta': 'kolkata', 'kolkatta': 'kolkata', 'kolkota': 'kolkata', 'kolkata': 'kolkata',
    'kolkatta ': 'kolkata', 'calcuta': 'kolkata', 'kalikata': 'kolkata',
    'madras': 'chennai', 'chennai': 'chennai', 'chenai': 'chennai', 'chennnai': 'chennai',
    'bangalore': 'bangalore', 'banglore': 'bangalore', 'bengaluru': 'bangalore', 'bengalooru': 'bangalore',
    'bangaluru': 'bangalore', 'bengalore': 'bangalore', 'bangalor': 'bangalore', 'bengalur': 'bangalore',
    'bangalore urban': 'bangalore', 'bengaluru urban': 'bangalore', 'bangalore rural': 'bangalore',
    'bengaluru rural': 'bangalore',
    'poona': 'pune', 'pune': 'pune', 'puna': 'pune', 'pimpri': 'pimpri chinchwad', 'pcmc': 'pimpri chinchwad',
    'pimpri chinchwad': 'pimpri chinchwad', 'chinchwad': 'pimpri chinchwad',
    'gurgaon': 'gurgaon', 'gurugram': 'gurgaon', 'gurgoan': 'gurgaon', 'gurgaun': 'gurgaon',
    'trivandrum': 'trivandrum', 'thiruvananthapuram': 'trivandrum', 'tiruvananthapuram': 'trivandrum',
    'thiruvanthapuram': 'trivandrum', 'tvm': 'trivandrum',
    'cochin': 'kochi', 'kochi': 'kochi', 'cochin ': 'kochi', 'kochin': 'kochi',
    'baroda': 'vadodara', 'vadodara': 'vadodara', 'vadodra': 'vadodara', 'badodara': 'vadodara',
    'mysore': 'mysore', 'mysuru': 'mysore', 'mysore ': 'mysore',
    'mangalore': 'mangalore', 'mangaluru': 'mangalore', 'manglore': 'mangalore',
    'belgaum': 'belgaum', 'belagavi': 'belgaum', 'belgaon': 'belgaum',
    'hubli': 'hubli', 'hubballi': 'hubli', 'hubli dharwad': 'hubli', 'hubbali': 'hubli',
    'allahabad': 'allahabad', 'prayagraj': 'allahabad', 'prayag raj': 'allahabad', 'alahabad': 'allahabad',
    'cawnpore': 'kanpur', 'kanpur': 'kanpur', 'kanpur nagar': 'kanpur',
    'benares': 'varanasi', 'banaras': 'varanasi', 'varanasi': 'varanasi', 'kashi': 'varanasi',
    'benaras': 'varanasi', 'banares': 'varanasi',
    'vizag': 'visakhapatnam', 'visakhapatnam': 'visakhapatnam', 'vishakhapatnam': 'visakhapatnam',
    'vishakapatnam': 'visakhapatnam', 'visakapatnam': 'visakhapatnam', 'waltair': 'visakhapatnam',
    'vizianagaram ': 'vizianagaram',
    'new delhi': 'delhi', 'delhi': 'delhi', 'dilli': 'delhi', 'dehli': 'delhi', 'newdelhi': 'delhi',
    'south delhi': 'delhi', 'north delhi': 'delhi', 'east delhi': 'delhi', 'west delhi': 'delhi',
    'central delhi': 'delhi', 'south west delhi': 'delhi', 'north west delhi': 'delhi',
    'north east delhi': 'delhi', 'south east delhi': 'delhi', 'shahdara': 'delhi',
    'panaji': 'panaji', 'panjim': 'panaji', 'ponnji': 'panaji', 'panjim goa': 'panaji',
    'shimla': 'shimla', 'simla': 'shimla',
    'ooty': 'ooty', 'udhagamandalam': 'ooty', 'ootacamund': 'ooty', 'udagamandalam': 'ooty',
    'trichy': 'trichy', 'tiruchirappalli': 'trichy', 'tiruchirapalli': 'trichy', 'tiruchi': 'trichy',
    'trichirappalli': 'trichy', 'trichirapalli': 'trichy', 'tiruchchirappalli': 'trichy',
    'tanjore': 'thanjavur', 'thanjavur': 'thanjavur', 'tanjavur': 'thanjavur',
    'tuticorin': 'tuticorin', 'thoothukudi': 'tuticorin', 'thoothukkudi': 'tuticorin', 'tuticorn': 'tuticorin',
    'quilon': 'kollam', 'kollam': 'kollam',
    'calicut': 'kozhikode', 'kozhikode': 'kozhikode', 'kozhikkode': 'kozhikode', 'calicut ': 'kozhikode',
    'cannanore': 'kannur', 'kannur': 'kannur', 'cannanore ': 'kannur',
    'alleppey': 'alappuzha', 'alappuzha': 'alappuzha', 'aleppey': 'alappuzha', 'allepey': 'alappuzha',
    'palghat': 'palakkad', 'palakkad': 'palakkad', 'palakad': 'palakkad',
    'trichur': 'thrissur', 'thrissur': 'thrissur', 'trissur': 'thrissur', 'thrisur': 'thrissur',
    'guwahati': 'guwahati', 'gauhati': 'guwahati', 'guahati': 'guwahati', 'gawahati': 'guwahati',
    'jubbulpore': 'jabalpur', 'jabalpur': 'jabalpur',
    'indore': 'indore', 'indor': 'indore',
    'bhubaneshwar': 'bhubaneswar', 'bhubaneswar': 'bhubaneswar', 'bhubneshwar': 'bhubaneswar',
    'bhubaneshwer': 'bhubaneswar', 'bhubneswar': 'bhubaneswar',
    'cuttack': 'cuttack', 'katak': 'cuttack',
    'pondicherry': 'puducherry', 'puducherry': 'puducherry', 'pondy': 'puducherry', 'pondichery': 'puducherry',
    'vellore': 'vellore', 'salem': 'salem',
    'nasik': 'nashik', 'nashik': 'nashik', 'nashik ': 'nashik',
    'aurangabad': 'aurangabad', 'sambhajinagar': 'aurangabad', 'chhatrapati sambhajinagar': 'aurangabad',
    'chhatrapati sambhaji nagar': 'aurangabad', 'sambhaji nagar': 'aurangabad',
    'osmanabad': 'osmanabad', 'dharashiv': 'osmanabad',
    'ahmednagar': 'ahmednagar', 'ahilyanagar': 'ahmednagar', 'ahmadnagar': 'ahmednagar',
    'ahmedabad': 'ahmedabad', 'amdavad': 'ahmedabad', 'ahmadabad': 'ahmedabad', 'ahemdabad': 'ahmedabad',
    'ahmdabad': 'ahmedabad', 'ahemadabad': 'ahmedabad',
    'hyderabad': 'hyderabad', 'hydrabad': 'hyderabad', 'hyd': 'hyderabad', 'hyderbad': 'hyderabad',
    'secunderabad': 'secunderabad', 'secundrabad': 'secunderabad', 'secbad': 'secunderabad',
    'rajahmundry': 'rajahmundry', 'rajamahendravaram': 'rajahmundry', 'rajamundry': 'rajahmundry',
    'vijayawada': 'vijayawada', 'bezwada': 'vijayawada', 'vijaywada': 'vijayawada',
    'tirupati': 'tirupati', 'tirupathi': 'tirupati',
    'kakinada': 'kakinada', 'cocanada': 'kakinada',
    'coimbatore': 'coimbatore', 'kovai': 'coimbatore', 'coimbatur': 'coimbatore', 'coimbtore': 'coimbatore',
    'tinnevelly': 'tirunelveli', 'tirunelveli': 'tirunelveli', 'nellai': 'tirunelveli',
    'shimoga': 'shimoga', 'shivamogga': 'shimoga', 'tumkur': 'tumkur', 'tumakuru': 'tumkur',
    'bellary': 'bellary', 'ballari': 'bellary', 'gulbarga': 'gulbarga', 'kalaburagi': 'gulbarga',
    'bijapur': 'bijapur', 'vijayapura': 'bijapur', 'hospet': 'hospet', 'hosapete': 'hospet',
    'chikmagalur': 'chikmagalur', 'chikkamagaluru': 'chikmagalur', 'bagalkot': 'bagalkot',
    'bagalkote': 'bagalkot', 'davangere': 'davangere', 'davanagere': 'davangere',
    'faizabad': 'ayodhya', 'ayodhya': 'ayodhya', 'hoshangabad': 'hoshangabad', 'narmadapuram': 'hoshangabad',
    'mughalsarai': 'mughalsarai', 'deen dayal upadhyaya nagar': 'mughalsarai',
    'noida': 'noida', 'greater noida': 'greater noida', 'gautam buddha nagar': 'noida',
    'gautam budh nagar': 'noida', 'ghaziabad': 'ghaziabad', 'gaziabad': 'ghaziabad', 'ghaziabd': 'ghaziabad',
    'faridabad': 'faridabad', 'faridabd': 'faridabad',
    'thane': 'thane', 'thana': 'thane', 'vasai': 'vasai', 'bassein': 'vasai',
    'dehradun': 'dehradun', 'dehra dun': 'dehradun', 'haridwar': 'haridwar', 'hardwar': 'haridwar',
    'lucknow': 'lucknow', 'lakhnau': 'lucknow', 'bareilly': 'bareilly', 'bareli': 'bareilly',
    'jamshedpur': 'jamshedpur', 'tatanagar': 'jamshedpur', 'sagar': 'sagar', 'saugor': 'sagar',
    'jalandhar': 'jalandhar', 'jullundur': 'jalandhar', 'jalandar': 'jalandhar',
    'mohali': 'mohali', 'sas nagar': 'mohali', 'sahibzada ajit singh nagar': 'mohali', 's a s nagar': 'mohali',
    'hisar': 'hisar', 'hissar': 'hisar', 'sonipat': 'sonipat', 'sonepat': 'sonipat',
    'yamunanagar': 'yamunanagar', 'yamuna nagar': 'yamunanagar',
    'bharuch': 'bharuch', 'broach': 'bharuch', 'valsad': 'valsad', 'bulsar': 'valsad',
    'mehsana': 'mehsana', 'mahesana': 'mehsana', 'solapur': 'solapur', 'sholapur': 'solapur',
    'amravati': 'amravati', 'amraoti': 'amravati', 'dhule': 'dhule', 'dhulia': 'dhule',
    'anantapur': 'anantapur', 'anantapuram': 'anantapur', 'kadapa': 'kadapa', 'cuddapah': 'kadapa',
    'eluru': 'eluru', 'ellore': 'eluru', 'machilipatnam': 'machilipatnam', 'masulipatnam': 'machilipatnam',
    'howrah': 'howrah', 'haora': 'howrah', 'burdwan': 'bardhaman', 'bardhaman': 'bardhaman',
    'barddhaman': 'bardhaman', 'midnapore': 'medinipur', 'medinipur': 'medinipur',
    'hooghly': 'hooghly', 'hugli': 'hooghly', 'serampore': 'serampore', 'srirampur': 'serampore',
    'baharampur': 'berhampore', 'berhampore': 'berhampore', 'berhampur': 'berhampur', 'brahmapur': 'berhampur',
    'balasore': 'balasore', 'baleshwar': 'balasore', 'darjeeling': 'darjeeling', 'darjiling': 'darjeeling',
    'margao': 'margao', 'madgaon': 'margao', 'vasco': 'vasco da gama', 'vasco da gama': 'vasco da gama',
    'mapusa': 'mapusa', 'mapuca': 'mapusa', 'tirupur': 'tirupur', 'tiruppur': 'tirupur',
    'kanchipuram': 'kanchipuram', 'conjeevaram': 'kanchipuram', 'kancheepuram': 'kanchipuram',
    'villupuram': 'villupuram', 'viluppuram': 'villupuram', 'nagercoil': 'nagercoil', 'nagarkovil': 'nagercoil',
    'kanyakumari': 'kanyakumari', 'cape comorin': 'kanyakumari', 'kasaragod': 'kasaragod', 'kasargod': 'kasaragod',
    'wayanad': 'wayanad', 'wynad': 'wayanad', 'aluva': 'aluva', 'alwaye': 'aluva',
    'changanassery': 'changanassery', 'changanacherry': 'changanassery', 'tiruvalla': 'tiruvalla',
    'thiruvalla': 'tiruvalla', 'vatakara': 'vadakara', 'vadakara': 'vadakara', 'badagara': 'vadakara',
    'tellicherry': 'thalassery', 'thalassery': 'thalassery', 'srinagar': 'srinagar', 'shrinagar': 'srinagar',
    'jaipur': 'jaipur', 'jaypur': 'jaipur', 'nagpur': 'nagpur', 'nagpore': 'nagpur',
    'kolhapur': 'kolhapur', 'kolhapore': 'kolhapur', 'surat': 'surat', 'rajkot': 'rajkot',
    'bhopal': 'bhopal', 'patna': 'patna', 'ranchi': 'ranchi', 'raipur': 'raipur',
    'chandigarh': 'chandigarh', 'chandigrah': 'chandigarh', 'chandigar': 'chandigarh',
    'ludhiana': 'ludhiana', 'ludhiyana': 'ludhiana', 'amritsar': 'amritsar', 'amritsar ': 'amritsar',
    'ernakulam': 'ernakulam', 'ernakulum': 'ernakulam', 'kottayam': 'kottayam',
    'meerut': 'meerut', 'merrut': 'meerut', 'agra': 'agra', 'jodhpur': 'jodhpur', 'udaipur': 'udaipur',
    'gwalior': 'gwalior', 'jhansi': 'jhansi', 'warangal': 'warangal', 'guntur': 'guntur', 'nellore': 'nellore',
    'kurnool': 'kurnool', 'madurai': 'madurai', 'madura': 'madurai', 'erode': 'erode', 'hosur': 'hosur',
    # --- France: saint/sainte abbreviations & common short forms ---
    'st louis': 'saint louis', 'saint louis': 'saint louis',
    'st herblain': 'saint herblain', 'saint herblain': 'saint herblain',
    'la teste de buch': 'la teste de buch', 'la teste': 'la teste de buch',
    'lege cap ferret': 'lege cap ferret', 'lege': 'lege cap ferret', 'cap ferret': 'lege cap ferret',
    'st etienne': 'saint etienne', 'saint etienne': 'saint etienne',
    'st denis': 'saint denis', 'saint denis': 'saint denis',
    'st nazaire': 'saint nazaire', 'saint nazaire': 'saint nazaire',
    'st malo': 'saint malo', 'saint malo': 'saint malo', 'st brieuc': 'saint brieuc', 'saint brieuc': 'saint brieuc',
    'st quentin': 'saint quentin', 'saint quentin': 'saint quentin',
    'st germain en laye': 'saint germain en laye', 'saint germain en laye': 'saint germain en laye',
    'st maur des fosses': 'saint maur des fosses', 'saint maur des fosses': 'saint maur des fosses',
    'st maur': 'saint maur des fosses',
    'st ouen': 'saint ouen', 'saint ouen': 'saint ouen', 'st ouen sur seine': 'saint ouen',
    'saint ouen sur seine': 'saint ouen', 'st cloud': 'saint cloud', 'saint cloud': 'saint cloud',
    'st priest': 'saint priest', 'saint priest': 'saint priest', 'st raphael': 'saint raphael',
    'saint raphael': 'saint raphael', 'st tropez': 'saint tropez', 'saint tropez': 'saint tropez',
    'st jean de luz': 'saint jean de luz', 'saint jean de luz': 'saint jean de luz',
    'st omer': 'saint omer', 'saint omer': 'saint omer', 'st avold': 'saint avold', 'saint avold': 'saint avold',
    'st die': 'saint die des vosges', 'saint die': 'saint die des vosges', 'st die des vosges': 'saint die des vosges',
    'saint die des vosges': 'saint die des vosges', 'st chamond': 'saint chamond', 'saint chamond': 'saint chamond',
    'st pierre': 'saint pierre', 'saint pierre': 'saint pierre', 'st paul': 'saint paul', 'saint paul': 'saint paul',
    'st andre': 'saint andre', 'saint andre': 'saint andre', 'st benoit': 'saint benoit', 'saint benoit': 'saint benoit',
    'st joseph': 'saint joseph', 'saint joseph': 'saint joseph', 'st leu': 'saint leu', 'saint leu': 'saint leu',
    'st gilles': 'saint gilles', 'saint gilles': 'saint gilles',
    'st laurent du var': 'saint laurent du var', 'saint laurent du var': 'saint laurent du var',
    'st martin d heres': 'saint martin d heres', 'saint martin d heres': 'saint martin d heres',
    'st martin dheres': 'saint martin d heres', 'saint martin dheres': 'saint martin d heres',
    'st egreve': 'saint egreve', 'saint egreve': 'saint egreve',
    'st jean de braye': 'saint jean de braye', 'saint jean de braye': 'saint jean de braye',
    'st sebastien sur loire': 'saint sebastien sur loire', 'saint sebastien sur loire': 'saint sebastien sur loire',
    'st medard en jalles': 'saint medard en jalles', 'saint medard en jalles': 'saint medard en jalles',
    'st cyr sur loire': 'saint cyr sur loire', 'saint cyr sur loire': 'saint cyr sur loire',
    'st pierre des corps': 'saint pierre des corps', 'saint pierre des corps': 'saint pierre des corps',
    'st avertin': 'saint avertin', 'saint avertin': 'saint avertin', 'st lo': 'saint lo', 'saint lo': 'saint lo',
    'st dizier': 'saint dizier', 'saint dizier': 'saint dizier', 'st mande': 'saint mande', 'saint mande': 'saint mande',
    'st michel sur orge': 'saint michel sur orge', 'saint michel sur orge': 'saint michel sur orge',
    'st fons': 'saint fons', 'saint fons': 'saint fons', 'st genis laval': 'saint genis laval',
    'saint genis laval': 'saint genis laval', 'st jean de vedas': 'saint jean de vedas',
    'saint jean de vedas': 'saint jean de vedas', 'st orens de gameville': 'saint orens de gameville',
    'saint orens de gameville': 'saint orens de gameville', 'st orens': 'saint orens de gameville',
    'st gaudens': 'saint gaudens', 'saint gaudens': 'saint gaudens', 'st girons': 'saint girons',
    'saint girons': 'saint girons', 'st jean': 'saint jean', 'saint jean': 'saint jean',
    'st martin': 'saint martin', 'saint martin': 'saint martin', 'st georges': 'saint georges',
    'saint georges': 'saint georges', 'st julien': 'saint julien', 'saint julien': 'saint julien',
    'st vincent': 'saint vincent', 'saint vincent': 'saint vincent', 'st laurent': 'saint laurent',
    'saint laurent': 'saint laurent', 'st remy': 'saint remy', 'saint remy': 'saint remy',
    'st germain': 'saint germain', 'saint germain': 'saint germain', 'st marcel': 'saint marcel',
    'saint marcel': 'saint marcel', 'st hilaire': 'saint hilaire', 'saint hilaire': 'saint hilaire',
    'ste genevieve des bois': 'sainte genevieve des bois', 'sainte genevieve des bois': 'sainte genevieve des bois',
    'ste genevieve': 'sainte genevieve des bois', 'ste maxime': 'sainte maxime', 'sainte maxime': 'sainte maxime',
    'ste foy les lyon': 'sainte foy les lyon', 'sainte foy les lyon': 'sainte foy les lyon',
    'ste marie': 'sainte marie', 'sainte marie': 'sainte marie', 'ste anne': 'sainte anne', 'sainte anne': 'sainte anne',
    'ste luce sur loire': 'sainte luce sur loire', 'sainte luce sur loire': 'sainte luce sur loire',
    'ste savine': 'sainte savine', 'sainte savine': 'sainte savine', 'ste rose': 'sainte rose',
    'sainte rose': 'sainte rose', 'ste clotilde': 'sainte clotilde', 'sainte clotilde': 'sainte clotilde',
    'ste suzanne': 'sainte suzanne', 'sainte suzanne': 'sainte suzanne', 'ste marthe': 'sainte marthe',
    'sainte marthe': 'sainte marthe', 'ste helene': 'sainte helene', 'sainte helene': 'sainte helene',
    'ste croix': 'sainte croix', 'sainte croix': 'sainte croix',
    'aix': 'aix en provence', 'aix en provence': 'aix en provence', 'aix en pce': 'aix en provence',
    'aix en pvce': 'aix en provence', 'villeneuve d ascq': 'villeneuve d ascq', 'villeneuve dascq': 'villeneuve d ascq',
    'marseilles': 'marseille', 'marseille': 'marseille', 'lyons': 'lyon', 'lyon': 'lyon',
    'strasburg': 'strasbourg', 'strasbourg': 'strasbourg', 'le havre': 'le havre', 'havre': 'le havre',
    'levallois': 'levallois perret', 'levallois perret': 'levallois perret',
    'issy': 'issy les moulineaux', 'issy les moulineaux': 'issy les moulineaux', 'issy les mx': 'issy les moulineaux',
    'neuilly': 'neuilly sur seine', 'neuilly sur seine': 'neuilly sur seine', 'neuilly s seine': 'neuilly sur seine',
    'boulogne billancourt': 'boulogne billancourt', 'boulogne bilt': 'boulogne billancourt',
    'boulogne billt': 'boulogne billancourt', 'boulogne s mer': 'boulogne sur mer', 'boulogne sur mer': 'boulogne sur mer',
    'clermont fd': 'clermont ferrand', 'clermont ferrand': 'clermont ferrand', 'clermont fer': 'clermont ferrand',
    'chalon s saone': 'chalon sur saone', 'chalon sur saone': 'chalon sur saone',
    'chalons en champagne': 'chalons en champagne', 'chalons sur marne': 'chalons en champagne',
    'chalons': 'chalons en champagne', 'la roche s yon': 'la roche sur yon', 'la roche sur yon': 'la roche sur yon',
    'villefranche s saone': 'villefranche sur saone', 'villefranche sur saone': 'villefranche sur saone',
    'villefranche s mer': 'villefranche sur mer', 'villefranche sur mer': 'villefranche sur mer',
    'evry': 'evry courcouronnes', 'evry courcouronnes': 'evry courcouronnes', 'courcouronnes': 'evry courcouronnes',
    'cergy': 'cergy', 'cergy pontoise': 'cergy', 'montigny le bx': 'montigny le bretonneux',
    'montigny le bretonneux': 'montigny le bretonneux', 'st quentin en yvelines': 'saint quentin en yvelines',
    'saint quentin en yvelines': 'saint quentin en yvelines', 'sqy': 'saint quentin en yvelines',
    'marne la vallee': 'marne la vallee', 'la defense': 'courbevoie', 'paris la defense': 'courbevoie',
    'courbevoie': 'courbevoie', 'puteaux': 'puteaux', 'nanterre': 'nanterre',
    'vitry s seine': 'vitry sur seine', 'vitry sur seine': 'vitry sur seine', 'ivry s seine': 'ivry sur seine',
    'ivry sur seine': 'ivry sur seine', 'choisy le roi': 'choisy le roi', 'asnieres': 'asnieres sur seine',
    'asnieres sur seine': 'asnieres sur seine', 'asnieres s seine': 'asnieres sur seine',
    'epinay s seine': 'epinay sur seine', 'epinay sur seine': 'epinay sur seine', 'epinay': 'epinay sur seine',
    'conflans ste honorine': 'conflans sainte honorine', 'conflans sainte honorine': 'conflans sainte honorine',
    'conflans': 'conflans sainte honorine', 'rueil': 'rueil malmaison', 'rueil malmaison': 'rueil malmaison',
    'fontenay s bois': 'fontenay sous bois', 'fontenay sous bois': 'fontenay sous bois',
    'noisy le gd': 'noisy le grand', 'noisy le grand': 'noisy le grand', 'champigny s marne': 'champigny sur marne',
    'champigny sur marne': 'champigny sur marne', 'champigny': 'champigny sur marne',
    'nogent s marne': 'nogent sur marne', 'nogent sur marne': 'nogent sur marne',
    'bry s marne': 'bry sur marne', 'bry sur marne': 'bry sur marne', 'joinville le pont': 'joinville le pont',
    'la seyne s mer': 'la seyne sur mer', 'la seyne sur mer': 'la seyne sur mer', 'la seyne': 'la seyne sur mer',
    'six fours': 'six fours les plages', 'six fours les plages': 'six fours les plages',
    'cagnes s mer': 'cagnes sur mer', 'cagnes sur mer': 'cagnes sur mer', 'cagnes': 'cagnes sur mer',
    'mandelieu': 'mandelieu la napoule', 'mandelieu la napoule': 'mandelieu la napoule',
    'porto vecchio': 'porto vecchio', 'ajaccio': 'ajaccio', 'bastia': 'bastia', 'nimes': 'nimes',
    'st jean de monts': 'saint jean de monts', 'saint jean de monts': 'saint jean de monts',
    'les sables d olonne': 'les sables d olonne', 'les sables dolonne': 'les sables d olonne',
    'sables d olonne': 'les sables d olonne', 'la baule': 'la baule escoublac', 'la baule escoublac': 'la baule escoublac',
    'le mans': 'le mans', 'le puy': 'le puy en velay', 'le puy en velay': 'le puy en velay',
    'bourg en bresse': 'bourg en bresse', 'bourg': 'bourg en bresse', 'st fargeau ponthierry': 'saint fargeau ponthierry',
    'saint fargeau ponthierry': 'saint fargeau ponthierry', 'tremblay en france': 'tremblay en france',
    'tremblay': 'tremblay en france', 'roissy en france': 'roissy en france', 'roissy': 'roissy en france',
    'roissy cdg': 'roissy en france', 'roissy charles de gaulle': 'roissy en france',
    # --- US: abbreviations / boroughs / renamings ---
    'new york city': 'new york', 'nyc': 'new york', 'ny city': 'new york', 'new york': 'new york',
    'manhattan': 'new york', 'brooklyn': 'brooklyn', 'bklyn': 'brooklyn', 'the bronx': 'bronx', 'bronx': 'bronx',
    'staten island': 'staten island', 'staten is': 'staten island', 'long island city': 'long island city',
    'lic': 'long island city', 'los angeles': 'los angeles', 'los angelos': 'los angeles', 'los angles': 'los angeles',
    'san francisco': 'san francisco', 'san fransisco': 'san francisco', 'san fran': 'san francisco',
    'philadelphia': 'philadelphia', 'philly': 'philadelphia', 'phila': 'philadelphia',
    'saint petersburg': 'saint petersburg', 'st petersburg': 'saint petersburg', 'st pete': 'saint petersburg',
    'saint augustine': 'saint augustine', 'st augustine': 'saint augustine', 'saint charles': 'saint charles',
    'st charles': 'saint charles', 'saint george': 'saint george', 'st george': 'saint george',
    'saint peters': 'saint peters', 'st peters': 'saint peters', 'saint clair shores': 'saint clair shores',
    'st clair shores': 'saint clair shores', 'saint louis park': 'saint louis park', 'st louis park': 'saint louis park',
    'saint johns': 'saint johns', 'st johns': 'saint johns', 'saint albans': 'saint albans', 'st albans': 'saint albans',
    'saint simons island': 'saint simons island', 'st simons island': 'saint simons island',
    'saint helens': 'saint helens', 'st helens': 'saint helens', 'saint marys': 'saint marys', 'st marys': 'saint marys',
    'saint michael': 'saint michael', 'st michael': 'saint michael', 'saint francis': 'saint francis',
    'st francis': 'saint francis', 'saint anthony': 'saint anthony', 'st anthony': 'saint anthony',
    'saint james': 'saint james', 'st james': 'saint james', 'saint matthews': 'saint matthews',
    'st matthews': 'saint matthews', 'saint rose': 'saint rose', 'st rose': 'saint rose',
    'sainte genevieve': 'sainte genevieve', 'ste genevieve': 'sainte genevieve',
    'fort worth': 'fort worth', 'ft worth': 'fort worth', 'fort lauderdale': 'fort lauderdale',
    'ft lauderdale': 'fort lauderdale', 'fort myers': 'fort myers', 'ft myers': 'fort myers',
    'fort wayne': 'fort wayne', 'ft wayne': 'fort wayne', 'fort collins': 'fort collins', 'ft collins': 'fort collins',
    'fort smith': 'fort smith', 'ft smith': 'fort smith', 'fort pierce': 'fort pierce', 'ft pierce': 'fort pierce',
    'fort walton beach': 'fort walton beach', 'ft walton beach': 'fort walton beach', 'fort mill': 'fort mill',
    'ft mill': 'fort mill', 'fort lee': 'fort lee', 'ft lee': 'fort lee', 'fort washington': 'fort washington',
    'ft washington': 'fort washington', 'fort dodge': 'fort dodge', 'ft dodge': 'fort dodge',
    'fort payne': 'fort payne', 'ft payne': 'fort payne', 'fort oglethorpe': 'fort oglethorpe',
    'ft oglethorpe': 'fort oglethorpe', 'fort atkinson': 'fort atkinson', 'ft atkinson': 'fort atkinson',
    'fort morgan': 'fort morgan', 'ft morgan': 'fort morgan', 'fort lupton': 'fort lupton', 'ft lupton': 'fort lupton',
    'fort madison': 'fort madison', 'ft madison': 'fort madison', 'fort bragg': 'fort bragg', 'ft bragg': 'fort bragg',
    'fort scott': 'fort scott', 'ft scott': 'fort scott', 'fort thomas': 'fort thomas', 'ft thomas': 'fort thomas',
    'fort mitchell': 'fort mitchell', 'ft mitchell': 'fort mitchell', 'fort stockton': 'fort stockton',
    'ft stockton': 'fort stockton', 'fort gratiot': 'fort gratiot', 'ft gratiot': 'fort gratiot',
    'fort hood': 'fort cavazos', 'ft hood': 'fort cavazos', 'fort cavazos': 'fort cavazos',
    'mount vernon': 'mount vernon', 'mt vernon': 'mount vernon', 'mount pleasant': 'mount pleasant',
    'mt pleasant': 'mount pleasant', 'mount laurel': 'mount laurel', 'mt laurel': 'mount laurel',
    'mount prospect': 'mount prospect', 'mt prospect': 'mount prospect', 'mount juliet': 'mount juliet',
    'mt juliet': 'mount juliet', 'mount airy': 'mount airy', 'mt airy': 'mount airy', 'mount kisco': 'mount kisco',
    'mt kisco': 'mount kisco', 'mount holly': 'mount holly', 'mt holly': 'mount holly', 'mount dora': 'mount dora',
    'mt dora': 'mount dora', 'mount clemens': 'mount clemens', 'mt clemens': 'mount clemens',
    'mount olive': 'mount olive', 'mt olive': 'mount olive', 'mount sinai': 'mount sinai', 'mt sinai': 'mount sinai',
    'mount pocono': 'mount pocono', 'mt pocono': 'mount pocono', 'mount joy': 'mount joy', 'mt joy': 'mount joy',
    'mount horeb': 'mount horeb', 'mt horeb': 'mount horeb', 'mount washington': 'mount washington',
    'mt washington': 'mount washington', 'mount carmel': 'mount carmel', 'mt carmel': 'mount carmel',
    'mount sterling': 'mount sterling', 'mt sterling': 'mount sterling', 'mount vernon ny': 'mount vernon',
    'port saint lucie': 'port saint lucie', 'port st lucie': 'port saint lucie',
    'las vegas': 'las vegas', 'vegas': 'las vegas', 'n las vegas': 'north las vegas', 'north las vegas': 'north las vegas',
    'washington dc': 'washington', 'washington d c': 'washington', 'dc': 'washington', 'd c': 'washington',
    'n hollywood': 'north hollywood', 'north hollywood': 'north hollywood', 'w hollywood': 'west hollywood',
    'west hollywood': 'west hollywood', 'e lansing': 'east lansing', 'east lansing': 'east lansing',
    'w palm beach': 'west palm beach', 'west palm beach': 'west palm beach', 'wpb': 'west palm beach',
    'n miami': 'north miami', 'north miami': 'north miami', 'n miami beach': 'north miami beach',
    'north miami beach': 'north miami beach', 's san francisco': 'south san francisco',
    'south san francisco': 'south san francisco', 'w des moines': 'west des moines',
    'west des moines': 'west des moines', 'n charleston': 'north charleston', 'north charleston': 'north charleston',
    'n little rock': 'north little rock', 'north little rock': 'north little rock', 'e hartford': 'east hartford',
    'east hartford': 'east hartford', 'w hartford': 'west hartford', 'west hartford': 'west hartford',
    's bend': 'south bend', 'south bend': 'south bend', 'n richland hills': 'north richland hills',
    'north richland hills': 'north richland hills', 'e brunswick': 'east brunswick', 'east brunswick': 'east brunswick',
    'n bergen': 'north bergen', 'north bergen': 'north bergen', 'w new york': 'west new york',
    'west new york': 'west new york', 'e orange': 'east orange', 'east orange': 'east orange',
    'w orange': 'west orange', 'west orange': 'west orange', 's orange': 'south orange', 'south orange': 'south orange',
    'e rutherford': 'east rutherford', 'east rutherford': 'east rutherford', 'n arlington': 'north arlington',
    'north arlington': 'north arlington', 'w chester': 'west chester', 'west chester': 'west chester',
    'e chicago': 'east chicago', 'east chicago': 'east chicago', 'e st louis': 'east saint louis',
    'east st louis': 'east saint louis', 'east saint louis': 'east saint louis', 'e saint louis': 'east saint louis',
    'w covina': 'west covina', 'west covina': 'west covina', 'e los angeles': 'east los angeles',
    'east los angeles': 'east los angeles', 'e palo alto': 'east palo alto', 'east palo alto': 'east palo alto',
    's lake tahoe': 'south lake tahoe', 'south lake tahoe': 'south lake tahoe', 's jordan': 'south jordan',
    'south jordan': 'south jordan', 'w jordan': 'west jordan', 'west jordan': 'west jordan',
    'w valley city': 'west valley city', 'west valley city': 'west valley city', 'n ogden': 'north ogden',
    'north ogden': 'north ogden', 's ogden': 'south ogden', 'south ogden': 'south ogden',
    's salt lake': 'south salt lake', 'south salt lake': 'south salt lake', 'salt lake': 'salt lake city',
    'salt lake city': 'salt lake city', 'slc': 'salt lake city', 'oklahoma city': 'oklahoma city', 'okc': 'oklahoma city',
    'kansas city': 'kansas city', 'kc': 'kansas city', 'santa fe': 'santa fe', 'sante fe': 'santa fe',
    'cincinatti': 'cincinnati', 'cincinnati': 'cincinnati', 'cincinnatti': 'cincinnati', 'cinci': 'cincinnati',
    'albuquerque': 'albuquerque', 'albequerque': 'albuquerque', 'albuqerque': 'albuquerque', 'abq': 'albuquerque',
    'tucson': 'tucson', 'tuscon': 'tucson', 'phoenix': 'phoenix', 'pheonix': 'phoenix', 'phx': 'phoenix',
    'san diego': 'san diego', 'san deigo': 'san diego', 'sacramento': 'sacramento', 'sacremento': 'sacramento',
    'saint louis mo': 'saint louis', 'st louis mo': 'saint louis', 'st lous': 'saint louis',
    'saint paul': 'saint paul', 'st paul mn': 'saint paul', 'mpls': 'minneapolis', 'minneapolis': 'minneapolis',
    'pgh': 'pittsburgh', 'pittsburgh': 'pittsburgh', 'pittsburgh pa': 'pittsburgh', 'chi': 'chicago', 'chicago': 'chicago',
    'atl': 'atlanta', 'atlanta': 'atlanta', 'nola': 'new orleans', 'new orleans': 'new orleans',
    'n orleans': 'new orleans', 'nyc ny': 'new york', 'la ca': 'los angeles', 'sf ca': 'san francisco',
    'hollywood': 'hollywood', 'hlywd': 'hollywood', 'jax': 'jacksonville', 'jacksonville': 'jacksonville',
    'indy': 'indianapolis', 'indianapolis': 'indianapolis', 'colo springs': 'colorado springs',
    'colorado springs': 'colorado springs', 'colo spgs': 'colorado springs', 'co springs': 'colorado springs',
    'va beach': 'virginia beach', 'virginia beach': 'virginia beach', 'vb': 'virginia beach',
    'ft lauderdale fl': 'fort lauderdale', 'winston salem': 'winston salem',
    'wilkes barre': 'wilkes barre', 'nyack': 'nyack', 'w nyack': 'west nyack',
    'west nyack': 'west nyack', 'n babylon': 'north babylon', 'north babylon': 'north babylon',
    'w babylon': 'west babylon', 'west babylon': 'west babylon', 'e meadow': 'east meadow', 'east meadow': 'east meadow',
    'e northport': 'east northport', 'east northport': 'east northport', 'n tonawanda': 'north tonawanda',
    'north tonawanda': 'north tonawanda', 'w seneca': 'west seneca', 'west seneca': 'west seneca',
    'e syracuse': 'east syracuse', 'east syracuse': 'east syracuse', 'n syracuse': 'north syracuse',
    'north syracuse': 'north syracuse', 'e greenbush': 'east greenbush', 'east greenbush': 'east greenbush',
    'e providence': 'east providence', 'east providence': 'east providence', 'n providence': 'north providence',
    'north providence': 'north providence', 'w warwick': 'west warwick', 'west warwick': 'west warwick',
    'n kingstown': 'north kingstown', 'north kingstown': 'north kingstown', 's kingstown': 'south kingstown',
    'south kingstown': 'south kingstown', 'e haven': 'east haven', 'east haven': 'east haven',
    'w haven': 'west haven', 'west haven': 'west haven', 'n haven': 'north haven', 'north haven': 'north haven',
    'e windsor': 'east windsor', 'east windsor': 'east windsor', 's windsor': 'south windsor',
    'south windsor': 'south windsor', 'w springfield': 'west springfield', 'west springfield': 'west springfield',
    'e longmeadow': 'east longmeadow', 'east longmeadow': 'east longmeadow', 'n attleboro': 'north attleboro',
    'north attleboro': 'north attleboro', 'n andover': 'north andover', 'north andover': 'north andover',
    'n reading': 'north reading', 'north reading': 'north reading', 'w roxbury': 'west roxbury',
    'west roxbury': 'west roxbury', 'e boston': 'east boston', 'east boston': 'east boston', 's boston': 'south boston',
    'south boston': 'south boston', 'jamaica plain': 'jamaica plain', 'jp': 'jamaica plain',
}
"""City renamings and common alternate spellings / abbreviations -> canonical
name (lowercase ASCII, spaces not hyphens). Indian canonical = the form most
common in the dataset (e.g. 'bangalore', 'gurgaon', 'trivandrum', 'mysore',
'allahabad'); French: 'st'/'ste' expanded to 'saint'/'sainte', 's' -> 'sur';
US: 'ft'/'mt'/'st' expanded, single-letter directionals expanded. Twin
cities that are distinct places (secunderabad/hyderabad, ernakulam/kochi,
boulogne-sur-mer/boulogne-billancourt) are NOT merged."""
CITY_ALIASES = {k.strip(): v for k, v in CITY_ALIASES.items()}

# ---------------------------------------------------------------------------
# 10. Blocking stopwords (exclude from blocking keys only, NOT from features)
# ---------------------------------------------------------------------------
STOPWORDS_BLOCKING = frozenset(set("""
the a an and or of in on at to for by with from as is are be it its this that these those
near behind beside opposite next off no number via through over under between above below
out up down into onto than then there here where when which who whom whose what how all any
each every some most other such only own same so too very can will just not but if also both
either neither nor yet per vs etc
de la le les du des et en un une au aux d l sur sous pour par chez dans avec sans vers entre
ou que qui ce cet cette ces se sa son ses leur leurs y il elle ils elles ne pas plus moins
ka ke ki ko se aur evam wale wala wali paas pass samne peeche mein me par tak va ya hai ji
sahab niche upar
st rd ave dr ln blvd ct cir pl pkwy hwy apt ste bldg fl unit rm n s e w ne nw se sw r av bd
all imp che rte res bat appt za zi zac zae h flat plot shop door sec ph blk pkt opp nr po
dist teh tal vill vpo nagar colony col marg gali crs main chowk cmplx soc indl est area extn
layout stg co no near behind beside opposite floor ground first second third road street
cross city town village india france usa us cedex bis ter bp cs ld lot n a na null none nil
nan unknown xxx zz test gf lgf ugf twr ward survey sy gat khasra hno sno pno dno fno plt flt
mkt bazar stn rly ps pin zip box pmb trl way sq plz hts mt pt twp ctr xing jct tpke fwy expy
aly ext encl vihar pura puram peth wadi mandal mun panchayat sub div sector phase block
pocket house lane avenue drive court circle place boulevard highway parkway terrace trail
square plaza building suite apartment room office ofc department dept basement bsmt lobby
lower upper rear front side space spc stop trailer trlr lieu dit allee impasse chemin route
residence batiment appartement zone artisanale industrielle esplanade sentier villa hameau
lotissement cite montee promenade rond point rpt saint sainte general marechal docteur
professeur commandant president capitaine colonel lieutenant monsieur madame notre dame nd
quai qu cours faubourg fg fbg passage pas rue centre center commercial cc immeuble imm
escalier esc etage etg porte pte entree ent domaine dom grand grande gd gde vieux vieille
ancien ancienne nationale departementale communale rn cd vc parc pont port prt gal mal pr
cdt pdt cne lt m mme mlle boite postale tehsil taluka taluk tq district distt dt post
adjacent adj inside infront landmark new old north south east west so do wo ho prop
proprietor proprietorship gram gaon mdl municipal palika nigam sect scheme wd godown shed
stall booth sco scf cts khata khewat flr hall halls
inc corp llc ltd pvt sarl sas sa eurl sasu sci ei private limited company corporation
incorporated llp plc pc pllc ets cie societe snc scop scp scm selarl selas sel gie earl eirl
sem sca scs sccv sarlu opc lp pa bros brothers sons fils freres associes associates partners
group groupe holdings holding enterprises enterprise entreprise services service solutions
industries international intl global trading traders m/s ms mr mrs smt shri sri shree sree
messrs
""".split()) | {'m s', 'n a', 'p o', 'c o'})
"""Tokens excluded when building BLOCKING keys only (never removed from the
strings that similarity features are computed on): English / French /
Hindi-roman function words, generic address descriptors (canonical
ADDR_ABBREV outputs and their long forms), placeholder values, legal-form
tokens and generic business-name fillers."""

# ---------------------------------------------------------------------------
# 11. Indic script detection + Indic-script legal abbreviations
# ---------------------------------------------------------------------------
INDIC_SCRIPT_RANGES = [
    (0x0900, 0x097F, 'devanagari'),
    (0x0980, 0x09FF, 'bengali'),
    (0x0A00, 0x0A7F, 'gurmukhi'),
    (0x0A80, 0x0AFF, 'gujarati'),
    (0x0B00, 0x0B7F, 'oriya'),
    (0x0B80, 0x0BFF, 'tamil'),
    (0x0C00, 0x0C7F, 'telugu'),
    (0x0C80, 0x0CFF, 'kannada'),
    (0x0D00, 0x0D7F, 'malayalam'),
    (0x0D80, 0x0DFF, 'sinhala'),
    (0xA8E0, 0xA8FF, 'devanagari'),   # Devanagari Extended
    (0x1CD0, 0x1CFF, 'devanagari'),   # Vedic Extensions
]
"""(start, end, script) Unicode code-point ranges (inclusive) of the Indic
scripts that occur in Indian business names / addresses."""


def detect_script(s):
    """Return 'indic' if any character of `s` falls in INDIC_SCRIPT_RANGES,
    'latin_accented' if any other non-ASCII character is present (French
    accents, curly quotes, ...), else 'latin'."""
    non_ascii = False
    for ch in s:
        o = ord(ch)
        if o < 128:
            continue
        non_ascii = True
        for lo, hi, _name in INDIC_SCRIPT_RANGES:
            if lo <= o <= hi:
                return 'indic'
    return 'latin_accented' if non_ascii else 'latin'


def detect_indic_script(s):
    """Return the name of the first Indic script found in `s` ('devanagari',
    'bengali', ...) or '' if none."""
    for ch in s:
        o = ord(ch)
        if o < 128:
            continue
        for lo, hi, name in INDIC_SCRIPT_RANGES:
            if lo <= o <= hi:
                return name
    return ''


INDIC_LEGAL_ABBREV = {
    # Devanagari (Hindi / Marathi)
    'प्रा': 'pvt', 'प्रा.': 'pvt', 'लि': 'ltd', 'लि.': 'ltd', 'प्रा लि': 'pvt ltd', 'प्रा. लि.': 'pvt ltd',
    'प्रा.लि.': 'pvt ltd', 'प्राइवेट': 'private', 'प्रायवेट': 'private', 'प्राईवेट': 'private',
    'लिमिटेड': 'limited', 'लिमीटेड': 'limited', 'लिमिटेड.': 'limited',
    'प्राइवेट लिमिटेड': 'private limited', 'प्रायवेट लिमिटेड': 'private limited',
    'एंड': 'and', 'एण्ड': 'and', 'अँड': 'and', 'ऍण्ड': 'and', 'आणि': 'and', 'और': 'and', 'एवं': 'and',
    'कंपनी': 'company', 'कम्पनी': 'company', 'कं': 'co', 'कं.': 'co', 'इंडिया': 'india',
    'एंटरप्राइजेज': 'enterprises', 'एन्टरप्राइजेस': 'enterprises', 'एंटरप्राइजेस': 'enterprises',
    'एंटरप्रायजेस': 'enterprises', 'सर्विसेज': 'services', 'सर्विसेस': 'services',
    'सोल्युशन्स': 'solutions', 'सॉल्यूशंस': 'solutions', 'इंडस्ट्रीज': 'industries',
    'इंडस्ट्रीज़': 'industries', 'ट्रेडर्स': 'traders', 'ट्रेडिंग': 'trading', 'एलएलपी': 'llp',
    'इंक': 'inc', 'कॉर्पोरेशन': 'corporation', 'कारपोरेशन': 'corporation', 'ब्रदर्स': 'brothers',
    'एंड संस': 'and sons', 'संस': 'sons', 'मेसर्स': 'messrs', 'मे.': 'messrs', 'श्री': 'shri',
    'श्रीमती': 'smt', 'डॉ': 'dr', 'डॉ.': 'dr', 'स्टोर': 'store', 'स्टोर्स': 'stores', 'सेंटर': 'center',
    'ग्रुप': 'group', 'इंटरनेशनल': 'international', 'टेक्नोलॉजीज': 'technologies',
    'सोसायटी': 'society', 'सहकारी': 'cooperative', 'मर्यादित': 'limited', 'प्रायव्हेट': 'private',
    'प्रायव्हेट लिमिटेड': 'private limited', 'उद्योग': 'industries', 'व्यापार': 'trading',
    # Gujarati
    'પ્રા': 'pvt', 'પ્રા.': 'pvt', 'લિ': 'ltd', 'લિ.': 'ltd', 'પ્રા લિ': 'pvt ltd', 'પ્રા. લિ.': 'pvt ltd',
    'પ્રાઇવેટ': 'private', 'પ્રાઈવેટ': 'private', 'લિમિટેડ': 'limited', 'એન્ડ': 'and', 'અને': 'and',
    'કંપની': 'company', 'શ્રી': 'shri', 'એન્ટરપ્રાઇઝ': 'enterprise', 'ઇન્ડસ્ટ્રીઝ': 'industries',
    'ટ્રેડર્સ': 'traders', 'સ્ટોર': 'store', 'સ્ટોર્સ': 'stores',
    # Bengali
    'প্রাইভেট': 'private', 'লিমিটেড': 'limited', 'প্রা': 'pvt', 'লি': 'ltd', 'প্রাঃ': 'pvt', 'লিঃ': 'ltd',
    'প্রাঃ লিঃ': 'pvt ltd', 'অ্যান্ড': 'and', 'এন্ড': 'and', 'এ্যান্ড': 'and', 'ও': 'and',
    'কোম্পানি': 'company', 'কোং': 'co', 'শ্রী': 'shri', 'এন্টারপ্রাইজ': 'enterprise',
    'ইন্ডাস্ট্রিজ': 'industries', 'ট্রেডার্স': 'traders', 'স্টোর': 'store', 'স্টোরস': 'stores',
    # Telugu
    'ప్రైవేట్': 'private', 'లిమిటెడ్': 'limited', 'ప్రై': 'pvt', 'లి': 'ltd', 'ప్రై లి': 'pvt ltd',
    'అండ్': 'and', 'కంపెనీ': 'company', 'శ్రీ': 'shri', 'ఎంటర్‌ప్రైజెస్': 'enterprises',
    'ఎంటర్ప్రైజెస్': 'enterprises', 'ఇండస్ట్రీస్': 'industries', 'ట్రేడర్స్': 'traders', 'స్టోర్స్': 'stores',
    # Tamil
    'பிரைவேட்': 'private', 'லிமிடெட்': 'limited', 'பிரை': 'pvt', 'லிட்': 'ltd', 'பி': 'pvt',
    'அண்ட்': 'and', 'கம்பெனி': 'company', 'ஸ்ரீ': 'shri', 'எண்டர்பிரைசஸ்': 'enterprises',
    'இண்டஸ்ட்ரீஸ்': 'industries', 'டிரேடர்ஸ்': 'traders', 'ஸ்டோர்ஸ்': 'stores',
    # Kannada
    'ಪ್ರೈವೇಟ್': 'private', 'ಲಿಮಿಟೆಡ್': 'limited', 'ಪ್ರೈ': 'pvt', 'ಲಿ': 'ltd', 'ಪ್ರೈ ಲಿ': 'pvt ltd',
    'ಅಂಡ್': 'and', 'ಕಂಪನಿ': 'company', 'ಶ್ರೀ': 'shri', 'ಎಂಟರ್‌ಪ್ರೈಸಸ್': 'enterprises',
    'ಇಂಡಸ್ಟ್ರೀಸ್': 'industries', 'ಟ್ರೇಡರ್ಸ್': 'traders', 'ಸ್ಟೋರ್ಸ್': 'stores',
    # Malayalam
    'പ്രൈവറ്റ്': 'private', 'ലിമിറ്റഡ്': 'limited', 'പ്രൈ': 'pvt', 'ലി': 'ltd', 'ആൻഡ്': 'and', 'ആന്റ്': 'and',
    'കമ്പനി': 'company', 'ശ്രീ': 'shri', 'എന്റർപ്രൈസസ്': 'enterprises', 'ഇൻഡസ്ട്രീസ്': 'industries',
    'ട്രേഡേഴ്സ്': 'traders', 'സ്റ്റോഴ്സ്': 'stores',
    # Gurmukhi (Punjabi)
    'ਪ੍ਰਾਈਵੇਟ': 'private', 'ਲਿਮਟਿਡ': 'limited', 'ਲਿਮਿਟੇਡ': 'limited', 'ਪ੍ਰਾ': 'pvt', 'ਲਿ': 'ltd',
    'ਪ੍ਰਾ ਲਿ': 'pvt ltd', 'ਐਂਡ': 'and', 'ਅਤੇ': 'and', 'ਕੰਪਨੀ': 'company', 'ਸ਼੍ਰੀ': 'shri',
    'ਐਂਟਰਪ੍ਰਾਈਜ਼ਿਜ਼': 'enterprises', 'ਇੰਡਸਟਰੀਜ਼': 'industries', 'ਟਰੇਡਰਜ਼': 'traders', 'ਸਟੋਰ': 'store',
    # Odia
    'ପ୍ରାଇଭେଟ': 'private', 'ଲିମିଟେଡ': 'limited', 'ପ୍ରା': 'pvt', 'ଲି': 'ltd', 'ଆଣ୍ଡ': 'and', 'ଓ': 'and',
    'କମ୍ପାନୀ': 'company', 'ଶ୍ରୀ': 'shri',
}
"""Indic-script tokens for legal forms / generic business words -> romanized
form. Keys are the native spellings (with and without trailing '.'); their
unidecode() transliterations are added at module load. Apply as whole-token
replacement. Note: 'ஓ'/'ও' (Bengali/Odia 'and') are single letters and
should only be replaced as standalone tokens."""


# ---------------------------------------------------------------------------
# Module-load: register unidecoded variants of every non-ASCII key
# ---------------------------------------------------------------------------
def _add_unidecoded_variants(d):
    """Add `unidecode(key).lower()` (plus punctuation-stripped forms) for every
    non-ASCII key, so lookups work on transliterated input too."""
    for k in list(d.keys()):
        if not k.isascii():
            u = ' '.join(unidecode(k).lower().split())
            if u and u not in d:
                d[u] = d[k]
            # unidecode emits apostrophes for some Indic letters; callers strip
            # punctuation, so register the stripped (and space-split) variants too
            u2 = ''.join(c for c in u if c.isalnum() or c == ' ')
            u3 = ' '.join(''.join(c if (c.isalnum() or c == ' ') else ' ' for c in u).split())
            for v in (u2, u3):
                if v and v not in d:
                    d[v] = d[k]


for _t in (IN_STATES, INDIC_LEGAL_ABBREV, CITY_ALIASES, FR_REGIONS, US_STATES, ADDR_ABBREV,
           LEGAL_SUFFIX_ABBREV, ORDINALS):
    _add_unidecoded_variants(_t)

__all__ = [
    'US_STATES', 'IN_STATES', 'FR_REGIONS', 'FR_DEPT_CODES', 'ADDR_ABBREV', 'ADDR_ABBREV_MULTIWORD',
    'ORDINALS', 'ORDINALS_RISKY', 'ORDINAL_SUFFIX_RE_HINT', 'LEGAL_SUFFIXES',
    'LEGAL_SUFFIX_DOTTED_RE_HINT', 'LEGAL_SUFFIX_TRAILING_DOT_RE_HINT', 'LEGAL_SUFFIX_ABBREV',
    'LEGAL_SUFFIX_ABBREV_KEYS', 'NAME_FILLER_WORDS', 'NAME_JUNK_PREFIX_RE_HINT',
    'NAME_JUNK_SUFFIX_RE_HINT', 'LEET_MAP', 'CITY_ALIASES', 'STOPWORDS_BLOCKING',
    'INDIC_SCRIPT_RANGES', 'detect_script', 'detect_indic_script', 'INDIC_LEGAL_ABBREV',
]

if __name__ == '__main__':
    import re
    _tables = [
        ('US_STATES', US_STATES), ('IN_STATES', IN_STATES), ('FR_REGIONS', FR_REGIONS),
        ('FR_DEPT_CODES', FR_DEPT_CODES), ('ADDR_ABBREV', ADDR_ABBREV),
        ('ADDR_ABBREV_MULTIWORD', ADDR_ABBREV_MULTIWORD), ('ORDINALS', ORDINALS),
        ('ORDINALS_RISKY', ORDINALS_RISKY), ('LEGAL_SUFFIXES', LEGAL_SUFFIXES),
        ('LEGAL_SUFFIX_ABBREV', LEGAL_SUFFIX_ABBREV), ('NAME_FILLER_WORDS', NAME_FILLER_WORDS),
        ('LEET_MAP', LEET_MAP), ('CITY_ALIASES', CITY_ALIASES),
        ('STOPWORDS_BLOCKING', STOPWORDS_BLOCKING), ('INDIC_SCRIPT_RANGES', INDIC_SCRIPT_RANGES),
        ('INDIC_LEGAL_ABBREV', INDIC_LEGAL_ABBREV),
    ]
    for _name, _tbl in _tables:
        print('%-24s %6d' % (_name, len(_tbl)))

    assert US_STATES['texas'] == 'tx' and US_STATES['tx'] == 'tx' and US_STATES['north carolina'] == 'nc'
    assert IN_STATES['telangana'] == 'tg' and IN_STATES['ts'] == 'tg' and IN_STATES['orissa'] == 'od'
    assert IN_STATES['महाराष्ट्र'] == 'mh' and IN_STATES[unidecode('महाराष्ट्र').lower()] == 'mh'
    assert FR_REGIONS['nord'] == 'hauts de france' and FR_REGIONS['gironde'] == 'nouvelle aquitaine'
    assert FR_REGIONS['aquitaine'] == 'nouvelle aquitaine' and FR_REGIONS['paris'] == 'ile de france'
    assert FR_DEPT_CODES['59'] == 'hauts de france' and FR_DEPT_CODES['2a'] == 'corse'
    assert FR_DEPT_CODES['971'] == 'guadeloupe' and FR_DEPT_CODES['13'] == 'provence alpes cote d azur'
    assert len([c for c, _n, _r in _FR_DEPTS]) == 101
    assert ADDR_ABBREV['street'] == 'st' and ADDR_ABBREV['boulevard'] == 'blvd' and ADDR_ABBREV['bd'] == 'blvd'
    assert ADDR_ABBREV['h no'] == 'h' and ADDR_ABBREV['sainte'] == 'ste' and ADDR_ABBREV['suite'] == 'ste'
    assert ADDR_ABBREV_MULTIWORD[0].count(' ') >= 1 and len(ADDR_ABBREV_MULTIWORD[0]) >= len(ADDR_ABBREV_MULTIWORD[-1])
    assert ORDINALS['ninth'] == '9' and ORDINALS['twenty first'] == '21' and ORDINALS['deuxieme'] == '2'
    assert ORDINALS_RISKY['un'] == '1' and 'un' not in ORDINALS
    assert re.sub(ORDINAL_SUFFIX_RE_HINT, '', '71st 1er 2eme 3e') == '71 1 2 3'
    assert LEGAL_SUFFIXES[0].count(' ') >= 1 and ' ' not in LEGAL_SUFFIXES[-1]
    assert LEGAL_SUFFIX_ABBREV['private limited'] == 'pvt ltd' and LEGAL_SUFFIX_ABBREV['societe anonyme'] == 'sa'
    assert re.sub(LEGAL_SUFFIX_DOTTED_RE_HINT, lambda m: re.sub(r'[.\s]', '', m.group(0)), 'acme s.a.r.l.') == 'acme sarl.'
    assert 'services' in NAME_FILLER_WORDS and 'enterprises' not in LEGAL_SUFFIXES
    assert LEET_MAP['0'] == 'o' and CITY_ALIASES['bombay'] == 'mumbai' and CITY_ALIASES['bengaluru'] == 'bangalore'
    assert CITY_ALIASES['st louis'] == 'saint louis' and 'secunderabad' in CITY_ALIASES
    assert 'de' in STOPWORDS_BLOCKING and 'st' in STOPWORDS_BLOCKING and 'ltd' in STOPWORDS_BLOCKING
    assert detect_script('फॉर्च्यून') == 'indic' and detect_script('café') == 'latin_accented' and detect_script('acme') == 'latin'
    assert detect_indic_script('ಕರ್ನಾಟಕ') == 'kannada'
    assert INDIC_LEGAL_ABBREV['प्रा'] == 'pvt' and INDIC_LEGAL_ABBREV['লিমিটেড'] == 'limited'
    for _name, _tbl in _tables:
        if isinstance(_tbl, dict):
            for _k in _tbl:
                assert _k == _k.lower().strip(), (_name, _k)
    print('all sanity checks passed')
