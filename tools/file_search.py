# import os
# import pandas as pd
# from rapidfuzz import process

# _excel_data_cache = {}  # cache of sheet_name -> DataFrame
# _EXCEL_FILE_PATH = "tools/Markaz.xlsx"  # fixed path to your file

# def get_excel_context(query: str) -> str:
#     """
#     Search the fixed Excel workbook with multiple sheets for relevant info.

#     Args:
#         query: User question

#     Returns:
#         String with relevant info to include in Gemma prompt
#     """
#     global _excel_data_cache

#     # Step 1: Load all sheets if not cached
#     if not _excel_data_cache:
#         if not os.path.exists(_EXCEL_FILE_PATH):
#             raise FileNotFoundError(f"Excel file not found: {_EXCEL_FILE_PATH}")
#         xls = pd.ExcelFile(_EXCEL_FILE_PATH)
#         for sheet in xls.sheet_names:
#             df = pd.read_excel(xls, sheet_name=sheet)
#             if not {'Savollar', 'Javoblar'}.issubset(df.columns.str.lower()):
#                 raise ValueError(f"Sheet '{sheet}' must have columns 'Savollar' and 'Javoblar'.")
#             _excel_data_cache[sheet] = df

#     # Step 2: Detect theme/sheet based on query
#     themes = list(_excel_data_cache.keys())
#     best_match = process.extractOne(query, themes, score_cutoff=60)
#     detected_theme = best_match[0] if best_match else None

#     # Step 3: Search Savollar in the selected sheet
#     if detected_theme:
#         df = _excel_data_cache[detected_theme]
#     else:
#         # fallback: search all sheets
#         df = pd.concat(_excel_data_cache.values(), ignore_index=True)

#     # Case-insensitive search in 'Savollar' column
#     mask = df['Savollar'].astype(str).str.contains(query, case=False)
#     results = df[mask]

#     if not results.empty:
#         Javoblar = results['Javoblar'].tolist()
#         return f"I found this info in theme '{detected_theme or 'general'}':\n" + "\n".join(Javoblar)
#     else:
#         return ""


#################################################################################################################

# import os
# import pandas as pd
# from rapidfuzz import process, fuzz
# import re

# _excel_data_cache = {}
# _EXCEL_FILE_PATH = "tools/Markaz.xlsx"

# def get_excel_context(query: str) -> str:
#     """
#     Search Excel file using semantic matching and fuzzy search.
#     Finds relevant information even if the query doesn't exactly match.
#     """
#     global _excel_data_cache

#     # Load sheets if not cached
#     if not _excel_data_cache:
#         if not os.path.exists(_EXCEL_FILE_PATH):
#             return f"Excel file not found: {_EXCEL_FILE_PATH}"

#         try:
#             xls = pd.ExcelFile(_EXCEL_FILE_PATH)
#             for sheet_name in xls.sheet_names:
#                 df = pd.read_excel(xls, sheet_name=sheet_name)

#                 if df.empty:
#                     continue

#                 # Find question and answer columns automatically
#                 question_col = None
#                 answer_col = None

#                 # Look for question column (more flexible matching)
#                 for col in df.columns:
#                     col_str = str(col).lower()
#                     if any(keyword in col_str for keyword in [
#                         'savol', 'question', 'q', 'sual', 'mashq', 'topshiriq'
#                     ]):
#                         question_col = col
#                         break

#                 # Look for answer column
#                 for col in df.columns:
#                     col_str = str(col).lower()
#                     if any(keyword in col_str for keyword in [
#                         'javob', 'answer', 'a', 'response', 'reply', 'natija', 'yechim'
#                     ]):
#                         answer_col = col
#                         break

#                 # If no specific columns found, use first two columns
#                 if question_col is None or answer_col is None:
#                     if len(df.columns) >= 2:
#                         question_col = df.columns[0]
#                         answer_col = df.columns[1]
#                     else:
#                         continue

#                 # Clean and standardize the data
#                 df_clean = df[[question_col, answer_col]].copy()
#                 df_clean.columns = ['questions', 'answers']
#                 df_clean = df_clean.dropna(subset=['questions', 'answers'])
#                 df_clean['questions'] = df_clean['questions'].astype(str)
#                 df_clean['answers'] = df_clean['answers'].astype(str)

#                 _excel_data_cache[sheet_name] = df_clean

#         except Exception as e:
#             return f"Error loading Excel file: {str(e)}"

#     if not _excel_data_cache:
#         return "No valid data found in Excel file."

#     # Extract keywords from query for better matching
#     query_keywords = extract_keywords(query)
#     query_lower = query.lower()

#     all_matches = []

#     # Search through each sheet
#     for sheet_name, df in _excel_data_cache.items():
#         sheet_matches = []

#         # 1. Direct fuzzy matching on questions
#         questions = df['questions'].tolist()
#         fuzzy_matches = process.extract(
#             query, questions,
#             scorer=fuzz.partial_ratio,
#             limit=5,
#             score_cutoff=45
#         )

#         for match in fuzzy_matches:
#             match_text = match[0]
#             score = match[1]
#             row = df[df['questions'] == match_text]
#             if not row.empty:
#                 answer = row['answers'].iloc[0]
#                 sheet_matches.append({
#                     'answer': answer,
#                     'score': score,
#                     'match_type': 'question_fuzzy',
#                     'sheet': sheet_name
#                 })


#         # 2. Keyword-based matching
#         for _, row in df.iterrows():
#             question = row['questions'].lower()
#             answer = row['answers'].lower()

#             # Count keyword matches in questions and answers
#             q_matches = sum(1 for kw in query_keywords if kw in question)
#             a_matches = sum(1 for kw in query_keywords if kw in answer)

#             if q_matches > 0 or a_matches > 0:
#                 # Calculate score based on keyword density
#                 total_keywords = len(query_keywords)
#                 keyword_score = ((q_matches * 2 + a_matches) / total_keywords) * 100

#                 if keyword_score > 20:  # Threshold for relevance
#                     sheet_matches.append({
#                         'answer': row['answers'],
#                         'score': keyword_score,
#                         'match_type': 'keyword',
#                         'sheet': sheet_name
#                     })

#         # 3. Semantic matching for specific domains
#         for _, row in df.iterrows():
#             question = row['questions'].lower()
#             answer = row['answers'].lower()
#             content = f"{question} {answer}"

#             semantic_score = calculate_semantic_score(query_lower, content)
#             if semantic_score > 30:
#                 sheet_matches.append({
#                     'answer': row['answers'],
#                     'score': semantic_score,
#                     'match_type': 'semantic',
#                     'sheet': sheet_name
#                 })

#         all_matches.extend(sheet_matches)

#     if not all_matches:
#         return f"No relevant information found for: '{query}'"

#     # Remove duplicates and sort by score
#     unique_matches = []
#     seen_answers = set()

#     for match in sorted(all_matches, key=lambda x: x['score'], reverse=True):
#         answer_clean = match['answer'].strip().lower()[:100]  # First 100 chars for dedup
#         if answer_clean not in seen_answers:
#             seen_answers.add(answer_clean)
#             unique_matches.append(match)

#     # Return top matches
#     results = []
#     for i, match in enumerate(unique_matches[:3]):  # Top 3 matches
#         score = int(match['score'])
#         sheet = match['sheet']
#         answer = match['answer']
#         results.append(f"[{score}% match from {sheet}] {answer}")

#     return "\n\n".join(results)


# def extract_keywords(text: str) -> list:
#     """Extract meaningful keywords from query"""
#     # Remove common stop words in Uzbek and Russian
#     stop_words = {
#         'va', 'yoki', 'lekin', 'uchun', 'bilan', 'dan', 'ga', 'ni', 'ning',
#         'qanday', 'nima', 'kim', 'qayer', 'qachon', 'nega', 'nima',
#         'и', 'или', 'но', 'для', 'с', 'от', 'к', 'что', 'как', 'кто', 'где', 'когда'
#     }

#     # Extract words (3+ characters, not stop words)
#     words = re.findall(r'\b\w{3,}\b', text.lower())
#     keywords = [word for word in words if word not in stop_words]

#     return keywords


# def calculate_semantic_score(query: str, content: str) -> float:
#     """Calculate semantic similarity score"""

#     # Domain-specific keyword matching for cybersecurity
#     cybersecurity_terms = {
#         'kiberxavfsizlik': ['cyber', 'security', 'xavfsizlik', 'himoya'],
#         'markaz': ['center', 'centr', 'institut', 'tashkilot', 'organization'],
#         'tashkilot': ['organization', 'org', 'muassasa', 'markaz'],
#         'xavfsizlik': ['security', 'himoya', 'protection', 'защита'],
#         'kiberhujum': ['cyberattack', 'attack', 'hujum'],
#         'malware': ['virus', 'trojan', 'zararli'],
#         'parol': ['password', 'pin', 'kod'],
#         'shifrlash': ['encryption', 'encrypt', 'crypto'],
#         'firewall': ['mudofaa', 'himoya', 'filter'],
#         'phishing': ['fishing', 'aldash', 'fake'],
#         'ddos': ['attack', 'hujum', 'dos'],
#         'antivirus': ['himoya', 'virus', 'protection']
#     }

#     score = 0
#     query_words = set(query.lower().split())
#     content_words = set(content.lower().split())

#     # Check for domain-specific term matches
#     for query_word in query_words:
#         if query_word in cybersecurity_terms:
#             related_terms = cybersecurity_terms[query_word]
#             for term in related_terms:
#                 if any(term in content_word for content_word in content_words):
#                     score += 30

#     # General word matching
#     common_words = query_words.intersection(content_words)
#     if common_words:
#         score += (len(common_words) / len(query_words)) * 50

#     # Partial word matching
#     for q_word in query_words:
#         for c_word in content_words:
#             if len(q_word) > 3 and (q_word in c_word or c_word in q_word):
#                 score += 10

#     return min(score, 100)  # Cap at 100


# # Test function to check what's in your Excel file
# def debug_excel_content():
#     """Debug function to see what's actually in your Excel file"""
#     global _excel_data_cache
#     _excel_data_cache = {}  # Reset cache

#     if not os.path.exists(_EXCEL_FILE_PATH):
#         print(f"❌ File not found: {_EXCEL_FILE_PATH}")
#         return

#     try:
#         xls = pd.ExcelFile(_EXCEL_FILE_PATH)
#         print(f"📋 Sheets found: {xls.sheet_names}")

#         for sheet_name in xls.sheet_names:
#             print(f"\n📄 Sheet: {sheet_name}")
#             df = pd.read_excel(xls, sheet_name=sheet_name)
#             print(f"   Columns: {list(df.columns)}")
#             print(f"   Rows: {len(df)}")

#             if not df.empty:
#                 print("   Sample data:")
#                 for i, row in df.head(3).iterrows():
#                     print(f"   Row {i}: {dict(row)}")
#     except Exception as e:
#         print(f"❌ Error: {e}")

# # Uncomment this line to debug your Excel file:
# # debug_excel_content()


#########################################################################################################################


import os
import re
import unicodedata
import pandas as pd
from rapidfuzz import fuzz

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
_EXCEL_FILE_PATH = os.getenv("MARKAZ_EXCEL_PATH", "tools/Markaz.xlsx")
_excel_data_cache = {}  # Cache: { sheet_name: DataFrame[questions, answers] }

# -------------------------------------------------------------------
# PUBLIC FUNCTIONS
# -------------------------------------------------------------------


def get_excel_context(query: str, top_k: int = 3) -> str:
    """
    Search Excel file for relevant info.
    Returns top-k answers formatted for LLM context.
    """
    ensure_loaded()
    if not _excel_data_cache:
        return f"Excel file not found or empty: {_EXCEL_FILE_PATH}"

    q_norm = _normalize(query)
    matches = _collect_matches(q_norm)

    if not matches:
        return f"No relevant information found for: '{query}'"

    lines = []
    for m in matches[:top_k]:
        lines.append(f"[{m['score']}% match from {m['sheet']}] {m['answer']}")
    return "\n\n".join(lines)


def get_excel_answer(query: str) -> str:
    """
    Returns only the single best answer as a string.
    If nothing is found, returns a message indicating that.
    """
    ensure_loaded()
    if not _excel_data_cache:
        return f"Excel file not found or empty: {_EXCEL_FILE_PATH}"

    q_norm = _normalize(query)
    matches = _collect_matches(q_norm)

    if not matches:
        return f"No relevant info found for: '{query}'"

    best_answer = matches[0].get("answer", "")
    if not str(best_answer).strip():
        return f"An entry was found for '{query}', but the answer is empty."
    return str(best_answer)


def debug_excel_content():
    """Prints all sheet names, columns, and first 3 rows for debugging."""
    if not os.path.exists(_EXCEL_FILE_PATH):
        print(f"❌ File not found: {_EXCEL_FILE_PATH}")
        return
    try:
        xls = pd.ExcelFile(_EXCEL_FILE_PATH)
        print(f"📋 Sheets: {xls.sheet_names}")
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            print(f"\n📄 Sheet: {sheet}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Rows: {len(df)}")
            for i, row in df.head(3).iterrows():
                print(f"   Row {i}: {dict(row)}")
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")

# -------------------------------------------------------------------
# INTERNAL HELPERS
# -------------------------------------------------------------------


def ensure_loaded():
    """Load and normalize all sheets into cache once."""
    global _excel_data_cache
    if _excel_data_cache:
        return

    if not os.path.exists(_EXCEL_FILE_PATH):
        _excel_data_cache = {}
        return

    try:
        xls = pd.ExcelFile(_EXCEL_FILE_PATH)
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            if df is None or df.empty:
                continue

            q_col, a_col = _detect_question_answer_columns(df)
            if not q_col or not a_col:
                continue

            df_clean = df[[q_col, a_col]].copy()
            df_clean.columns = ["questions", "answers"]
            df_clean.dropna(subset=["questions", "answers"], inplace=True)
            df_clean["questions"] = df_clean["questions"].astype(str)
            df_clean["answers"] = df_clean["answers"].astype(str)

            # Store normalized versions for matching
            df_clean["_q_norm"] = df_clean["questions"].map(_normalize)
            df_clean["_a_norm"] = df_clean["answers"].map(_normalize)

            _excel_data_cache[sheet] = df_clean.reset_index(drop=True)
    except Exception as e:
        _excel_data_cache = {}
        print(f"❌ Error loading Excel: {e}")


def _detect_question_answer_columns(df: pd.DataFrame):
    """
    Finds the question column by keyword and assumes the answer is in the next column.
    Falls back to the first two columns if detection fails.
    """
    q_col = None
    a_col = None
    col_list = list(df.columns)

    # 1. Find the question column by keyword and take the next column as the answer
    for i, col_name in enumerate(col_list):
        c = str(col_name).strip().lower()
        if any(k in c for k in ["savol", "question", "q", "sual", "topshiriq"]):
            if i + 1 < len(col_list):
                q_col = col_name
                a_col = col_list[i + 1]
                break

    # 2. Fallback if the above logic fails
    if q_col is None or a_col is None and len(col_list) >= 2:
        q_col = col_list[0]
        a_col = col_list[1]

    return q_col, a_col


# Strong Uzbek text normalization
_APOSTROPHES = ["ʼ", "’", "‘", "`", "´", "ʹ", "ʽ", "ˈ", "ˊ", "ʻ", "’"]


def _normalize(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    for ap in _APOSTROPHES:
        s = s.replace(ap, "'")
    s = s.replace("o‘", "o'").replace("g‘", "g'")
    s = s.replace("oʼ", "o'").replace("gʼ", "g'")
    s = s.replace("o`", "o'").replace("g`", "g'")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Domain-specific synonyms
_SYNONYM_SETS = {
    "kiber_markaz": [
        "kiber xavfsizlik markazi", "kiberxavfsizlik markazi",
        "cyber security center", "cybersecurity center",
        "axborot xavfsizligi markazi", "axborot xavfsizligi milliy markazi",
        "kiber markaz", "kiberxavfsizlik", "kiber xavfsizlik"
    ],
    "rahbar": [
        "rahbar", "direktor", "boshliq", "director", "chief", "head", "rahbari", "rəhbari"
    ]
}

_SHEET_BOOSTS = {
    "rahbariyat": 12,
    "markaz": 8,
    "direktsiya": 10,
}


def _collect_matches(q_norm: str):
    """Score each row in all sheets and return ranked matches."""
    all_rows = []
    for sheet, df in _excel_data_cache.items():
        sheet_boost = _sheet_boost(sheet)
        for _, row in df.iterrows():
            qn = row["_q_norm"]
            an = row["_a_norm"]

            if not qn:
                continue

            s_q = fuzz.token_set_ratio(q_norm, qn)
            if fuzz.ratio(q_norm, qn) > 95:
                s_q = 100

            s_syn = _synonym_boost(q_norm, qn + " " + an)
            s_intent = _intent_boost(q_norm, qn + " " + an)
            score = min(100, int(s_q + s_syn + s_intent + sheet_boost))
            if score >= 65:
                all_rows.append({
                    "sheet": sheet,
                    "question": row["questions"],
                    "answer": row["answers"],
                    "score": score
                })

    if not all_rows:
        return []

    # Deduplicate by normalized answer text
    seen = set()
    uniq = []
    for m in sorted(all_rows, key=lambda x: x["score"], reverse=True):
        key = _normalize(m["answer"])[:120]
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    return uniq


def _sheet_boost(sheet_name: str) -> int:
    s = _normalize(sheet_name)
    for key, boost in _SHEET_BOOSTS.items():
        if key in s:
            return boost
    return 0


def _synonym_boost(q_norm: str, content_norm: str) -> int:
    boost = 0
    for group in _SYNONYM_SETS.values():
        in_q = any(v in q_norm for v in group)
        in_c = any(v in content_norm for v in group)
        if in_q and in_c:
            boost += 12
    return boost


def _intent_boost(q_norm: str, content_norm: str) -> int:
    """Extra weight if asking about 'who?/leader' and content has leader terms."""
    intent_kim = (" kim" in f" {q_norm} ") or ("kim?" in q_norm)
    intent_lead = any(t in q_norm for t in _SYNONYM_SETS["rahbar"])
    has_lead_terms = any(t in content_norm for t in _SYNONYM_SETS["rahbar"])
    return 15 if (intent_kim or intent_lead) and has_lead_terms else 0
