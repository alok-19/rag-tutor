from rag_tutor.study.quiz import parse_quiz, grade_quiz, Question, build_quiz_prompt
from rag_tutor.study.flashcards import parse_flashcards, Flashcard, build_flashcard_prompt
from rag_tutor.study.generate import _sources_to_context
from rag_tutor.retrieval.vector_store import RetrievedSource


# ============================================================
# Quiz prompt construction
# ============================================================

def test_quiz_prompt_contains_essentials():
    prompt = build_quiz_prompt("OS", "context here", num_questions=5)
    assert "OS" in prompt
    assert "context here" in prompt
    assert "JSON" in prompt
    assert "correct_index" in prompt


def test_flashcard_prompt_contains_essentials():
    prompt = build_flashcard_prompt("OS", "context here", num_cards=8)
    assert "OS" in prompt
    assert "context here" in prompt
    assert "JSON" in prompt
    assert "front" in prompt and "back" in prompt


# ============================================================
# Quiz parsing — well-formed
# ============================================================

def test_parse_quiz_well_formed():
    raw = '''{
      "quiz": [
        {"question": "What is paging?", "options": ["A", "B", "C", "D"], "correct_index": 1, "explanation": "because"},
        {"question": "What is a PCB?", "options": ["X", "Y", "Z", "W"], "correct_index": 0, "explanation": "reasons"}
      ]
    }'''
    qs = parse_quiz(raw)
    assert len(qs) == 2
    assert qs[0].question == "What is paging?"
    assert qs[0].options == ["A", "B", "C", "D"]
    assert qs[0].correct_index == 1
    assert qs[0].explanation == "because"


def test_parse_quiz_with_prose_preamble_and_fences():
    raw = '''Sure! Here is your quiz:
    ```json
    {"quiz": [{"question": "Q1", "options": ["a","b","c","d"], "correct_index": 2}]}
    ```
    Hope this helps!'''
    qs = parse_quiz(raw)
    assert len(qs) == 1
    assert qs[0].correct_index == 2
    assert qs[0].options == ["a", "b", "c", "d"]


def test_parse_quiz_letter_answer():
    raw = '{"quiz": [{"question": "Q1", "options": ["a","b","c","d"], "correct": "C"}]}'
    qs = parse_quiz(raw)
    assert len(qs) == 1
    assert qs[0].correct_index == 2  # C -> 2


def test_parse_quiz_trailing_commas():
    raw = '''{"quiz": [
        {"question": "Q1", "options": ["a","b","c","d"], "correct_index": 0,},
    ]}'''
    qs = parse_quiz(raw)
    assert len(qs) == 1


def test_parse_quiz_accepts_bare_list():
    raw = '[{"question": "Q1", "options": ["a","b"], "correct_index": 0}]'
    qs = parse_quiz(raw)
    assert len(qs) == 1


def test_parse_quiz_accepts_alt_keys():
    raw = '{"questions": [{"q": "Q1", "choices": ["a","b","c","d"], "answer_index": 3}]}'
    qs = parse_quiz(raw)
    assert len(qs) == 1
    assert qs[0].question == "Q1"
    assert qs[0].correct_index == 3


# ============================================================
# Quiz parsing — malformed / degradation
# ============================================================

def test_parse_quiz_empty():
    assert parse_quiz("") == []
    assert parse_quiz("   ") == []


def test_parse_quiz_completely_broken_json():
    assert parse_quiz("the model just rambled with no braces") == []


def test_parse_quiz_skips_invalid_questions():
    # First question missing options, second fine, third bad index.
    raw = '''{"quiz": [
        {"question": "no options", "correct_index": 0},
        {"question": "ok", "options": ["a","b","c","d"], "correct_index": 1},
        {"question": "bad idx", "options": ["a","b","c","d"], "correct_index": 99}
    ]}'''
    qs = parse_quiz(raw)
    assert len(qs) == 1
    assert qs[0].question == "ok"


def test_parse_quiz_truncated_json():
    raw = '{"quiz": [{"question": "Q1", "options": ["a","b","c","d"], "correct_index"'  # cut off
    qs = parse_quiz(raw)
    assert qs == []


def test_parse_quiz_too_few_options_rejected():
    raw = '{"quiz": [{"question": "Q1", "options": ["only one"], "correct_index": 0}]}'
    qs = parse_quiz(raw)
    assert qs == []


def test_parse_quiz_truncates_to_four_options():
    raw = '{"quiz": [{"question": "Q1", "options": ["a","b","c","d","e"], "correct_index": 0}]}'
    qs = parse_quiz(raw)
    assert len(qs[0].options) == 4


# ============================================================
# Grading
# ============================================================

def test_grade_quiz_all_correct():
    qs = [
        Question(question="a", options=["x","y"], correct_index=0),
        Question(question="b", options=["x","y"], correct_index=1),
    ]
    score = grade_quiz(qs, [0, 1])
    assert score.correct == 2
    assert score.total == 2
    assert score.percent == 100.0


def test_grade_quiz_partial():
    qs = [
        Question(question="a", options=["x","y"], correct_index=0),
        Question(question="b", options=["x","y"], correct_index=1),
    ]
    score = grade_quiz(qs, [0, 0])
    assert score.correct == 1
    assert score.total == 2
    assert score.percent == 50.0


def test_grade_quiz_missing_answers_safe():
    qs = [Question(question="a", options=["x","y"], correct_index=0)]
    # No answers provided at all.
    score = grade_quiz(qs, [])
    assert score.correct == 0
    assert score.total == 1


def test_grade_quiz_none_answer_incorrect():
    qs = [Question(question="a", options=["x","y"], correct_index=0)]
    score = grade_quiz(qs, [None])
    assert score.correct == 0


def test_grade_quiz_empty():
    score = grade_quiz([], [])
    assert score.correct == 0
    assert score.total == 0
    assert score.percent == 0.0


# ============================================================
# Flashcard parsing
# ============================================================

def test_parse_flashcards_well_formed():
    raw = '{"cards": [{"front": "What is X?", "back": "X is..."}, {"front": "Term", "back": "Def"}]}'
    cards = parse_flashcards(raw)
    assert len(cards) == 2
    assert cards[0].front == "What is X?"
    assert cards[1].back == "Def"


def test_parse_flashcards_with_fences():
    raw = '```json\n{"cards": [{"front": "F", "back": "B"}]}\n```'
    cards = parse_flashcards(raw)
    assert len(cards) == 1


def test_parse_flashcards_empty_fields_rejected():
    raw = '{"cards": [{"front": "", "back": "B"}, {"front": "F", "back": "B"}]}'
    cards = parse_flashcards(raw)
    assert len(cards) == 1
    assert cards[0].front == "F"


def test_parse_flashcards_broken_returns_empty():
    assert parse_flashcards("nope") == []
    assert parse_flashcards("") == []


def test_parse_flashcards_accepts_alt_keys():
    raw = '{"flashcards": [{"term": "T", "definition": "D"}]}'
    cards = parse_flashcards(raw)
    assert len(cards) == 1
    assert cards[0].front == "T"
    assert cards[0].back == "D"


# ============================================================
# Context helper
# ============================================================

def test_sources_to_context_formats_passages():
    sources = [
        RetrievedSource(source="a.pdf", page=3, text="passage one"),
        RetrievedSource(source="b.pdf", page=7, text="passage two"),
    ]
    ctx = _sources_to_context(sources)
    assert "a.pdf" in ctx
    assert "Page: 3" in ctx
    assert "[1]" in ctx and "[2]" in ctx
    assert "passage one" in ctx


def test_sources_to_context_empty():
    assert "No context" in _sources_to_context([])
