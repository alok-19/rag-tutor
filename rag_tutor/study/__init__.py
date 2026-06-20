"""Structured study artifacts generated from study-material context."""
from .quiz import Question, Score, build_quiz_prompt, parse_quiz, grade_quiz
from .flashcards import Flashcard, build_flashcard_prompt, parse_flashcards
from .generate import generate_quiz, generate_flashcards

__all__ = [
    "Question",
    "Score",
    "build_quiz_prompt",
    "parse_quiz",
    "grade_quiz",
    "Flashcard",
    "build_flashcard_prompt",
    "parse_flashcards",
    "generate_quiz",
    "generate_flashcards",
]
