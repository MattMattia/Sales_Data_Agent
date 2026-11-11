"""Tests for i18n.py: translation parity and language normalization."""

import pytest

import i18n


def test_every_ui_string_exists_in_both_languages():
    english, spanish = i18n.TRANSLATIONS["en"], i18n.TRANSLATIONS["es"]

    assert set(english) == set(spanish)
    assert all(value.strip() for value in [*english.values(), *spanish.values()])


@pytest.mark.parametrize(
    "tag, expected",
    [("es", "es"), ("es-AR", "es"), ("es_419", "es"), ("ES-mx", "es"), ("en-US", "en"),
     ("fr-FR", "en"), ("", "en"), (None, "en")],
)
def test_normalize_language_maps_locales_to_supported_codes(tag, expected):
    assert i18n.normalize_language(tag) == expected


def test_t_formats_values_and_falls_back_to_english():
    assert i18n.t("graph_unavailable", "es", error="boom") == "Error: no se pudo cargar el motor del grafo: boom"
    assert i18n.t("clear_conversation", "de") == "🗑️ Clear Conversation"


def test_language_instruction_targets_latin_american_spanish():
    assert "español latinoamericano (es-419)" in i18n.language_instruction("es-CL")
    assert i18n.language_instruction(None) == "Write your answer in English."


def test_each_language_offers_three_sample_queries():
    assert len(i18n.sample_queries("en")) == len(i18n.sample_queries("es")) == 3


def test_readmes_link_to_each_other():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    english = (root / "README.md").read_text(encoding="utf-8")
    spanish = (root / "README.es.md").read_text(encoding="utf-8")

    assert "[Español (Latinoamérica)](README.es.md)" in english.splitlines()[2]
    assert "[English](README.md)" in spanish.splitlines()[2]
