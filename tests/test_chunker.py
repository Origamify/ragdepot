"""Chunker unit tests: packing, headings, junk, empty input."""

from ragdepot.chunker import MIN_WORDS, TARGET_WORDS, chunk


def para(words: int, word: str = "w") -> str:
    return " ".join([f"{word}{i}" for i in range(words)])


def test_empty_input():
    assert chunk("") == []
    assert chunk("\n\n  \n\n") == []


def test_heading_attachment():
    text = "# Title\n\n" + para(50) + "\n\n## Sub\n\n" + para(50)
    chunks = chunk(text)
    assert [c.heading for c in chunks] == ["Title", "Sub"]
    assert all("Title" not in c.text and "Sub" not in c.text for c in chunks)


def test_packing_under_target():
    chunks = chunk("# H\n\n" + para(400) + "\n\n" + para(400) + "\n\n" + para(400))
    assert len(chunks) == 2
    assert all(TARGET_WORDS >= len(c.text.split()) >= 400 for c in chunks)


def test_oversized_paragraph_not_split():
    chunks = chunk(para(TARGET_WORDS + 500))
    assert len(chunks) == 1
    assert len(chunks[0].text.split()) == TARGET_WORDS + 500


def test_heading_boundary_forces_new_chunk():
    # two tiny paragraphs under target, but different headings -> two chunks
    text = "# A\n\n" + para(30) + "\n\n# B\n\n" + para(30)
    chunks = chunk(text)
    assert [c.heading for c in chunks] == ["A", "B"]


def test_junk_dropped():
    text = "# Nav\n\nHome | FAQ | Contact\n\n# Real\n\n" + para(100)
    chunks = chunk(text)
    assert len(chunks) == 1
    assert chunks[0].heading == "Real"
    assert "Home | FAQ" not in chunks[0].text


def test_boundary_fragments_dropped():
    # 800-word chunk + 5-word fragment -> fragment dropped
    chunks = chunk(para(TARGET_WORDS) + "\n\n" + para(5))
    assert len(chunks) == 1
    assert len(chunks[0].text.split()) == TARGET_WORDS


def test_min_words_allows_valid_short_chunks():
    assert len(chunk(para(MIN_WORDS))) == 1
    assert chunk(para(MIN_WORDS - 1)) == []


def test_heading_only_input():
    assert chunk("# Just\n\n## Headings\n\n# Here") == []
