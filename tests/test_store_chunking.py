from pathlib import Path

from ledgerline.store.chunking import _periodic_sections, chunk_document

# Real Apple Q3 FY2026 10-Q text (TOC + the first ~1200 chars of every section).
TEN_Q = (Path(__file__).parent / "fixtures" / "aapl_10q_excerpt.txt").read_text()


def test_offsets_are_exact_slices():
    for c in chunk_document(TEN_Q, "10-Q", max_tokens=60, overlap=10):
        assert TEN_Q[c.char_start:c.char_end] == c.text
        assert c.text.strip() and c.token_count <= 60


def test_toc_does_not_create_sections():
    paths = [p for _, p in _periodic_sections(TEN_Q)]
    assert len(paths) == len(set(paths))  # the TOC repeats every header; each kept once
    assert "Part I > Item 2. Management’s Discussion and Analysis of Financial Condition " \
           "and Results of Operations" in paths
    assert "Part II > Item 1A. Risk Factors" in paths
    chunks = chunk_document(TEN_Q, "10-Q")
    toc = TEN_Q.index("TABLE OF CONTENTS")
    assert next(c for c in chunks if c.char_start <= toc < c.char_end).section_path == "Cover"
    # Bare "PART I — FINANCIAL INFORMATION" lines carry no content, so no chunk of their own.
    assert all(" > " in c.section_path for c in chunks if c.section_path != "Cover")


def test_chunks_never_cross_sections_and_overlap_within_them():
    bounds = [0] + [o for o, _ in _periodic_sections(TEN_Q)] + [len(TEN_Q)]
    chunks = chunk_document(TEN_Q, "10-Q", max_tokens=60, overlap=15)
    for c in chunks:
        # every chunk lies strictly inside one [boundary, next boundary) interval
        assert any(a <= c.char_start and c.char_end <= b for a, b in zip(bounds, bounds[1:]))
    same_section = [(a, b) for a, b in zip(chunks, chunks[1:]) if a.section_path == b.section_path]
    assert same_section and any(b.char_start < a.char_end for a, b in same_section)
    assert all(b.char_start > a.char_start for a, b in same_section)  # always progresses


def test_cross_reference_is_not_a_header():
    text = ("PART I\nItem 2. Results\nSee Part I,\nItem 1A of the 2025 Form 10-K for risks.\n"
            "Net sales rose.\n")
    assert [p for _, p in _periodic_sections(text)] == ["Part I", "Part I > Item 2. Results"]


def test_press_release_headings_and_run_on_split():
    para = "Revenue grew strongly in every region this quarter. " * 6
    text = (f"Company reports results\n{para}\nCONDENSED CONSOLIDATED BALANCE SHEETS\n"
            f"Cash\n1,234\n{'word ' * 2000}\n")
    chunks = chunk_document(text, "8-K")
    assert {c.section_path for c in chunks} == {
        "Company reports results", "CONDENSED CONSOLIDATED BALANCE SHEETS"}
    assert all(c.token_count <= 800 for c in chunks)
    assert all(text[c.char_start:c.char_end] == c.text for c in chunks)
