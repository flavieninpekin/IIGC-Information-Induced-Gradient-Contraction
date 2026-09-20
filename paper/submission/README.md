# ICLR 2027 Submission

Anonymous submission source built from the canonical main text
(`paper/drafts/main_text_v1.md`; Chinese reading companion:
`paper/drafts/main_text_v1_zh.md`).

Two PDFs are produced from the same content:

| File | Use | Author block |
|---|---|---|
| `main.tex` / `main.pdf` | Anonymous ICLR 2027 submission | Hidden by the style |
| `main_discussion.tex` / `main_discussion.pdf` | Draft to share with colleagues | Placeholder, fill before sending |

Do not upload the discussion version to OpenReview. Chinese cover note for the
discussion email: `paper/discussion_email_zh.md`. Official style package:
`https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip`
(verified unchanged on 2026-09-16).

## Layout

```text
paper/submission/iclr2027/
├── main.tex               ← paper source (edit this)
├── references.bib         ← bibliography (verify TODO entries before submission)
├── main.bbl               ← generated; keep for portable source bundles
├── main.pdf               ← generated; the file uploaded to OpenReview
├── math_commands.tex      ← notation macros (official)
└── iclr2027_conference.*  ← official style and bst (from media.iclr.cc),
                             plus bundled natbib.sty and fancyhdr.sty
```

## Compile

```text
latexmk -pdf main.tex
```

requires a Perl runtime; on the current Windows/MiKTeX setup use the manual
chain instead:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Last verified build: 10 pages total; references start on page 9, so the main
text stays within the 9-page limit (references and appendix do not count).

The real-data check in Section 5.5 uses DICES-350. The CSV is downloaded from
the public dataset repository into `data/external/dices/` (gitignored, not
redistributed). To regenerate the audit numbers:

```text
python experiments/common_basis/supervised/run_dices_group_audit.py
```

## Before submitting

1. Verify the `TODO` comment in `references.bib`.
2. If the companion phenomenon paper is public by the submission date, add a
   third-person citation for it in the Related Work paragraph; otherwise keep
   the current "citation withheld for double-blind review" wording.
3. Confirm that all numbers still match `notes/canonical_metric_results.md`.
4. Check the required AI-use statement and, if applicable, the ethics
   statement.
5. Rebuild and confirm the main text is at most 9 pages; references and
   appendix do not count toward the limit.
6. If any field definition changed, regenerate the controlled visibility
   contrast and the Overcooked field axis:
   `python experiments/common_basis/toy/verify_visibility_control.py` and
   `python experiments/common_basis/server_tasks/run_field_axis.py --force`.
