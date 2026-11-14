# RAG corpus sources

Ten public monetary-policy statements, one per `.txt` file, used by `scripts/build_rag.py`
and `scripts/eval_rag.py`. Each page was fetched once with `curl` on 2026-11-14. Only the
statement body was kept: navigation, share buttons, release-time lines, media-contact
lines and related links were removed, whitespace was normalised and two non-breaking
hyphens (in "12-month" in the 2020-03-15 statement and "mortgage-backed" in the
2024-09-18 statement) became plain hyphens, but the wording was not changed. Each file
starts with `Source: <URL> (retrieved 2026-11-14)`, then `Title:`, `Date:` and
`Publisher:` lines. `load_corpus()` moves these four lines into metadata, so only the
statement text is chunked and embedded.

| File | Title | Date | URL | Licence / terms | Distinctive content (for writing questions) |
|---|---|---|---|---|---|
| `fed_fomc_statement_2008-12-16.txt` | FOMC statement | 2008-12-16 | https://www.federalreserve.gov/newsevents/pressreleases/monetary20081216b.htm | Public domain (Fed) | target range set to 0 to 1/4 percent; agency debt and MBS purchases; Term Asset-Backed Securities Loan Facility; discount rate cut to 1/2 percent |
| `fed_fomc_statement_2015-12-16.txt` | Federal Reserve issues FOMC statement | 2015-12-16 | https://www.federalreserve.gov/newsevents/pressreleases/monetary20151216a.htm | Public domain (Fed) | target range raised to 1/4 to 1/2 percent; policy "remains accommodative"; only gradual increases expected |
| `fed_fomc_statement_2020-03-15.txt` | Federal Reserve issues FOMC statement | 2020-03-15 | https://www.federalreserve.gov/newsevents/pressreleases/monetary20200315a.htm | Public domain (Fed) | coronavirus; target range lowered to 0 to 1/4 percent; Treasury holdings up by at least $500 billion, agency MBS by at least $200 billion; Loretta J. Mester dissented |
| `fed_fomc_statement_2022-03-16.txt` | Federal Reserve issues FOMC statement | 2022-03-16 | https://www.federalreserve.gov/newsevents/pressreleases/monetary20220316a.htm | Public domain (Fed) | invasion of Ukraine by Russia; target range raised to 1/4 to 1/2 percent; balance-sheet reduction to begin at a coming meeting; James Bullard preferred a 0.5 point rise |
| `fed_fomc_statement_2023-07-26.txt` | Federal Reserve issues FOMC statement | 2023-07-26 | https://www.federalreserve.gov/newsevents/pressreleases/monetary20230726a.htm | Public domain (Fed) | "U.S. banking system is sound and resilient"; target range raised to 5-1/4 to 5-1/2 percent |
| `fed_fomc_statement_2024-09-18.txt` | Federal Reserve issues FOMC statement | 2024-09-18 | https://www.federalreserve.gov/newsevents/pressreleases/monetary20240918a.htm | Public domain (Fed) | cut by 1/2 percentage point to 4-3/4 to 5 percent; Michelle W. Bowman preferred a 1/4 point cut |
| `fed_fomc_statement_2026-10-28.txt` | Federal Reserve issues FOMC statement | 2026-10-28 | https://www.federalreserve.gov/newsevents/pressreleases/monetary20261028a.htm | Public domain (Fed) | target range kept at 3-3/4 to 4 percent; 10 – 2 vote; Lisa D. Cook and Beth M. Hammack preferred a 1/4 point rise |
| `ecb_monetary_policy_decisions_2022-07-21.txt` | Monetary policy decisions (ECB press release) | 2022-07-21 | https://www.ecb.europa.eu/press/pr/date/2022/html/ecb.mp220721~53e5bdd317.en.html | © ECB, reproduced with acknowledgement | key ECB rates raised by 50 basis points (0.50%, 0.75%, 0.00% from 27 July 2022); Transmission Protection Instrument (TPI) approved |
| `ecb_monetary_policy_decisions_2024-06-06.txt` | Monetary policy decisions (ECB press release) | 2024-06-06 | https://www.ecb.europa.eu/press/pr/date/2024/html/ecb.mp240606~2148ecdb3c.en.html | © ECB, reproduced with acknowledgement | key ECB rates lowered by 25 basis points (4.25%, 4.50%, 3.75% from 12 June 2024); PEPP holdings cut by €7.5 billion per month in the second half of 2024; staff projections |
| `ecb_monetary_policy_decisions_2026-10-29.txt` | Monetary policy decisions (ECB press release) | 2026-10-29 | https://www.ecb.europa.eu/press/pr/date/2026/html/ecb.mp261029~9a493adf17.en.html | © ECB, reproduced with acknowledgement | key ECB rates unchanged (deposit facility 2.50%, main refinancing 2.65%, marginal lending 2.90%); energy shock |

Watch out when writing `data/rag_eval.csv`: two Fed statements (2015-12-16 and
2022-03-16) both raise the target range to 1/4 to 1/2 percent, and two (2008-12-16 and
2020-03-15) both set it to 0 to 1/4 percent. A question about those ranges needs another
detail (for example "coronavirus" or "Ukraine") to be answerable from exactly one file.
All three ECB releases mention the Transmission Protection Instrument; only 2022-07-21
approves it, so a TPI question should ask about the approval.
Dates are not part of the embedded text either (FOMC statements do not repeat their own
date; it is only in the file name and the `Date:` line, which are metadata), so ask about
content: "What did the Fed decide in October 2026?" gives the retriever nothing
distinctive to match, while "Which two officials preferred to raise the target range by
1/4 percentage point?" does.

## Licence and terms notes (checked 2026-11-14)

- **Federal Reserve Board** (`fed_*.txt`): the Board's website policy
  (https://www.federalreserve.gov/disclaimer.htm) says: "Unless otherwise indicated,
  information on Board's website is in the public domain and may be copied and
  distributed without permission. Please cite to the Board as the source of the
  information." The source URL is the first line of every file.
- **European Central Bank** (`ecb_*.txt`): Copyright © European Central Bank, Frankfurt
  am Main, Germany. The ECB's disclaimer and copyright page
  (https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html)
  allows free use of information taken directly from the website provided that it is
  reproduced accurately and the ECB is cited as the source (the exception, documents
  that bear their authors' names such as Working Papers, does not apply to these press
  releases). Source: European Central Bank, URL on the first line of each file. Only the
  press-release body was extracted; the text itself is unchanged.

## To add manually

- **Reserve Bank of India (RBI)**: not copied. The RBI website terms
  (`www.rbi.org.in`, page "Disclaimer", checked 2026-11-14) state "© Reserve Bank of
  India. All Rights Reserved.", prohibit caching of the site or its contents, and require
  prior permission from RBI even for hyperlinks to internal pages. They give no
  permission to reproduce text, so no RBI text is included and no deep links are listed
  here. If you want RBI monetary policy statements (for example the Monetary Policy
  Committee resolutions or the Governor's statements), start from the RBI home page
  (`https://www.rbi.org.in`, press releases section) and add a file only if you have
  RBI's permission or are otherwise sure your use is allowed. Use the same file format
  (first line `Source: <URL> (retrieved <date>)`, then the `Title:`, `Date:` and
  `Publisher:` lines with no blank line in between, then a blank line and the text), add a
  row to the table above, and rerun `scripts/build_rag.py` and `scripts/eval_rag.py`.
