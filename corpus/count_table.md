| Step | Reason | Methods | Commit messages | Review threads | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| Collected |  | 584,610 | 403,190 | 95,432 | 1,083,232 |
| Duplicate repositories | same project listed twice by GHS (old and new name) | −16,076 | −19,592 | −8,034 | −43,702 |
| Exact duplicates | identical token sequence | −69,759 | −54,329 | −149 | −124,237 |
| Near duplicates | Jaccard >= 0.8 on token 3-grams | −19,722 | −9,153 | −1,876 | −30,751 |
| **Deduplicated** |  | **479,053** | **320,116** | **85,373** | **884,542** |
| Filter: parse errors | method does not parse cleanly | −277 | 0 | 0 | −277 |
| Filter: generated | generated file (path, header, generator tool or @Generated) | −14,056 | 0 | 0 | −14,056 |
| Filter: trivial without Javadoc | getter, setter, empty body or one-call delegation, with no Javadoc | −60,528 | 0 | 0 | −60,528 |
| Filter: AI-assisted | commit marked as written with an AI assistant | 0 | −2,970 | 0 | −2,970 |
| Filter: bots | commit by a bot account | 0 | −14,985 | 0 | −14,985 |
| Filter: revert/merge | Revert "..." and Merge branch/pull request messages | 0 | −1,190 | 0 | −1,190 |
| Filter: bots (whole thread) | thread with no human comment once bot comments are removed | 0 | 0 | −9,676 | −9,676 |
| Filter: per-hunk cap | more than 10 threads on the same diff hunk (earliest kept) | 0 | 0 | −516 | −516 |
| Filter: non-English | 2+ letters from a non-Latin script, outside code and URLs | 0 | −13,283 | −1,622 | −14,905 |
| **Filtered** |  | **404,192** | **287,688** | **73,559** | **765,439** |
| Decontamination | contains a validation/test method, revised method, or comment of >= 4 words | −46 | −11 | −265 | −322 |
| **Decontaminated** |  | **404,146** | **287,677** | **73,294** | **765,117** |
| Empty message | commit with no message text | 0 | −1 | 0 | −1 |
| Under 10 tokens | assignment rule: 10 to 512 tokens under our tokenizer | −754 | −137,388 | −2 | −138,144 |
| Over 512 tokens | assignment rule: 10 to 512 tokens under our tokenizer | −6,650 | −364 | −14,408 | −21,422 |
| **Final** |  | **396,742** | **149,924** | **58,884** | **605,550** |
