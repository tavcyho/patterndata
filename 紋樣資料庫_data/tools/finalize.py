#!/usr/bin/env python3
"""finalize.py -- merge the selections (sel.json, made with addsel.py) into books.json and write out/pcats_final.json.
   books.json    : one entry per book (file, title, kind, pages, kinds{page:kind}, long_edge, short)
   pcats_final   : page id -> category   (pilot books come from pcats_pilot.json)
Run after addsel.py, before pages.py / pack.py."""
import os, json
from pclassify import PCATS, KINDS, norm
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("PATTERN_OUT", os.path.join(HERE, "out"))
J = lambda n, d=None: json.load(open(os.path.join(HERE, n))) if os.path.exists(os.path.join(HERE, n)) else d
books = J("books.json", {})
newb, sel = J("newbooks.json", {}), J("sel.json", {})
pil = J("pcats_pilot.json", {})
for b in books.values():
    if b["kind"] == "天花板照片": b["kind"] = "照片"
cats = dict(pil)
for bid, s in sel.items():
    nb = newb[bid]
    old = books.get(bid, {})
    books[bid] = dict(file=nb["file"], title=nb["title"], kind=nb["kind"], pages=[[p, p] for p in s["pages"]],
                      kinds={k: v for k, v in s.get("kinds", {}).items() if v != nb["kind"]},
                      long_edge=old.get("long_edge", 2400), cfg=old.get("cfg", {}), short=nb["short"])
    for p in s["pages"]:
        c = norm(s["cats"].get(str(p)) or s["default"])
        assert c in PCATS, (bid, p, c)
        cats[f"{bid}_p{p:03d}"] = c
    for k in s.get("kinds", {}).values(): assert k in KINDS, k
json.dump(books, open(os.path.join(HERE, "books.json"), "w"), ensure_ascii=False, indent=1)
os.makedirs(OUT, exist_ok=True)
json.dump(cats, open(os.path.join(OUT, "pcats_final.json"), "w"), ensure_ascii=False)
print(len(books), "books;", sum(len(b["pages"]) for b in books.values()), "pages;", len(cats), "categorised")
