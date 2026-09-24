#!/usr/bin/env python3
"""addsel.py <book> <pages> [default_cat] [cat=pages;cat=pages] [kind=pages;...]
pages like "11,13,16-23"  -> appended to sel.json"""
import sys, json, os
def parse(s):
    out=[]
    for t in s.split(","):
        t=t.strip()
        if not t: continue
        if "-" in t:
            a,b=t.split("-"); out+=list(range(int(a),int(b)+1))
        else: out.append(int(t))
    return out
book,pages=sys.argv[1],parse(sys.argv[2])
default=sys.argv[3] if len(sys.argv)>3 else ""
cats={}
if len(sys.argv)>4 and sys.argv[4]:
    for part in sys.argv[4].split(";"):
        c,p=part.split("="); 
        for x in parse(p): cats[x]=c
kinds={}
if len(sys.argv)>5 and sys.argv[5]:
    for part in sys.argv[5].split(";"):
        c,p=part.split("=")
        for x in parse(p): kinds[x]=c
fn="sel.json"
d=json.load(open(fn)) if os.path.exists(fn) else {}
d[book]=dict(pages=sorted(set(pages)),default=default,cats={str(k):v for k,v in cats.items()},kinds={str(k):v for k,v in kinds.items()})
json.dump(d,open(fn,"w"),ensure_ascii=False,indent=0)
print(book,len(set(pages)),"pages; overrides",len(cats),len(kinds))
