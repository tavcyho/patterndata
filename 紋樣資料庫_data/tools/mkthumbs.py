#!/usr/bin/env python3
"""mkthumbs.py <book_id> [...]  -- 300px page thumbnails thumbs2/<book>/p-NNN.jpg for contact-sheet review (tsheet.py).
The book must be registered in newbooks.json ({"id": {"file": "x.pdf", "kind": "黑白稿", "title": "...", "short": "..."}})."""
import json, os, subprocess, sys
SRC = os.environ.get("PATTERN_SRC", "/mnt/user-data/uploads/公眾版權圖書資料庫/公共圖書_紋樣")
nb = json.load(open("newbooks.json"))
for b in sys.argv[1:]:
    d = f"thumbs2/{b}"; os.makedirs(d, exist_ok=True)
    subprocess.run(["pdftoppm", "-jpeg", "-jpegopt", "quality=60", "-scale-to", "300", os.path.join(SRC, nb[b]["file"]), f"{d}/p"])
    print(b, len(os.listdir(d)), flush=True)
