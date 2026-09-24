import sys, glob, os
from PIL import Image, ImageDraw
book, start, n, cols, cw, ch, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
files = sorted(glob.glob(f"thumbs2/{book}/p-*.jpg"))[start:start+n]
rows = (len(files) + cols - 1) // cols
S = Image.new("RGB", (cols * cw, rows * (ch + 13)), "white"); d = ImageDraw.Draw(S)
for k, f in enumerate(files):
    im = Image.open(f).convert("RGB"); im.thumbnail((cw - 4, ch - 2))
    x = (k % cols) * cw; y = (k // cols) * (ch + 13)
    S.paste(im, (x + 2, y + 13)); d.text((x + 2, y + 1), str(int(os.path.basename(f)[2:-4])), fill=(200, 0, 0))
S.save(out, quality=70); print(S.size, len(files))
