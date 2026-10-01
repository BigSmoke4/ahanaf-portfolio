# Regenerates ../assets.js from the art in this folder (run after editing the art).
# 1) python3 make_eyes.py   -> head_clean.png (head without pupils), pupilA/B.png, maskA/B.png, eyes.json
# 2) python3 build_assets.py
# The images are embedded as data URIs so WebGL can use them even when index.html is opened by double-click.
from PIL import Image
import base64, io, os, json
here = os.path.dirname(os.path.abspath(__file__))
def b64(img, fmt, **kw):
    buf = io.BytesIO(); img.save(buf, fmt, **kw)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()
def mask_rgba(path):                      # greyscale mask -> white image whose alpha is the mask
    m = Image.open(path).convert('L'); w = Image.new('RGBA', m.size, (255, 255, 255, 255)); w.putalpha(m); return w
head = Image.open(os.path.join(here, 'head_clean.png')).convert('RGBA').crop((204, 25, 1107, 699))
lidA = Image.open(os.path.join(here, 'lidA.png')).convert('RGBA')
lidB = Image.open(os.path.join(here, 'lidB.png')).convert('RGBA')
eyes = json.load(open(os.path.join(here, 'eyes.json')))
a = {
  'head': b64(head, 'WEBP', quality=93, alpha_quality=100, method=5),
  'lidA': b64(lidA, 'PNG'), 'lidB': b64(lidB, 'PNG'),
  'pupilA': b64(Image.open(os.path.join(here, 'pupilA.png')).convert('RGBA'), 'PNG'),
  'pupilB': b64(Image.open(os.path.join(here, 'pupilB.png')).convert('RGBA'), 'PNG'),
  'maskA': b64(mask_rgba(os.path.join(here, 'maskA.png')), 'PNG'),
  'maskB': b64(mask_rgba(os.path.join(here, 'maskB.png')), 'PNG'),
  'eyes': {k: {'cx': round(v['cx']), 'cy': round(v['cy']), 'm': v['m']} for k, v in eyes.items()},
}
js = "window.CHAR_ASSETS = " + json.dumps(a, indent=1) + ";\n"
open(os.path.join(here, '..', 'assets.js'), 'w').write(js)
print('assets.js written', len(js) // 1024, 'KB')
