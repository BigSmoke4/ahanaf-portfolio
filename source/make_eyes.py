# Splits his own pupils out of head.png so they can move independently.
# Writes: head_clean.png (head with pupils filled by eye-white), pupilA/B.png (his pupil, completed to a full disc),
#         maskA/B.png (eye opening, used to clip the pupil), eyes.json (geometry in art coordinates).
import numpy as np, cv2, json, os
from PIL import Image
here = os.path.dirname(os.path.abspath(__file__))
head = np.array(Image.open(os.path.join(here, 'head.png')).convert('RGBA'))
BOXES = {'A': (591, 312, 192, 112), 'B': (770, 297, 173, 94)}     # same boxes as the eyelids
SHIFT_DX = {'A': -38, 'B': 42}                                      # where to borrow clean eye-white + lid from (along the lid line)
disc = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
clean = head.copy(); meta = {}

for k, (x, y, w, h) in BOXES.items():
    c = head[y:y+h, x:x+w].astype(np.float32)
    R, G, B, A = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    L = 0.3*R + 0.59*G + 0.11*B
    scl = ((A > 200) & (B >= R - 10) & (L > 110)).astype(np.uint8)
    dark = ((A > 200) & (L < 75)).astype(np.uint8)

    # his pupil = the dark blob that survives removing the thin eyelid line
    core = cv2.morphologyEx(dark, cv2.MORPH_OPEN, disc(9))
    n, lab, st, cen = cv2.connectedComponentsWithStats(core)
    i = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]); core = (lab == i).astype(np.uint8)
    blob = dark & cv2.dilate(core, disc(3))
    ys, xs = np.nonzero(blob); x0, x1 = xs.min(), xs.max()

    # eyelid lower edge across the pupil, interpolated from the columns either side of it
    def lid_edge(col):
        s = np.nonzero(scl[:, col])[0]; return s.min() if len(s) else None
    L_cols = [(cx, lid_edge(cx)) for cx in range(max(0, x0-14), x0-3)]
    R_cols = [(cx, lid_edge(cx)) for cx in range(x1+4, min(w, x1+15))]
    pts = np.array([(cx, e) for cx, e in L_cols + R_cols if e is not None], np.float32)
    a, b = np.polyfit(pts[:, 0], pts[:, 1], 1)
    yy, xx = np.mgrid[0:h, 0:w]
    below_lid = yy >= (a*xx + b) - 0.5
    sprite_blob = blob.astype(bool) & below_lid
    hole = cv2.dilate(blob, disc(3)).astype(bool) & (A > 200)
    hole &= (yy >= (a*xx + b) - 7)                                   # reaches up through the lid band
    hole &= (xx >= x0 - 3) & (xx <= x1 + 3)
    hole = cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_CLOSE, disc(2)).astype(bool)

    # borrow eye-white AND eyelid from beside the pupil (same distance from the lid), then match the local light level
    dx = SHIFT_DX[k]; dy = int(round(a * dx))
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])                        # the pixel at (x+dx, y+dy) lands on (x, y)
    patch = cv2.warpAffine(c, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    known = (scl.astype(bool) & ~cv2.dilate(hole.astype(np.uint8), disc(4)).astype(bool)).astype(np.float32)
    def lowfreq(img, wgt, sg=7):
        num = cv2.GaussianBlur(img * wgt[..., None], (0, 0), sg); den = cv2.GaussianBlur(wgt, (0, 0), sg)[..., None]
        return num / np.maximum(den, 1e-3)
    E = lowfreq(c[..., :3], known)
    pk = cv2.warpAffine(known, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    Pl = lowfreq(patch[..., :3], pk)
    corr = (E - Pl) * below_lid[..., None]                            # no correction inside the lid band
    patch_adj = patch.copy(); patch_adj[..., :3] += corr
    soft = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 1.0)
    soft = np.where(hole, 1.0, soft)[..., None] * (c[..., 3:4] > 200)
    out = c * (1 - soft) + patch_adj * soft; out[..., 3] = c[..., 3]
    clean[y:y+h, x:x+w] = np.clip(out, 0, 255).astype(np.uint8)

    # pupil sprite: alpha from darkness vs. the eye-white, completed to a full disc by mirroring his own pixels
    ys, xs = np.nonzero(blob & below_lid)
    pts2 = np.column_stack([xs, ys]).astype(np.float32)
    edge = blob & below_lid & ~cv2.erode((blob & below_lid).astype(np.uint8), disc(1)).astype(bool)
    ey, ex = np.nonzero(edge & (yy > (a*xx + b) + 3))                    # visible arc only (not the straight cut along the lid)
    A_ = np.column_stack([ex, ey, np.ones(len(ex))]); bb = ex**2 + ey**2
    sol = np.linalg.lstsq(A_, bb, rcond=None)[0]
    cx_, cy_ = sol[0]/2, sol[1]/2; r = float(np.sqrt(sol[2] + cx_**2 + cy_**2))
    m = int(np.ceil(r)) + 3
    S = 2*m + 1
    sprite = np.zeros((S, S, 4), np.float32)
    pc = c[..., :3]
    ref_white = np.median(c[hole][:, :3], axis=0) if False else np.array([200, 204, 210], np.float32)
    # darkest colour of his pupil
    pup_col = c[core.astype(bool)][:, :3].mean(0)
    for j in range(S):
        for i2 in range(S):
            sx, sy = int(round(cx_ - m + i2)), int(round(cy_ - m + j))
            d = np.hypot(i2 - m, j - m)
            if d > r + 1.5: continue
            if 0 <= sx < w and 0 <= sy < h and blob[sy, sx] and below_lid[sy, sx]:
                sprite[j, i2, :3] = c[sy, sx, :3]; sprite[j, i2, 3] = 255
    vis = sprite[..., 3] > 0
    # hidden upper part: mirror his own visible pixels about the pupil's horizontal axis
    for j in range(S):
        for i2 in range(S):
            if vis[j, i2]: continue
            d = np.hypot(i2 - m, j - m)
            if d <= r + 0.5:
                mj = 2*m - j
                if 0 <= mj < S and vis[mj, i2]: sprite[j, i2] = sprite[mj, i2]
                else: sprite[j, i2, :3] = pup_col; sprite[j, i2, 3] = 255
    # soft round rim
    yy2, xx2 = np.mgrid[0:S, 0:S]; d = np.hypot(xx2 - m, yy2 - m)
    rim = np.clip(r + 0.5 - d, 0, 1)
    sprite[..., 3] = np.maximum(sprite[..., 3] * 0, 255 * rim)
    Image.fromarray(sprite.astype(np.uint8), 'RGBA').save(os.path.join(here, f'pupil{k}.png'))

    # eye opening (clip mask): eye-white + the original pupil area, closed up and slightly grown
    op = (scl | blob).astype(np.uint8) & below_lid.astype(np.uint8)
    op = cv2.morphologyEx(op, cv2.MORPH_CLOSE, disc(3))
    n, lab = cv2.connectedComponents(op)
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    op = (lab == big).astype(np.uint8)
    op = cv2.dilate(op, disc(1))
    cnts, _ = cv2.findContours(op, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    op = np.zeros_like(op); cv2.drawContours(op, cnts, -1, 1, -1)       # fill pinholes (fur flecks inside the eye)
    # let the mask reach up to the lid line so the pupil tucks under it
    top_ext = cv2.dilate(op, np.ones((5, 1), np.uint8), anchor=(0, 4)) & below_lid.astype(np.uint8)
    op = np.maximum(op, top_ext)
    mk = cv2.GaussianBlur(op.astype(np.float32), (0, 0), 0.8)
    Image.fromarray((mk * 255).astype(np.uint8), 'L').save(os.path.join(here, f'mask{k}.png'))
    meta[k] = dict(x=x, y=y, w=w, h=h, cx=float(cx_), cy=float(cy_), r=r, sprite=S, m=m)
    print(k, 'pupil centre (box px)', round(cx_, 1), round(cy_, 1), 'r', round(r, 1), 'lid line', round(a, 3), round(b, 1))

Image.fromarray(clean).save(os.path.join(here, 'head_clean.png'))
json.dump(meta, open(os.path.join(here, 'eyes.json'), 'w'), indent=1)
