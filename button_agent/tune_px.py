import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
from scipy import ndimage as ndi
from skimage import color
import fx

def over(im, bg):
    base = Image.new("RGBA", im.size, bg)
    base.alpha_composite(im)
    return base.convert("RGB")

o = Image.open("sprites/PX_B_SELLPLANES/orig.png")
tw, th = 1000, 640
sh = 160
grey = (128, 128, 128, 255)
lo = np.asarray(over(o, grey), np.float32) / 255
lab_o = color.rgb2lab(lo)

ao = np.asarray(o.getchannel("A"), np.float32) / 255.0
alpha_5x = cv2.resize(ao, (tw, th), interpolation=cv2.INTER_LANCZOS4)
alpha_5x = np.clip(alpha_5x, 0, 1)

# Precompute distance transform for each state
solid = alpha_5x > 0.5
dist_sheets = []
for s in range(4):
    st_pad = np.pad(solid[s*sh:(s+1)*sh], ((1, 1), (0, 0)), mode="constant", constant_values=False)
    d = ndi.distance_transform_edt(st_pad)[1:-1]
    dist_sheets.append(d)
dist_5x = np.vstack(dist_sheets)

# Precompute noise and vertical lighting
noise = fx.fbm(tw, sh, scale=40, octaves=4, seed=123, tile=True)
mottle = (noise - 0.5) * 0.15
y_coord = np.linspace(0, 1, sh)[:, None]
vert_light = 1.0 + 0.32 * np.exp(-((y_coord - 0.28) ** 2) / 0.06) - 0.30 * (y_coord ** 1.3)

# Precompute text
font = ImageFont.truetype("fonts/BarlowCondensed-Bold.ttf", 84)
text = "SELL PLANES"
tracking = 5
advances = [font.getlength(ch) for ch in text]
total_w = sum(advances) + (len(text) - 1) * tracking
cx, cy = tw / 2.0, sh * 0.46
bbox_H = font.getbbox("H")
cap_h = bbox_H[3] - bbox_H[1]
base_y = cy - cap_h / 2.0 - bbox_H[1]

text_im = Image.new("L", (tw, sh), 0)
d = ImageDraw.Draw(text_im)
cur_x = cx - total_w / 2.0
for ch, adv in zip(text, advances):
    d.text((cur_x, base_y), ch, font=font, fill=255)
    cur_x += adv + tracking
t_alpha = np.asarray(text_im, np.float32) / 255.0

shadow_alpha = fx.blur(t_alpha, 3.5)
M_shadow = np.float32([[1, 0, 5], [0, 1, 7]])
shadow_alpha = cv2.warpAffine(shadow_alpha, M_shadow, (tw, sh), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
glow_alpha = fx.blur(t_alpha, 4.5)
M_emboss = np.float32([[1, 0, -2], [0, 1, -2]])
emboss_alpha = np.clip(cv2.warpAffine(t_alpha, M_emboss, (tw, sh), borderMode=cv2.BORDER_CONSTANT, borderValue=0) - t_alpha, 0, 1)

def eval_params(bp0, bp1, st_dark, st_body, st_hl):
    p0 = bp0 * (vert_light[:, :, None] + mottle[:, :, None])
    p1 = bp1 * (vert_light[:, :, None] + mottle[:, :, None])

    sheets = []
    for s in range(4):
        dist = dist_sheets[s]
        rgb = (p0 if s == 0 else p1).copy()

        # Border
        rgb[dist <= 4] = st_dark
        # Tube
        m_tube = (dist > 4) & (dist <= 16)
        tube_t = (dist[m_tube] - 4) / 12.0
        tube_hl = np.sin(tube_t * np.pi) ** 1.3
        tube_rgb = st_body[None, :] + (st_hl - st_body)[None, :] * tube_hl[:, None]
        rgb[m_tube] = tube_rgb
        # Crease
        m_crease = (dist > 16) & (dist <= 22)
        crease_t = (dist[m_crease] - 16) / 6.0
        crease_factor = np.sin(crease_t * np.pi)
        rgb[m_crease] = rgb[m_crease] * (1.0 - crease_factor[:, None] * 0.75) + st_dark[None, :] * (crease_factor[:, None] * 0.75)
        # Bevel
        m_bevel = (dist > 22) & (dist <= 35)
        bevel_t = (dist[m_bevel] - 22) / 13.0
        shadow_factor = (1.0 - bevel_t) * 0.35
        rgb[m_bevel] = rgb[m_bevel] * (1.0 - shadow_factor[:, None])

        # Lettering
        if s == 0:
            text_rgb = np.array([194.0, 194.0, 194.0], np.float32)
            shadow_rgb = np.array([30.0, 28.0, 26.0], np.float32)
            hl_rgb = np.array([225.0, 225.0, 225.0], np.float32)
            rgb = rgb * (1.0 - shadow_alpha[..., None] * 0.8) + shadow_rgb * (shadow_alpha[..., None] * 0.8)
            rgb = np.minimum(rgb + hl_rgb * (emboss_alpha[..., None] * 0.35), 255.0)
            rgb = rgb * (1.0 - t_alpha[..., None]) + text_rgb * t_alpha[..., None]
        elif s == 1:
            text_rgb = np.array([254.0, 226.0, 157.0], np.float32)
            glow_rgb = np.array([215.0, 170.0, 95.0], np.float32)
            shadow_rgb = np.array([20.0, 15.0, 8.0], np.float32)
            rgb = rgb * (1.0 - shadow_alpha[..., None] * 0.85) + shadow_rgb * (shadow_alpha[..., None] * 0.85)
            rgb = rgb * (1.0 - glow_alpha[..., None] * 0.45) + glow_rgb * (glow_alpha[..., None] * 0.45)
            rgb = rgb * (1.0 - t_alpha[..., None]) + text_rgb * t_alpha[..., None]
        elif s == 2:
            text_rgb = np.array([255.0, 176.0, 92.0], np.float32)
            glow_rgb = np.array([220.0, 120.0, 40.0], np.float32)
            shadow_rgb = np.array([15.0, 5.0, 2.0], np.float32)
            rgb = rgb * (1.0 - shadow_alpha[..., None] * 0.85) + shadow_rgb * (shadow_alpha[..., None] * 0.85)
            rgb = rgb * (1.0 - glow_alpha[..., None] * 0.5) + glow_rgb * (glow_alpha[..., None] * 0.5)
            rgb = rgb * (1.0 - t_alpha[..., None]) + text_rgb * t_alpha[..., None]
        elif s == 3:
            text_rgb = np.array([249.0, 128.0, 90.0], np.float32)
            glow_rgb = np.array([200.0, 55.0, 30.0], np.float32)
            shadow_rgb = np.array([12.0, 2.0, 2.0], np.float32)
            rgb = rgb * (1.0 - shadow_alpha[..., None] * 0.85) + shadow_rgb * (shadow_alpha[..., None] * 0.85)
            rgb = rgb * (1.0 - glow_alpha[..., None] * 0.5) + glow_rgb * (glow_alpha[..., None] * 0.5)
            rgb = rgb * (1.0 - t_alpha[..., None]) + text_rgb * t_alpha[..., None]

        layer = np.dstack([np.clip(rgb / 255.0, 0, 1), alpha_5x[s*sh:(s+1)*sh]])
        sheets.append(layer)

    full = np.vstack(sheets)
    full[0] = full[-1]

    # Downsample and compute dE
    im_full = Image.fromarray((full * 255.0 + 0.5).astype(np.uint8), "RGBA")
    c_small = im_full.resize((200, 128), Image.BOX)
    lc = np.asarray(over(c_small, grey), np.float32) / 255
    lab_c = color.rgb2lab(lc)
    de = color.deltaE_cie76(lab_o, lab_c)
    return de.mean(), full

# Test default
de, full = eval_params(
    np.array([118.0, 108.0, 80.0], np.float32),
    np.array([135.0, 112.0, 62.0], np.float32),
    np.array([15.0, 15.0, 14.0], np.float32),
    np.array([75.0, 72.0, 70.0], np.float32),
    np.array([125.0, 120.0, 115.0], np.float32)
)
print("Initial test de_mean:", de)
