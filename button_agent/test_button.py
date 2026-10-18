import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
from scipy import ndimage as ndi
import fx

def make_plate(W, H, sh, alpha_5x):
    """
    Generate the 4-state plate background (W, H, 4).
    Each state has height sh.
    State 0: Grey parchment plate + tubular steel frame.
    State 1, 2, 3: Brass-tan parchment plate + tubular steel frame.
    """
    noise = fx.fbm(W, sh, scale=40, octaves=4, seed=123, tile=True)
    mottle = (noise - 0.5) * 0.16 # subtle ±8% parchment mottling

    y_coord = np.linspace(0, 1, sh)[:, None]

    # Vertical lighting gradient on the parchment plate:
    vert_light = 1.0 + 0.30 * np.exp(-((y_coord - 0.28) ** 2) / 0.06) - 0.28 * (y_coord ** 1.3)

    base_plate_0 = np.array([120.0, 110.0, 92.0], dtype=np.float32)
    base_plate_1 = np.array([136.0, 115.0, 76.0], dtype=np.float32)

    p0 = base_plate_0 * (vert_light[:, :, None] + mottle[:, :, None])
    p1 = base_plate_1 * (vert_light[:, :, None] + mottle[:, :, None])

    steel_dark = np.array([18.0, 18.0, 17.0], dtype=np.float32)
    steel_body = np.array([95.0, 92.0, 90.0], dtype=np.float32)
    steel_hl   = np.array([140.0, 136.0, 134.0], dtype=np.float32)
    crease     = np.array([25.0, 22.0, 18.0], dtype=np.float32)

    sheets = []
    for s in range(4):
        state_alpha = alpha_5x[s*sh:(s+1)*sh]
        solid = state_alpha > 0.5

        # Pad in y to ensure top and bottom boundaries are recognized as outer edges
        st_pad = np.pad(solid, ((1, 1), (0, 0)), mode="constant", constant_values=False)
        dist = ndi.distance_transform_edt(st_pad)[1:-1]

        p_base = p0 if s == 0 else p1
        rgb = p_base.copy()

        # Outer dark border (dist <= 4)
        m_border = dist <= 4
        rgb[m_border] = steel_dark

        # Tubular frame body (4 < dist <= 16)
        m_tube = (dist > 4) & (dist <= 16)
        tube_t = (dist[m_tube] - 4) / 12.0
        tube_hl = np.sin(tube_t * np.pi) ** 1.3
        tube_rgb = steel_body[None, :] + (steel_hl - steel_body)[None, :] * tube_hl[:, None]
        rim_noise = fx.fbm(W, sh, scale=12, octaves=3, seed=777, tile=True)[m_tube, None]
        tube_rgb += (rim_noise - 0.5) * 10.0
        rgb[m_tube] = tube_rgb

        # Inner crease (16 < dist <= 22)
        m_crease = (dist > 16) & (dist <= 22)
        crease_t = (dist[m_crease] - 16) / 6.0
        crease_factor = np.sin(crease_t * np.pi)
        rgb[m_crease] = rgb[m_crease] * (1.0 - crease_factor[:, None] * 0.75) + crease[None, :] * (crease_factor[:, None] * 0.75)

        # Inner bevel shadow into parchment (22 < dist <= 35)
        m_bevel = (dist > 22) & (dist <= 35)
        bevel_t = (dist[m_bevel] - 22) / 13.0
        shadow_factor = (1.0 - bevel_t) * 0.30
        rgb[m_bevel] = rgb[m_bevel] * (1.0 - shadow_factor[:, None])

        state_layer = np.dstack([np.clip(rgb / 255.0, 0, 1), state_alpha])
        sheets.append(state_layer)

    return np.vstack(sheets)

def render_button(label, W, H, orig_alpha_path):
    sh = H // 4
    scale = W // (W // 5) # 5

    o = Image.open(orig_alpha_path)
    ao = np.asarray(o.getchannel("A"), np.float32) / 255.0
    alpha_5x = cv2.resize(ao, (W, H), interpolation=cv2.INTER_LANCZOS4)
    alpha_5x = np.clip(alpha_5x, 0, 1)

    # Plate
    plate = make_plate(W, H, sh, alpha_5x)

    if not label:
        # Blank button
        plate[0] = plate[-1]
        return plate

    # Lettering font
    font_path = "fonts/BarlowCondensed-Bold.ttf"
    # Cap height is ~37.5% of state_h
    target_cap_h = sh * 0.38
    # Probe font size
    probe_font = ImageFont.truetype(font_path, 100)
    bbox_H = probe_font.getbbox("H")
    fsize = int(round(100.0 * (target_cap_h / (bbox_H[3] - bbox_H[1]))))
    font = ImageFont.truetype(font_path, fsize)

    tracking = max(1, int(round(sh * 0.03))) # ~5px at sh=160
    advances = [font.getlength(ch) for ch in label]
    total_w = sum(advances) + (len(label) - 1) * tracking

    cx = W / 2.0
    cy = sh * 0.46 # centered in the parchment area (slightly above middle because of bottom rim/bar)

    bbox_H = font.getbbox("H")
    cap_h = bbox_H[3] - bbox_H[1]
    base_y = cy - cap_h / 2.0 - bbox_H[1]

    text_im = Image.new("L", (W, sh), 0)
    d = ImageDraw.Draw(text_im)
    cur_x = cx - total_w / 2.0
    for ch, adv in zip(label, advances):
        d.text((cur_x, base_y), ch, font=font, fill=255)
        cur_x += adv + tracking

    t_alpha = np.asarray(text_im, np.float32) / 255.0

    # Shadow and glow
    shadow_sigma = max(1.0, sh * 0.022) # ~3.5px
    glow_sigma = max(1.5, sh * 0.028)   # ~4.5px

    shadow_alpha = fx.blur(t_alpha, shadow_sigma)
    dx = int(round(sh * 0.03)) # ~5px
    dy = int(round(sh * 0.045)) # ~7px
    M_shadow = np.float32([[1, 0, dx], [0, 1, dy]])
    shadow_alpha = cv2.warpAffine(shadow_alpha, M_shadow, (W, sh), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    glow_alpha = fx.blur(t_alpha, glow_sigma)

    M_emboss = np.float32([[1, 0, -2], [0, 1, -2]])
    emboss_alpha = np.clip(cv2.warpAffine(t_alpha, M_emboss, (W, sh), borderMode=cv2.BORDER_CONSTANT, borderValue=0) - t_alpha, 0, 1)

    out_states = []
    for s in range(4):
        st_plate = plate[s*sh:(s+1)*sh].copy()

        if s == 0:
            text_rgb = np.array([194.0, 194.0, 194.0], np.float32) / 255.0
            shadow_rgb = np.array([40.0, 38.0, 36.0], np.float32) / 255.0
            hl_rgb = np.array([225.0, 225.0, 225.0], np.float32) / 255.0

            st_plate[..., :3] = st_plate[..., :3] * (1.0 - shadow_alpha[..., None] * 0.7) + shadow_rgb * (shadow_alpha[..., None] * 0.7)
            st_plate[..., :3] = np.minimum(st_plate[..., :3] + hl_rgb * (emboss_alpha[..., None] * 0.35), 1.0)
            st_plate[..., :3] = st_plate[..., :3] * (1.0 - t_alpha[..., None]) + text_rgb * t_alpha[..., None]

        elif s == 1:
            text_rgb = np.array([254.0, 226.0, 157.0], np.float32) / 255.0
            glow_rgb = np.array([215.0, 170.0, 95.0], np.float32) / 255.0
            shadow_rgb = np.array([25.0, 18.0, 10.0], np.float32) / 255.0

            st_plate[..., :3] = st_plate[..., :3] * (1.0 - shadow_alpha[..., None] * 0.75) + shadow_rgb * (shadow_alpha[..., None] * 0.75)
            st_plate[..., :3] = st_plate[..., :3] * (1.0 - glow_alpha[..., None] * 0.45) + glow_rgb * (glow_alpha[..., None] * 0.45)
            st_plate[..., :3] = st_plate[..., :3] * (1.0 - t_alpha[..., None]) + text_rgb * t_alpha[..., None]

        elif s == 2:
            text_rgb = np.array([255.0, 176.0, 92.0], np.float32) / 255.0
            glow_rgb = np.array([220.0, 120.0, 40.0], np.float32) / 255.0
            shadow_rgb = np.array([20.0, 5.0, 2.0], np.float32) / 255.0

            st_plate[..., :3] = st_plate[..., :3] * (1.0 - shadow_alpha[..., None] * 0.8) + shadow_rgb * (shadow_alpha[..., None] * 0.8)
            st_plate[..., :3] = st_plate[..., :3] * (1.0 - glow_alpha[..., None] * 0.5) + glow_rgb * (glow_alpha[..., None] * 0.5)
            st_plate[..., :3] = st_plate[..., :3] * (1.0 - t_alpha[..., None]) + text_rgb * t_alpha[..., None]

        elif s == 3:
            text_rgb = np.array([249.0, 128.0, 90.0], np.float32) / 255.0
            glow_rgb = np.array([200.0, 55.0, 30.0], np.float32) / 255.0
            shadow_rgb = np.array([15.0, 2.0, 2.0], np.float32) / 255.0

            st_plate[..., :3] = st_plate[..., :3] * (1.0 - shadow_alpha[..., None] * 0.8) + shadow_rgb * (shadow_alpha[..., None] * 0.8)
            st_plate[..., :3] = st_plate[..., :3] * (1.0 - glow_alpha[..., None] * 0.5) + glow_rgb * (glow_alpha[..., None] * 0.5)
            st_plate[..., :3] = st_plate[..., :3] * (1.0 - t_alpha[..., None]) + text_rgb * t_alpha[..., None]

        out_states.append(st_plate)

    full = np.vstack(out_states)
    # Ensure wrap seam <= 1.5
    full[0] = full[-1]
    return full
