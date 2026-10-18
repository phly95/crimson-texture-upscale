"""Recipe for ripple: circular water impact ripple with concentric capillary waves,
frothy foam filaments, subtle bubble texture, and soft translucent falloff.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx

def make_ripple():
    W, H = 512, 512
    orig = fx.load("sprites/ripple/orig.png")
    a_orig = orig[..., 3]

    # 1. Smooth upscaled envelope of original sprite
    orig_up = cv2.resize(orig, (W, H), interpolation=cv2.INTER_LANCZOS4)
    env_a = np.clip(orig_up[..., 3], 0, 1)

    # Coordinates
    r = fx.radius(W, H)
    theta = fx.angle(W, H)

    # 2. Tangential & radial flow vectors
    # Water ripples have foam stretched tangentially along the wave crest
    tangent_x = -np.sin(theta)
    tangent_y = np.cos(theta)
    radial_x = np.cos(theta)
    radial_y = np.sin(theta)

    # Domain warp: mix of radial expansion perturbation and tangential swirl
    warp_fbm1 = (fx.fbm(W, H, scale=40, octaves=4, seed=15) - 0.5) * 12.0
    warp_fbm2 = (fx.fbm(W, H, scale=40, octaves=4, seed=25) - 0.5) * 12.0

    warp_x = warp_fbm1 * tangent_x + warp_fbm2 * radial_x
    warp_y = warp_fbm1 * tangent_y + warp_fbm2 * radial_y
    r_warped = fx.warp(r, warp_x, warp_y)

    # 3. Fine concentric capillary wavefronts
    # Primary wave packet
    capillary1 = np.sin(r_warped * 150.0 + fx.fbm(W, H, scale=30, octaves=3, seed=33) * 3.0)
    capillary1 = (capillary1 * 0.5 + 0.5) ** 1.8

    # Secondary ultra-fine capillary ripples
    capillary2 = np.sin(r_warped * 290.0 + fx.fbm(W, H, scale=20, octaves=2, seed=44) * 2.0)
    capillary2 = (capillary2 * 0.5 + 0.5) ** 2.2

    # Capillary wave on the inner ring as well!
    inner_capillary = np.sin(r_warped * 160.0) * 0.5 + 0.5

    # 4. Foam filaments stretched tangentially along the wavefront
    # To stretch tangentially, we sample noise with anisotropic scaling
    # We can warp the coordinates along the tangent
    fbm1 = fx.fbm(W, H, scale=16, octaves=4, seed=55)
    fbm2 = fx.fbm(W, H, scale=10, octaves=3, seed=66)
    foam_filaments = (1.0 - np.abs(fbm1 - 0.5) * 2.0) ** 2.2
    foam_sub = (1.0 - np.abs(fbm2 - 0.5) * 2.0) ** 2.5

    # Micro bubble / froth beads
    fbm_bubbles = fx.fbm(W, H, scale=7, octaves=3, seed=77)
    bubbles = fx.smoothstep(0.45, 0.85, fbm_bubbles)

    # Fine specular spray droplets on crests
    fbm_fine = fx.fbm(W, H, scale=4, octaves=2, seed=88)
    spray = (fbm_fine > 0.82).astype(np.float32) * (fbm_fine - 0.82) * 5.5

    # Combine details: concentric structure + foam netting + fine bubbles
    outer_detail = (
        0.50
        + 0.28 * capillary1
        + 0.16 * capillary2
        + 0.30 * foam_filaments
        + 0.15 * foam_sub
        + 0.18 * bubbles
        + 0.12 * spray
    )

    inner_detail = 0.70 + 0.30 * inner_capillary + 0.20 * foam_sub

    # Modulate outer ring vs inner ring
    is_inner = fx.smoothstep(0.28, 0.22, r)
    detail_mod = outer_detail * (1.0 - is_inner) + inner_detail * is_inner

    # Apply detail where alpha exists, smoothly fading at faint outskirts
    presence = fx.smoothstep(0.015, 0.20, env_a)
    a_hd = env_a * ((1.0 - presence) + presence * detail_mod)

    # 5. Low-frequency calibration: ensure exact 8x8 block downsampling matches orig.png
    im_a = Image.fromarray((np.clip(a_hd, 0, 1) * 255).astype(np.uint8))
    a_down = np.asarray(im_a.resize((64, 64), Image.BOX), np.float32) / 255.0

    # Low-frequency residual error
    diff = a_orig - a_down
    diff_up = cv2.resize(diff, (W, H), interpolation=cv2.INTER_CUBIC)

    # Add residual correction masked by where original has non-zero support
    support = fx.dilate((a_orig > 0).astype(np.float32), 1)
    support_up = cv2.resize(support, (W, H), interpolation=cv2.INTER_LINEAR)

    a_calibrated = np.clip(a_hd + diff_up * support_up, 0, 1)

    strict_zero = cv2.resize((a_orig == 0).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    a_final = np.where((strict_zero > 0.5) & (env_a < 0.005), 0.0, a_calibrated)

    # 6. Shading & Color
    # Water foam highlights on crests, deeper tint in troughs
    crest_intensity = np.clip(
        capillary1 * 0.35 + capillary2 * 0.15 + foam_filaments * 0.35 + bubbles * 0.20 + spray * 0.40,
        0, 1
    )

    base_color = np.array([0.785, 0.795, 0.812], dtype=np.float32)
    highlight_color = np.array([0.88, 0.895, 0.92], dtype=np.float32)
    trough_color = np.array([0.74, 0.755, 0.78], dtype=np.float32)

    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        rgb[..., c] = base_color[c] + (highlight_color[c] - base_color[c]) * crest_intensity * 0.65 \
                                   - (base_color[c] - trough_color[c]) * (1.0 - crest_intensity) * 0.30

    layer = np.dstack([np.clip(rgb, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/ripple")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/ripple/draw.png")

if __name__ == "__main__":
    make_ripple()
