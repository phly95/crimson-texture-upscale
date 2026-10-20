"""headtex_portrait.py TARGET SEED... : generate a normal front portrait (klein from a plain grey input, prompt only) -> headtex/portrait_<tg>_s<seed>.png"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_fin as F
T = F.T; D = F.H.ROOT / "face_audit/headtex"
PROMPTS = {
 "woman": "A studio portrait photograph, perfectly front-facing and symmetrical, of a smiling young woman with dark brown skin and a warm natural smile showing her teeth, both eyes open and level, wearing a snug, smooth, plain dark brown leather skull cap that hugs the whole head, with no visor, no brim and no ear flaps, ending in one clean smooth line across the forehead, ears covered by the cap, bare neck and shoulders, plain mid-grey background, even flat lighting, sharp focus, photorealistic.",
 "fbhead1": "A studio portrait photograph, perfectly front-facing and symmetrical, of a smiling elderly man with round wire glasses and a cheerful gentle smile, wearing a thick white knitted woolly cap that ends in one clean line across the forehead, a tweed collar and neck, plain mid-grey background, even flat lighting, sharp focus, photorealistic.",
}
if __name__ == "__main__":
    tg = sys.argv[1]; g = D / "grey_1024.png"; Image.fromarray(np.full((1024, 1024, 3), 128, np.uint8)).save(g)
    for s in map(int, sys.argv[2:]):
        out = D / f"portrait_{tg}_s{s}.png"; T.locked_gen_wh(str(g), str(out), PROMPTS[tg], 1024, 1024, s); print(out.name, flush=True)
