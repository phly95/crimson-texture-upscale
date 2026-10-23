"""tile_mags.py [NAME ...]: tiled FLUX 4x of the eight scrapbook magazine covers, one tile_job per cover (tile_job/mag_<STEM>_4x.png), unattended.
Each cover's global hint quotes its printed lettering verbatim (per-tile prompts beat one generic prompt, and a cover is one or two s2 tiles). The tiny body text of the
covers is not quoted: it is set afterwards from the game's own strings (typeset_page), as are the headline words the redraw still garbles. Nothing is submitted here."""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tile_batch as tb

SB = Path.home() / "Games/CSVM/extracted/rof/ASSETS/GRAPHICS/SCRAPBOOK"
LEAD = tb.NEUTRAL + "It shows a vintage printed magazine cover, tilted, with worn paper edges. Keep every printed word exactly as it is in the crop, with the same letters, size, colour and place; "
TAIL = " Do not add any lettering, logos or creases that are not already in the crop, and do not change anyone's face, pose or clothing."
COVERS = {
    "SB_00_00_MAG1": 'the lettering of this cover reads: "North America\'s #1 Publication Covering Topics Important to Aviators and Aviation." / "AIR ACTION WEEKLY MAGAZINE" / "No. 13" / "25¢" / "March 17, 1939" / "Exclusive! Interview with NATHAN ZACHARY SEE PAGE 3" / "Secret Photos! Inside a PIRATE Zeppelin SEE PAGE 6" / "Nathan Zachary: Gentleman Pirate!".',
    "SB_01_02_MAG2": 'the lettering of this cover reads: "VOLUME LII" / "NUMBER TWO" / "Global Traveler\'s Quarterly" / "Hawaiian Paradise\'s Ancient Mystery .......... With 54 Ills." / "JUNE" / "1937" / "$3.50 A YEAR" / "50¢ THE COPY".',
    "SB_11_01_MAG3": 'the lettering of this cover reads: "HOLLYWOOD Confidential" / "13¢" / "May 22, 1937" / "Air Pirate Nathan Zachary Wows Audiences!" / "He dodged! He swooped! It was spectacular ..." and a small column of tiny body text on the left.',
    "SB_11_01_MAG4": 'the lettering of this cover reads: "HOLLYWOOD Confidential" / "May 22, 1937" / "Air Pirate Nathan Zachary Ruins Movie" / "Rockets and bullets were flying everywhere, planes crashing into sets ... it was a disaster." and a small column of tiny body text on the left.',
    "SB_14_01_MAG5": 'the lettering of this cover reads: "Daily Variety" / "AUGUST 1937" / "10¢" / "Cooper" / "Opens New Studio" and a small column of tiny body text on the left.',
    "SB_21_01_MAG7": 'the lettering of this cover reads: "10¢" / "New York Weekly" / "THE NEWS YOU CAN USE." / "POLICE HARASS CITIZENS!" / "November 11, 1937" printed vertically in red, and a small column of tiny body text.',
    "SB_19_01_ZPJCOVER": 'the lettering of this cover reads: "Zeppelin" / "Pilot\'s Journal" / "10¢" / "August 2, 1937" / "New Zeppelin Battle Tactics by Eugene \\"Sparks\\" Rasmussen" / "The Empire State-Texas Zeppelin Race Heats Up!" / "Imperial Air Opens New South America Route" / handwritten in red "Made the big time, Sparks! Zachary".',
    "SB_22_01_COMICOVER": 'the lettering of this comic cover reads: "NATHAN ZACHARY" / "Fortune Hunter!" / "3¢" / "GUTS OF STEEL, NERVES OF STEEL, HEART OF GOLD" / "NATHAN ZACHARY TAKES ON THE BLACK HATS and THE BLACK SWAN. WHO GETS WHO? FIND OUT!!!!".',
}

PLAIN = (tb.NEUTRAL + "It shows a vintage printed magazine cover, tilted, with worn paper edges. Keep every printed letter exactly as it appears in the crop; never write a word or number "
         "that is not already in the crop, and lettering the crop cuts off stays cut off. Tiny body text stays tiny soft grey lines." + TAIL)

if __name__ == "__main__":
    plain = "--plain" in sys.argv                                              # no quoted lettering: with the whole cover's words in every tile Klein writes them into tiles that hold none
    for stem in [a for a in sys.argv[1:] if a != "--plain"] or COVERS:
        t0 = time.time()
        try:
            out = tb.upscale_job(("magp_" if plain else "mag_") + stem, SB / (stem + ".PNG"), PLAIN if plain else LEAD + COVERS[stem] + TAIL)
            print(stem, out, f"{time.time() - t0:.0f}s", flush=True)
        except Exception as e:                                                  # noqa: BLE001
            print("FAILED", stem, repr(e), flush=True)
