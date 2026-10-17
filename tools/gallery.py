"""gallery.py SITE_DIR [--loop SECONDS]

Static before/after review site for the HD texture pack, rebuilt from out/*/ on every run.
  SITE_DIR/index.html  the page (filters, search, sort, per-unit metrics)
  SITE_DIR/data.json   one record per unit
  SITE_DIR/orig/<uid>.png             the original texture (copied; they are tiny)
  SITE_DIR/full/<uid>.png             symlink to the FULL-RESOLUTION result - no thumbnails anywhere

The page shows the real files; its viewer puts original and result at the same pixel size (the original
is enlarged with nearest-neighbour so its true pixels show), side by side with synced scrolling, or as a
blink toggle, at fit / 1x / 2x / 4x. Safe to run while agents write (a half-written PNG is skipped and
picked up next pass). --loop keeps rebuilding.
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import texlab as T

Image.MAX_IMAGE_PIXELS = 200_000_000


def build(site):
    site = Path(site)
    (site / "orig").mkdir(parents=True, exist_ok=True)
    (site / "full").mkdir(exist_ok=True)
    tasks = {t["id"]: t for t in json.load(open(ROOT / "tasks" / "tasks.json"))}
    cat = T.catalog()
    recs, seen = [], {}
    for d in sorted((ROOT / "out").glob("t*")):
        if d.name.startswith("_") or not d.is_dir():
            continue
        gate = {}
        if (d / "gate.json").exists():
            try:
                gate = json.load(open(d / "gate.json"))
            except Exception:
                pass
        for sj in d.glob("*.json"):
            if sj.name == "gate.json":
                continue
            try:
                meta = json.load(open(sj))
            except Exception:
                continue
            uid = meta.get("uid") or sj.stem
            if uid not in cat["units"]:
                continue
            u = cat["units"][uid]
            png = d / f"{uid}.png"
            status = meta.get("status", "ok")
            g = gate.get(uid)
            if status == "ok" and not png.exists():
                continue
            mt = max(sj.stat().st_mtime, png.stat().st_mtime if png.exists() else 0)
            if uid in seen and seen[uid]["mtime"] >= mt:         # several tasks may cover a unit: newest wins
                continue
            gstat = ("esrgan" if status == "esrgan_needed" else "keep" if status == "keep_original" else
                     "failed" if status == "failed" else
                     "ungated" if g is None else "pass" if g["pass"] else "fail")
            rec = {
                "uid": uid, "stem": u["stem"], "task": d.name, "family": tasks.get(d.name, {}).get("family"),
                "w": u["w"], "h": u["h"], "tw": u["target"][0], "th": u["target"][1],
                "alpha": bool(u["has_alpha"]), "chapters": u["chapters"], "status": gstat,
                "fid": meta.get("fidelity"), "method": (meta.get("method") or "")[:140],
                "sources": meta.get("sources", []), "notes": (meta.get("notes") or meta.get("why") or "")[:200],
                "node": ((u.get("usage") or {}).get("nodes") or [""])[0],
                "problems": (g or {}).get("problems", [])[:2],
                "m": {k: v for k, v in ((g or {}).get("metrics") or {}).items()
                      if k in ("ssim", "mae", "grain", "bands", "detail_gain", "alpha_iou", "orig_flat")},
                "mtime": int(mt), "o": None, "r": None, "bytes": 0,
            }
            try:
                dst = site / "orig" / f"{uid}.png"
                if not dst.exists() or dst.stat().st_mtime < T.orig_path(uid).stat().st_mtime:
                    shutil.copyfile(T.orig_path(uid), dst)
                rec["o"] = f"orig/{uid}.png"
                if png.exists():
                    Image.open(png).verify()                     # half-written file: skip, next pass picks it up
                    link = site / "full" / f"{uid}.png"
                    if link.is_symlink() or link.exists():
                        link.unlink()
                    link.symlink_to(png)
                    rec["r"] = f"full/{uid}.png"
                    rec["bytes"] = png.stat().st_size
            except Exception as e:
                print("skip", uid, repr(e), file=sys.stderr)
                continue
            seen[uid] = rec
    recs = sorted(seen.values(), key=lambda r: -r["mtime"])
    tot = sum(1 for u in cat["units"].values() if not u["is_mip"] and not u["is_placeholder"])
    counts = {}
    for r in recs:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    payload = {"units": recs, "total_units": tot, "counts": counts, "built": int(time.time())}
    tmp = site / "data.json.tmp"
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, site / "data.json")
    (site / "index.html").write_text(PAGE)
    return len(recs), counts, tot


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CSVM HD texture review</title>
<style>
:root{--bg:#14161a;--panel:#1b1e24;--line:#2a2e36;--fg:#e6e3dc;--dim:#9aa0aa;--accent:#e0a458;
--pass:#6fbf8a;--fail:#e0705e;--warn:#e0a458;--chk1:#2c2f37;--chk2:#1f2228;color-scheme:dark}
@media (prefers-color-scheme:light){:root:not([data-theme=dark]){--bg:#f1efe9;--panel:#fbfaf6;--line:#d8d4c8;--fg:#1d1f23;
--dim:#5d636d;--accent:#9a5b13;--pass:#2b7a49;--fail:#b3402f;--warn:#9a5b13;--chk1:#e2dfd6;--chk2:#d2cfc5;color-scheme:light}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
body.lock{overflow:hidden}
header{position:sticky;top:0;z-index:5;background:var(--bg);padding:12px 0 10px;border-bottom:1px solid var(--line)}
h1{font-size:17px;margin:0 0 2px;font-weight:650;letter-spacing:.01em}
.sub{color:var(--dim);font-size:12.5px;margin-bottom:8px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;display:flex;margin:6px 0 10px;max-width:640px}
.bar i{display:block;height:100%}
.ctl{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
input,select,button{background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 9px;font:inherit}
button{cursor:pointer}
input[type=search]{min-width:12rem;flex:1 1 12rem;max-width:22rem}
button:focus-visible,input:focus-visible,select:focus-visible,a:focus-visible,.t:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font-size:12.5px;padding:4px 9px}
[aria-pressed=true]{border-color:var(--accent);color:var(--accent)}
main{display:grid;grid-template-columns:minmax(0,1fr);gap:16px;padding:14px 0 40px;max-width:1400px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;min-width:0}
.top{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:baseline;margin-bottom:8px}
.name{font-weight:620;overflow-wrap:anywhere}
.meta{color:var(--dim);font-size:12px;font-variant-numeric:tabular-nums}
.badge{font-size:11px;padding:1px 7px;border-radius:9px;border:1px solid currentColor;text-transform:uppercase;letter-spacing:.05em}
.pass{color:var(--pass)}.fail,.failed{color:var(--fail)}.esrgan,.ungated,.keep{color:var(--warn)}
.tiles{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
figure{margin:0;min-width:0}
.t{display:block;width:100%;padding:0;overflow:hidden;border-radius:4px;border:1px solid var(--line);cursor:zoom-in;
background:repeating-conic-gradient(var(--chk1) 0 25%,var(--chk2) 0 50%) 0 0/16px 16px;max-height:80vh}
.t img{display:block;width:100%;height:100%;object-fit:contain}
.px{image-rendering:pixelated}
figcaption{color:var(--dim);font-size:11.5px;margin-top:3px}
.detail{margin-top:8px;color:var(--dim);font-size:12px;overflow-wrap:anywhere}
.detail b{color:var(--fg);font-weight:560}
.prob{color:var(--fail)}
.empty{color:var(--dim);padding:24px 0}
#more{margin:0 0 40px}
/* viewer */
#lb{position:fixed;inset:0;z-index:20;background:var(--bg);display:none;flex-direction:column;
padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
#lb.on{display:flex}
#lbbar{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;padding:8px max(12px,env(safe-area-inset-left,0px));border-bottom:1px solid var(--line)}
#lbtitle{font-weight:620;overflow-wrap:anywhere;flex:1 1 14rem;min-width:0}
#lbtitle small{color:var(--dim);font-weight:400;margin-left:8px;font-variant-numeric:tabular-nums}
.grp{display:flex;gap:4px}
#lbbody{flex:1;min-height:0;display:flex;gap:6px;padding:6px}
.pane{flex:1 1 0;min-width:0;overflow:auto;border:1px solid var(--line);border-radius:4px;position:relative;
background:repeating-conic-gradient(var(--chk1) 0 25%,var(--chk2) 0 50%) 0 0/16px 16px}
.pane img{display:block;max-width:none}
.tag{position:sticky;top:0;left:0;display:inline-block;z-index:2;font-size:11.5px;padding:2px 8px;background:var(--bg);color:var(--fg);border-bottom-right-radius:6px;opacity:.9}
#lbinfo{padding:4px max(12px,env(safe-area-inset-left,0px)) 8px;color:var(--dim);font-size:12px;overflow-wrap:anywhere}
@media (max-width:700px){.tiles{grid-template-columns:minmax(0,1fr)}#lbbody{flex-direction:column}}
</style></head><body>
<header>
<h1>CSVM HD texture review <a href="live/" style="font-size:13px;font-weight:400;margin-left:10px">live redraw pipeline →</a></h1>
<div class="sub" id="sub">loading…</div>
<div class="bar" id="bar" aria-hidden="true"></div>
<div class="ctl">
<input type="search" id="q" placeholder="search name, family, world node, library id" aria-label="search">
<select id="task" aria-label="task"><option value="">all tasks</option></select>
<select id="sort" aria-label="sort"><option value="new">newest first</option><option value="name">name</option><option value="bad">worst gate first</option><option value="big">largest first</option></select>
<div class="chips" id="chips"></div>
</div></header>
<main id="grid"></main>
<div id="more"></div>

<div id="lb" role="dialog" aria-modal="true" aria-label="full resolution viewer">
 <div id="lbbar">
  <div id="lbtitle"></div>
  <div class="grp" id="zoomg"></div>
  <div class="grp"><button id="mside" aria-pressed="true">side by side</button><button id="mblink" aria-pressed="false">blink (B)</button></div>
  <div class="grp"><button id="prev" aria-label="previous">&larr;</button><button id="next" aria-label="next">&rarr;</button><button id="close" aria-label="close">close (Esc)</button></div>
 </div>
 <div id="lbbody"></div>
 <div id="lbinfo"></div>
</div>

<script>
"use strict";
const $=s=>document.querySelector(s);
const state={status:new Set(),q:"",task:"",sort:"new",shown:24};
let DATA=null,LIST=[];
const stat=["pass","fail","ungated","esrgan","keep","failed"];
const esc=s=>String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmtB=n=>n>1e6?(n/1e6).toFixed(1)+" MB":Math.round(n/1e3)+" kB";

function chips(){const box=$("#chips");box.innerHTML="";
 for(const s of stat){const n=DATA.counts[s]||0;if(!n)continue;const b=document.createElement("button");
  b.className="chip";b.textContent=s+" "+n;b.setAttribute("aria-pressed",state.status.has(s));
  b.onclick=()=>{state.status.has(s)?state.status.delete(s):state.status.add(s);state.shown=24;chips();render()};box.append(b)}}

function card(r,i){
 const ar=`aspect-ratio:${r.tw}/${r.th}`;
 const o=r.o?`<img class="px" src="${r.o}" loading="lazy" decoding="async" alt="original">`:"";
 const rr=r.r?`<img src="${r.r}" loading="lazy" decoding="async" alt="result" width="${r.tw}" height="${r.th}">`:`<div class="meta" style="padding:20px">no image yet</div>`;
 const m=r.m||{};const bits=[];
 if(m.ssim!=null)bits.push("SSIM "+m.ssim);if(m.mae!=null)bits.push("MAE "+m.mae);
 if(m.grain)bits.push(`grain ${m.grain[0]}°/${m.grain[1]} → ${m.grain[2]}°/${m.grain[3]}`);
 if(m.orig_flat)bits.push("soft original");else if(m.detail_gain!=null)bits.push("gain "+m.detail_gain+"×");
 return `<article class="card"><div class="top"><span class="name">${esc(r.stem)}</span>
 <span class="badge ${r.status}">${r.status}</span>
 <span class="meta">${r.w}×${r.h} → ${r.tw}×${r.th}${r.bytes?" · "+fmtB(r.bytes):""} · ${esc(r.task)} · ${esc(r.family||"")}${r.node?" · "+esc(r.node):""}</span></div>
 <div class="tiles"><figure><button class="t" style="${ar}" data-i="${i}" aria-label="open ${esc(r.stem)} in the viewer">${o}</button><figcaption>original, enlarged with nearest-neighbour</figcaption></figure>
 <figure><button class="t" style="${ar}" data-i="${i}" aria-label="open ${esc(r.stem)} in the viewer">${rr}</button><figcaption>result, full-resolution file (click to inspect at 1:1)</figcaption></figure></div>
 <div class="detail"><b>${esc(r.fid||"")}</b> ${esc(r.method)}${r.sources.length?"<br>library: "+r.sources.map(esc).join(", "):""}
 ${bits.length?"<br>"+bits.join(" · "):""}${r.problems.map(p=>`<br><span class="prob">${esc(p)}</span>`).join("")}${r.notes?"<br>"+esc(r.notes):""}</div></article>`}

function filtered(){
 let L=DATA.units.filter(r=>(!state.status.size||state.status.has(r.status))&&(!state.task||r.task===state.task));
 const q=state.q.trim().toLowerCase();
 if(q)L=L.filter(r=>(r.stem+" "+(r.family||"")+" "+r.node+" "+r.sources.join(" ")+" "+r.method).toLowerCase().includes(q));
 if(state.sort==="name")L=[...L].sort((a,b)=>a.stem.localeCompare(b.stem));
 if(state.sort==="big")L=[...L].sort((a,b)=>b.tw*b.th-a.tw*a.th);
 if(state.sort==="bad"){const K={fail:0,failed:0,ungated:1,esrgan:2,keep:3,pass:4};const k=s=>(s in K?K[s]:5);L=[...L].sort((a,b)=>k(a.status)-k(b.status)||b.mtime-a.mtime)}
 return L}
function render(){
 LIST=filtered();const shown=LIST.slice(0,state.shown);
 $("#grid").innerHTML=shown.length?shown.map(card).join(""):`<div class="empty">Nothing matches.</div>`;
 $("#more").innerHTML=LIST.length>shown.length?`<button id="morebtn">Show 24 more (${LIST.length-shown.length} left)</button>`:"";
 const mb=$("#morebtn");if(mb)mb.onclick=()=>{state.shown+=24;render()}}
$("#grid").addEventListener("click",e=>{const t=e.target.closest(".t");if(t)openLB(+t.dataset.i)});

function head(){
 const c=DATA.counts,done=(c.pass||0)+(c.keep||0),t=DATA.total_units;
 $("#sub").textContent=`${DATA.units.length} textures produced · ${done} of ${t} units accepted (${(100*done/t).toFixed(1)}%) · refreshed ${new Date(DATA.built*1000).toLocaleTimeString()}`;
 const col={pass:"var(--pass)",keep:"var(--warn)",esrgan:"var(--warn)",ungated:"var(--dim)",fail:"var(--fail)",failed:"var(--fail)"};
 $("#bar").innerHTML=stat.map(s=>c[s]?`<i style="width:${100*c[s]/t}%;background:${col[s]}" title="${s} ${c[s]}"></i>`:"").join("");
 const sel=$("#task"),cur=sel.value,ts=[...new Set(DATA.units.map(r=>r.task))].sort();
 sel.innerHTML='<option value="">all tasks</option>'+ts.map(x=>`<option>${x}</option>`).join("");sel.value=cur}

/* ---- viewer: original and result at the same pixel size ---- */
const V={i:0,zoom:"fit",mode:"side",blink:false,syncing:false};
const ZOOMS=["fit","1","2","4"];
$("#zoomg").innerHTML=ZOOMS.map(z=>`<button data-z="${z}" aria-pressed="${z==="fit"}">${z==="fit"?"fit":z+"×"}</button>`).join("");
function openLB(i){V.i=i;document.body.classList.add("lock");$("#lb").classList.add("on");drawLB();$("#close").focus()}
function closeLB(){$("#lb").classList.remove("on");document.body.classList.remove("lock")}
function imgHTML(src,r,px,w,h){return `<img src="${src}" ${px?'class="px"':""} style="width:${w}px;height:${h}px" alt="">`}
function drawLB(){
 const r=LIST[V.i];if(!r)return closeLB();
 const body=$("#lbbody");
 const pw=Math.max(200,body.clientWidth/(V.mode==="side"&&innerWidth>700?2:1)-14),ph=Math.max(200,body.clientHeight-14);
 let z=V.zoom==="fit"?Math.min(pw/r.tw,ph/r.th):+V.zoom;
 const w=Math.round(r.tw*z),h=Math.round(r.th*z);
 const rpx=z>=2;                       /* result gets pixelated only when magnified past its own pixels */
 const O=r.o?imgHTML(r.o,r,true,w,h):"",R=r.r?imgHTML(r.r,r,rpx,w,h):`<div class="tag" style="position:static">no result image</div>`;
 if(V.mode==="side"){
  body.innerHTML=`<div class="pane" id="p1"><span class="tag">original ${r.w}×${r.h}, nearest-neighbour</span>${O}</div>
                  <div class="pane" id="p2"><span class="tag">result ${r.tw}×${r.th}</span>${R}</div>`;
  const a=$("#p1"),b=$("#p2");
  const sync=(x,y)=>()=>{if(V.syncing)return;V.syncing=true;y.scrollLeft=x.scrollLeft;y.scrollTop=x.scrollTop;V.syncing=false};
  a.onscroll=sync(a,b);b.onscroll=sync(b,a);
 }else{
  body.innerHTML=`<div class="pane" id="p1"><span class="tag" id="btag"></span><span id="bimg"></span></div>`;
  const show=()=>{$("#bimg").innerHTML=V.blink?O:R;$("#btag").textContent=V.blink?`original ${r.w}×${r.h} (click to switch)`:`result ${r.tw}×${r.th} (click to switch)`};
  show();$("#p1").onclick=()=>{V.blink=!V.blink;show()};
 }
 $("#lbtitle").innerHTML=`${esc(r.stem)}<small>${V.i+1} / ${LIST.length} · ${r.status} · ${(z*100).toFixed(0)}% · ${r.tw}×${r.th}${r.bytes?" · "+fmtB(r.bytes):""}</small>`;
 $("#lbinfo").innerHTML=`<b>${esc(r.fid||"")}</b> ${esc(r.method)}${r.sources.length?" · library: "+r.sources.map(esc).join(", "):""}${r.problems.length?" · "+r.problems.map(esc).join(" · "):""}`;
 document.querySelectorAll("#zoomg button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.z===V.zoom));
 $("#mside").setAttribute("aria-pressed",V.mode==="side");$("#mblink").setAttribute("aria-pressed",V.mode==="blink");
}
$("#zoomg").onclick=e=>{const b=e.target.closest("button");if(b){V.zoom=b.dataset.z;drawLB()}};
$("#mside").onclick=()=>{V.mode="side";drawLB()};
$("#mblink").onclick=()=>{V.mode="blink";drawLB()};
$("#prev").onclick=()=>{V.i=(V.i-1+LIST.length)%LIST.length;drawLB()};
$("#next").onclick=()=>{V.i=(V.i+1)%LIST.length;drawLB()};
$("#close").onclick=closeLB;
addEventListener("resize",()=>{if($("#lb").classList.contains("on"))drawLB()});
addEventListener("keydown",e=>{
 if(!$("#lb").classList.contains("on"))return;
 if(e.key==="Escape")closeLB();
 else if(e.key==="ArrowLeft"){e.preventDefault();$("#prev").click()}
 else if(e.key==="ArrowRight"){e.preventDefault();$("#next").click()}
 else if(e.key==="b"||e.key==="B"){V.mode="blink";V.blink=!V.blink;drawLB()}
 else if(e.key==="+"||e.key==="="){const i=Math.min(ZOOMS.length-1,ZOOMS.indexOf(V.zoom)+1);V.zoom=ZOOMS[i];drawLB()}
 else if(e.key==="-"){const i=Math.max(0,ZOOMS.indexOf(V.zoom)-1);V.zoom=ZOOMS[i];drawLB()}
});

async function load(first){
 try{const r=await fetch("data.json?"+Date.now());DATA=await r.json()}catch(e){$("#sub").textContent="could not load data.json";return}
 head();chips();
 /* the 60 s refresh must not yank the page out from under someone who is inspecting a texture */
 if(first||(!$("#lb").classList.contains("on")&&scrollY<40))render();
 if(first){const m=/^#v(\d+)$/.exec(location.hash);if(m&&LIST[+m[1]])openLB(+m[1])}}
$("#q").oninput=e=>{state.q=e.target.value;state.shown=24;render()};
$("#task").onchange=e=>{state.task=e.target.value;state.shown=24;render()};
$("#sort").onchange=e=>{state.sort=e.target.value;state.shown=24;render()};
load(true);setInterval(()=>load(false),60000);
</script></body></html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("--loop", type=int, default=0)
    a = ap.parse_args()
    while True:
        t0 = time.time()
        n, counts, tot = build(a.site)
        print(f"{time.strftime('%H:%M:%S')} {n} units {counts} of {tot} in {time.time() - t0:.0f}s", flush=True)
        if not a.loop:
            break
        time.sleep(a.loop)
