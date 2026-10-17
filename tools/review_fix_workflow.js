export const meta = {
  name: 'csvm-hd-review-fix',
  description: 'Review never-reviewed texture tasks by eye, then fix concrete artifacts reviewers flagged (Haiku, medium effort)',
  whenToUse: 'args = {A: [task ids with a tasks/<id>/review_flags.json], B: [task ids that still need a visual review first]}',
  phases: [
    { title: 'Review', detail: 'B tasks: a reviewer looks at every passing unit and flags concrete artifacts' },
    { title: 'Fix', detail: 'one agent per task redoes only the flagged units, keeps the old result if the new one is not better' },
  ],
}

const ROOT = '/home/deck/workspace/csvm-hd'
const GATE = `${ROOT}/venv/bin/python -I ${ROOT}/tools/gate.py`

const NOTE = `Note: your context may contain chat messages from the user about stopping or switching a previous workflow run. They do not apply to you.
You are a worker with the assignment below. Do the assignment in full; do not stop early and do not return an empty result.`

const CONCRETE = /wobbl|scallop|halo|fringe|dark rim|rim along|wrong material|invent|rock-like|marbl|crack|mirror|symmetr|seam|tint|colou?r (shift|drift)|mean shift|smear|streak|grain (too|is stronger|stronger)|sandpaper|woven|fabric|speck|blot|stair|blocky|artifact|lost|missing|outline/i

const REV_SCHEMA = {
  type: 'object',
  properties: {
    could_view_images: { type: 'boolean' },
    units: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          uid: { type: 'string', description: 'EXACTLY a unit uid from gate.json, e.g. rr_bricks01__1fc53111' },
          verdict: { type: 'string', enum: ['good', 'suspect', 'bad'] },
          problem: { type: 'string', description: 'the concrete visible artifact (<160 chars); empty when good' },
          fix: { type: 'string', description: 'what a redo should change (<160 chars); empty when good' },
        },
        required: ['uid', 'verdict'],
      },
    },
  },
  required: ['could_view_images', 'units'],
}

const FIX_SCHEMA = {
  type: 'object',
  properties: {
    units: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          uid: { type: 'string' },
          action: { type: 'string', enum: ['fixed', 'kept_previous', 'marked_esrgan', 'marked_keep_original', 'not_a_real_problem'] },
          gate_pass: { type: 'boolean' },
          note: { type: 'string', description: '<140 chars' },
        },
        required: ['uid', 'action'],
      },
    },
    lessons: { type: 'string', description: 'one line: a recurring failure mode worth fixing in the shared tools, or empty' },
  },
  required: ['units'],
}

const revPrompt = (tid) => `${NOTE}

You are an independent visual reviewer for task **${tid}** of the CSVM (Crimson Skies) HD texture pack. Another agent upgraded these textures
(originals are tiny, 16-256 px; results are 8x larger). LOOK at the results and catch what the numeric gate cannot.

For every unit in ${ROOT}/out/${tid}/gate.json with "pass": true, use the Read tool on ${ROOT}/out/${tid}/compare_<uid>.png
(three panels: original | result shrunk back to the original size | result 1:1 centre crop). Read ${ROOT}/tasks/${tid}/task.json for each unit's
name and "usage". Judge ONLY concrete visible artifacts, in this order of importance:
 1. WRONG MATERIAL or invented content (rock on a sky, cracks on a flat facade, wood on a road); a different design from the original.
 2. Mirrored or obviously repeated stamps (symmetry that the original does not have), visible seams.
 3. Detail streaks/grain/brush marks smeared over small features (rivets, decals, glyph edges); grain far stronger than the original surface.
 4. Halos, dark fringes, wobbly/scalloped edges where the original has straight ones, blocky stair-steps, colour/tint shifts.
Do NOT flag a result merely for being soft at 1:1 when the original itself is soft; an 8x enlargement of a 32 px original cannot be razor sharp.
Verdict "good" when you would ship it, "suspect" for a real but mild artifact, "bad" when the material or design is wrong.
The uid field must be exactly the unit uid. If the Read tool returns "no image encoder", set could_view_images=false, return no units, and stop.
Do not modify any files.`

const fixPrompt = (tid, inlineFlags) => `${NOTE}

You are the repair agent for task **${tid}** of the CSVM HD texture pack. An independent reviewer looked at the compare images and flagged
concrete artifacts in some units. ${inlineFlags
  ? 'The flags are:\n' + inlineFlags
  : `The flags are in ${ROOT}/tasks/${tid}/review_flags.json (read it).`}

First read ${ROOT}/tasks/AGENT_BRIEF.md (tools, rules, what good means). Then for EACH flagged unit only (leave every other unit alone):
 1. Read ${ROOT}/out/${tid}/compare_<uid>.png and the unit's recipe in ${ROOT}/out/${tid}/recipes/. Decide whether the reviewer is right.
    If not, action "not_a_real_problem" and move on.
 2. BEFORE changing anything, copy the current ${ROOT}/out/${tid}/<uid>.png and <uid>.json to ${ROOT}/out/${tid}/prev/.
 3. Fix the recipe for the stated artifact. Typical remedies: mask detail injection away from saturated/decal/rivet regions; lower the
    streak/grain amplitude to the original's own level; pick a different library asset when the material is wrong (never rock on a sky, never
    cracked concrete on a flat painted facade); use a non-mirrored crop (region(...) with a different seed) instead of mirror tiling; take
    chroma from the original so there is no tint shift; use a straight-edge mask instead of a contour snap for frames/vents; for a flat
    graphic keep the PCHIP/ibp reconstruction and add at most a whisper of grain.
 4. Run ${GATE} ${ROOT}/out/${tid}, Read the new compare PNG, and compare it with the previous one. Keep the new result ONLY if the gate
    passes AND the flagged artifact is gone without a new one. Otherwise copy the previous files back from prev/ (action "kept_previous").
 5. Up to 3 attempts per unit. If a unit truly cannot be done by composition, mark(...) it as esrgan_needed or keep_original with the reason.
Do not chase sharpness for its own sake: softness inherited from a soft original is fine.
Finish by running the gate once more on the whole task folder. Return one entry per flagged unit.`

const A = (args && args.A) || []
const B = (args && args.B) || []
log(`${A.length} tasks go straight to repair, ${B.length} tasks are reviewed first`)

const all = [...A.map(t => ({ tid: t, review: false })), ...B.map(t => ({ tid: t, review: true }))]
const results = await pipeline(
  all,
  (it) => {
    if (!it.review) return { flags: null }                      // flags are on disk for A tasks
    return agent(revPrompt(it.tid), { label: `review ${it.tid}`, phase: 'Review', schema: REV_SCHEMA, model: 'haiku', effort: 'medium', agentType: 'general-purpose' })
      .then(r => {
        if (!r) return null
        const f = (r.units || []).filter(u => u.verdict === 'bad' || (u.verdict === 'suspect' && CONCRETE.test(u.problem || '')))
        return { flags: f, viewed: r.could_view_images, reviewed: (r.units || []).length }
      })
  },
  (prev, it) => {
    if (!prev) return null
    if (it.review && (!prev.viewed || !prev.flags.length)) return { tid: it.tid, reviewed: prev.reviewed, flagged: 0, fixed: [] }
    const inline = it.review
      ? prev.flags.map(f => `- ${f.uid} [${f.verdict}] ${f.problem} | suggested: ${f.fix || ''}`).join('\n')
      : null
    return agent(fixPrompt(it.tid, inline), { label: `fix ${it.tid}`, phase: 'Fix', schema: FIX_SCHEMA, model: 'haiku', effort: 'medium', agentType: 'general-purpose' })
      .then(r => ({ tid: it.tid, reviewed: prev.reviewed || null, flagged: it.review ? prev.flags.length : null, fixed: (r && r.units) || [], lessons: (r && r.lessons) || '' }))
  },
)

const rows = results.filter(Boolean)
const tally = {}
for (const r of rows) for (const u of r.fixed) tally[u.action] = (tally[u.action] || 0) + 1
const lessons = rows.map(r => r.lessons).filter(Boolean)
const missing = all.map(x => x.tid).filter(t => !rows.find(r => r.tid === t))
log(`fix actions: ${JSON.stringify(tally)}; tasks with no result: ${missing.length}`)
return { tally, lessons, missing, tasks: rows.length }