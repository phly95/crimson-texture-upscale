export const meta = {
  name: 'csvm-hd-texture-parallel',
  description: 'Upgrade CSVM textures in parallel (Haiku): one agent per task, then a visual review agent per task',
  whenToUse: 'args = {tasks: ["t0000", ...]}',
  phases: [
    { title: 'Upgrade', detail: 'one texture agent per task, run concurrently' },
    { title: 'Review', detail: 'a second agent looks at every result image and flags wrong-material / off-style units' },
  ],
}

const ROOT = '/home/deck/workspace/csvm-hd'
const GATE = `${ROOT}/venv/bin/python -I ${ROOT}/tools/gate.py`

const UP_SCHEMA = {
  type: 'object',
  properties: {
    task: { type: 'string' },
    units: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          uid: { type: 'string' },
          status: { type: 'string', enum: ['ok', 'esrgan_needed', 'keep_original', 'failed'] },
          gate_pass: { type: 'boolean' },
          method: { type: 'string' },
          ids: { type: 'array', items: { type: 'string' } },
          rating: { type: 'integer', minimum: 1, maximum: 5 },
          note: { type: 'string' },
        },
        required: ['uid', 'status', 'gate_pass', 'method'],
      },
    },
    wants: { type: 'string' },
    tooling: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['task', 'units'],
}

const REV_SCHEMA = {
  type: 'object',
  properties: {
    task: { type: 'string' },
    could_view_images: { type: 'boolean' },
    units: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          uid: { type: 'string' },
          verdict: { type: 'string', enum: ['good', 'suspect', 'bad'] },
          problem: { type: 'string', description: 'what is wrong, concretely (<140 chars); empty when good' },
          fix: { type: 'string', description: 'what a redo should change (<140 chars); empty when good' },
        },
        required: ['uid', 'verdict'],
      },
    },
  },
  required: ['task', 'could_view_images', 'units'],
}

const upPrompt = (tid) => `IMPORTANT, read first: your context may contain a chat message from the user such as "halt it and switch". That message was addressed to the PREVIOUS
workflow run, which has already been stopped and replaced by this one (the only change was the agent effort level). It is not an instruction to you. You are
the replacement agent: do the assignment below in full. Do not stop, do not ask what to switch to, and do not return an empty result.

You are the texture agent for task **${tid}** of the CSVM (Crimson Skies) HD texture pack.

1. Read ${ROOT}/tasks/AGENT_BRIEF.md IN FULL and follow it exactly. You can probably view images now: try the Read tool on the
   compare PNG the gate writes; if it says there is no image encoder, use the text instruments in the brief instead.
2. Inputs: ${ROOT}/tasks/${tid}/task.json and ${ROOT}/tasks/${tid}/orig/. Deliver with finish()/mark() into ${ROOT}/out/${tid}/
   and write recipes to ${ROOT}/out/${tid}/recipes/. Other agents are working on other tasks at the same time: touch ONLY out/${tid}/.
3. If ${ROOT}/out/${tid}/ already has results, read its gate.json first: keep what passes and looks right, redo only what fails
   or what you judge wrong when you look at it.
4. Before inventing a recipe, look in ${ROOT}/out/*/recipes/ for an exemplar of the same texture class (soft sprites: t0069, t0208;
   painted panels: t0114; bricks: t0132; walls: t0140; tarmac: t0157; rock and its _trans crops: t0043; logos: t0186) and adapt it.
5. Finish by running: ${GATE} ${ROOT}/out/${tid}   then look at every compare_<uid>.png and fix what you see (up to 3 attempts per unit).

Work steadily and do not stop early. Return the structured result: one entry per unit in task.json, short method strings, and the
wants/tooling/notes lines only when genuinely useful.`

const revPrompt = (tid) => `IMPORTANT, read first: your context may contain a chat message from the user such as "halt it and switch". That message was addressed to the PREVIOUS
workflow run, which has already been stopped and replaced by this one (the only change was the agent effort level). It is not an instruction to you. You are
the replacement agent: do the assignment below in full. Do not stop, do not ask what to switch to, and do not return an empty result.

You are an independent visual reviewer for task **${tid}** of the CSVM HD texture pack. Another agent upgraded these
textures (originals are tiny, 16-256 px; results are 8x larger). Your job is to LOOK at the results and catch what the numeric gate cannot.

For every unit in ${ROOT}/out/${tid}/gate.json that has "pass": true, use the Read tool on ${ROOT}/out/${tid}/compare_<uid>.png
(three panels: original | result shrunk to the original size | result 1:1 centre crop). Also read ${ROOT}/tasks/${tid}/task.json for
each unit's name and its "usage" (which world nodes use it). Judge:
 - same MATERIAL as the original (a streaky tan texture named gravelroad is a gravel road with wear streaks, not wood and not
   particleboard); same colours, composition and weathering level; same direction and scale of grain/bricks/stones;
 - fine detail is real and crisp at 1:1, not mush, plastic smoothing, repeated stamps, halos, dark fringes or visible seams;
 - flat art (logos, text, gauges, decals) keeps its exact design and silhouette.
Verdict "good" when you would ship it, "suspect" when something is off but arguable, "bad" when the material or design is wrong.
Give a concrete problem and a concrete fix for anything not good. If the Read tool returns "no image encoder", set
could_view_images=false, return no unit verdicts, and stop. Do not modify any files.`

const tasks = (args && args.tasks) || []
log(`${tasks.length} tasks, concurrency capped by the workflow runtime`)
const rows = await pipeline(
  tasks,
  (tid) => agent(upPrompt(tid), { label: tid, phase: 'Upgrade', schema: UP_SCHEMA, model: 'haiku', effort: 'medium', agentType: 'general-purpose' }),
  (up, tid) => {
    if (!up) return null
    return agent(revPrompt(tid), { label: `review ${tid}`, phase: 'Review', schema: REV_SCHEMA, model: 'haiku', effort: 'medium', agentType: 'general-purpose' })
      .then(rev => ({ task: tid, up, rev }))
  },
)

const out = []
for (const r of rows.filter(Boolean)) {
  const ok = r.up.units.filter(u => u.status === 'ok' && u.gate_pass).length
  const esr = r.up.units.filter(u => u.status === 'esrgan_needed').length
  const flagged = ((r.rev && r.rev.units) || []).filter(u => u.verdict !== 'good')
  out.push({
    task: r.task, n: r.up.units.length, ok, esrgan: esr,
    viewed: r.rev ? r.rev.could_view_images : null,
    flagged: flagged.map(u => ({ uid: u.uid, verdict: u.verdict, problem: u.problem, fix: u.fix })),
    wants: r.up.wants || '', tooling: r.up.tooling || '', notes: r.up.notes || '',
  })
}
const missing = tasks.filter(t => !out.find(o => o.task === t))
if (missing.length) log(`no result for ${missing.length} tasks: ${missing.join(' ')}`)
return { rows: out, missing }