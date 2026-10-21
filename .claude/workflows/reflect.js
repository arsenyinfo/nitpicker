export const meta = {
  name: 'reflect',
  description: 'Reflect on recent nitpicker sessions: per-session friction analysis, then a repo-verified synthesis',
  whenToUse: 'Same as `nitpicker reflect`, but runs the LLM stages on the Claude subscription. args: {n?: number, sessionsDir?: string}',
  phases: [
    { title: 'Load', detail: 'nitpicker reflect --dump: deterministic session dossiers' },
    { title: 'Analyze', detail: 'one analyst per session (prompts/reflect/map.md)', model: 'sonnet' },
    { title: 'Synthesize', detail: 'cross-session synthesis with repo verification (prompts/reflect/reduce.md)' },
  ],
}

const n = (args && args.n) || 20
const sessionsDirFlag = args && args.sessionsDir ? ` --sessions-dir '${args.sessionsDir}'` : ''

phase('Load')
const loaded = await agent(
  `Run exactly this shell command from the repository root and report the file paths it prints on stdout (one per line), in order:

    cargo run -q -- reflect --n ${n}${sessionsDirFlag} --dump "$(mktemp -d)"

Do not read the files. If the command fails, return an empty list and put the stderr tail in \`error\`.`,
  {
    label: 'dump sessions',
    model: 'sonnet',
    effort: 'low',
    schema: {
      type: 'object',
      properties: {
        paths: { type: 'array', items: { type: 'string' } },
        error: { type: 'string' },
      },
      required: ['paths'],
    },
  },
)
if (!loaded || loaded.paths.length === 0) {
  throw new Error(`no session dossiers produced: ${(loaded && loaded.error) || 'loader agent died'}`)
}
log(`${loaded.paths.length} session dossier(s) loaded`)

const sessionName = path => path.split('/').pop().replace(/\.md$/, '')

const analyses = await pipeline(loaded.paths, path =>
  agent(
    `Read prompts/reflect/map.md in the repository: it is your full instruction set. Then read the session dossier at ${path} completely (page through it with offset/limit if it is too large for one read) and produce the report those instructions describe.

Work from the dossier alone: do not explore the repository or run commands, since code-level causes are the synthesizer's job. Return only the report text.`,
    { label: `analyze ${sessionName(path)}`, phase: 'Analyze', model: 'sonnet' },
  ).then(text => ({ name: sessionName(path), text })),
)

const succeeded = analyses.filter(a => a && a.text)
const failed = loaded.paths.length - succeeded.length
if (failed > 0) log(`${failed} session analysis(es) failed and are excluded from the dataset`)
if (succeeded.length === 0) throw new Error('all session analyses failed')

phase('Synthesize')
const reports = succeeded.map(a => `## ${a.name}\n\n${a.text}`).join('\n\n---\n\n')
const report = await agent(
  `Read prompts/reflect/reduce.md in the repository: it is your full instruction set. Use the repository (read files, grep, git log/show) to verify code-level causes as it requires. Do not modify any files. Return only the final report.

Dataset: ${succeeded.length} successfully analyzed session(s). Use this denominator for every support count.

${reports}`,
  { label: 'synthesize', phase: 'Synthesize' },
)

return report
