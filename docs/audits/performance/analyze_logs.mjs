// Reproduce the historical ETL evidence without running ETL or opening databases.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../../..');
const log = fs.readFileSync(path.join(root, 'logs/stock_etl.log'), 'utf8');
const records = [];
for (const [i, line] of log.split(/\r?\n/).entries()) {
  const m = line.match(/^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) \| (\w+) \| \[([^\]]+)\] (.*)$/);
  if (m) records.push({ line: i + 1, ts: `${m[1]}.${m[2]}`,
    ms: Date.parse(m[1].replace(' ', 'T') + '.' + m[2] + 'Z'), module: m[4], msg: m[5] });
  else if (records.length) records.at(-1).msg += '\n' + line;
}
const starts = records.flatMap((r, i) => r.msg.includes('STARTING ETL PIPELINE') ? [i] : []);
const runs = starts.map((start, i) => {
  const all = records.slice(start, starts[i + 1] ?? records.length);
  const complete = all.find(e => e.msg.includes('PIPELINE COMPLETED SUCCESSFULLY'));
  const events = all.filter(e => !complete || e.ms <= complete.ms);
  const get = tag => events.find(e => e.msg.includes(tag));
  const seconds = (a, b) => a && b ? (b.ms - a.ms) / 1000 : null;
  const number = regex => Number(events.map(e => e.msg.match(regex)).find(Boolean)?.[1].replaceAll(',', '')) || null;
  const memory = events.filter(e => e.msg.includes('Memory optimized:'));
  const gaps = events.slice(1).map((e, j) => ({ seconds: seconds(events[j], e),
    fromLine: events[j].line, toLine: e.line, from: events[j].msg.trim(), to: e.msg.trim() }))
    .sort((a, b) => b.seconds - a.seconds).slice(0, 3);
  return { run: i + 1, start: events[0].ts, startLine: events[0].line,
    finish: complete?.ts, finishLine: complete?.line, mode: get('Mode:')?.msg,
    extractSeconds: number(/Extract: ([\d.]+)/), transformSeconds: number(/Transform: ([\d.]+)/),
    loadSeconds: number(/Load: ([\d.]+)/), reportedTotalSeconds: number(/Total time\s*:\s*([\d.]+)/),
    wallSeconds: seconds(events[0], complete),
    postSwapSeconds: seconds(get('ATOMIC SWAP COMPLETE'), get('Score snapshot:')),
    snapshotNames: number(/Score snapshot: (\d+)/), priceRows: number(/Prices: ([\d,]+)/),
    postReadGapSeconds: seconds(memory.at(-1), get('Score snapshot:')),
    memoryMessages: memory.map(({ line, ts, msg }) => ({ line, ts, msg })), largestGaps: gaps };
});
fs.writeFileSync(path.join(here, 'historical_timings.json'), JSON.stringify({
  kind: 'Observed historical logging timestamps, not controlled new benchmarks',
  logSha256: crypto.createHash('sha256').update(log).digest('hex'),
  clockNote: 'Local timestamp differences; timezone, sleep, clock adjustment, source versions and provider states not controlled.',
  runs }, null, 2) + '\n');
console.log(JSON.stringify(runs.map(({ run, wallSeconds, extractSeconds, postSwapSeconds, reportedTotalSeconds }) =>
  ({ run, wallSeconds, extractSeconds, postSwapSeconds, reportedTotalSeconds })), null, 2));
