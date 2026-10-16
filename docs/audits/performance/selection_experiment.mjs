// Audit-only algorithm experiment, NOT a Python/pandas/application benchmark.
// No network, database access, package installation, or production writes.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { performance } from 'node:perf_hooks';

const here = path.dirname(fileURLToPath(import.meta.url));
const sizes = [1, 10, 100, 500, 1000, 5000];
const observations = 756;
const repeats = 5;

function fixture(n) {
  const ids = new Uint32Array(n * observations);
  const prices = new Float64Array(ids.length);
  for (let day = 0; day < observations; day++) {
    for (let ticker = 0; ticker < n; ticker++) {
      const row = day * n + ticker;
      ids[row] = ticker;
      prices[row] = 100 + ticker % 97 + day * 0.125;
    }
  }
  return { ids, prices };
}

// A cheaper analogue of repeated full-frame ticker masking: no pandas sorting,
// strings, DataFrame allocation or financial calculations. Do not transfer its
// absolute runtime/speedup to the real application.
function repeatedScan({ ids, prices }, n) {
  const result = new Float64Array(n * 3);
  for (let ticker = 0; ticker < n; ticker++) {
    let count = 0, sum = 0, last = 0;
    for (let row = 0; row < ids.length; row++) {
      if (ids[row] === ticker) { count++; sum += prices[row]; last = prices[row]; }
    }
    result.set([count, sum, last], ticker * 3);
  }
  return result;
}

// General two-pass partitioning, preserving each ticker's original row order.
// The extra row-index arrays are included in timed work and memory measurements.
function partitionOnce({ ids, prices }, n) {
  const counts = new Uint32Array(n);
  for (const ticker of ids) counts[ticker]++;
  const groups = Array.from(counts, count => new Uint32Array(count));
  const offsets = new Uint32Array(n);
  for (let row = 0; row < ids.length; row++) groups[ids[row]][offsets[ids[row]]++] = row;
  const result = new Float64Array(n * 3);
  for (let ticker = 0; ticker < n; ticker++) {
    let sum = 0, last = 0;
    for (const row of groups[ticker]) { sum += prices[row]; last = prices[row]; }
    result.set([groups[ticker].length, sum, last], ticker * 3);
  }
  return result;
}

function verify(result, n) {
  // Prices use binary-exact .125 increments; exact equality is appropriate here.
  for (let ticker = 0; ticker < n; ticker++) {
    const first = 100 + ticker % 97;
    const expectedSum = observations * first + .125 * observations * (observations - 1) / 2;
    assert.equal(result[ticker * 3], observations);
    assert.equal(result[ticker * 3 + 1], expectedSum);
    assert.equal(result[ticker * 3 + 2], first + (observations - 1) * .125);
  }
}

function measure(fn) {
  global.gc?.();
  const before = process.memoryUsage();
  const cpu0 = process.cpuUsage();
  const t0 = performance.now();
  const result = fn();
  const wallMs = performance.now() - t0;
  const cpu = process.cpuUsage(cpu0);
  const after = process.memoryUsage();
  return { result, metrics: { wallMs, cpuUserMs: cpu.user / 1000,
    cpuSystemMs: cpu.system / 1000, cpuMs: (cpu.user + cpu.system) / 1000,
    rssBefore: before.rss, rssAfter: after.rss, heapBefore: before.heapUsed,
    heapAfter: after.heapUsed, arrayBuffersAfter: after.arrayBuffers } };
}

if (process.argv[2] === '--worker') {
  const n = Number(process.argv[3]), variant = process.argv[4];
  const setup = measure(() => fixture(n));
  const fn = () => (variant === 'baseline' ? repeatedScan : partitionOnce)(setup.result, n);
  const first = measure(fn); verify(first.result, n);
  for (let warm = 0; warm < 2; warm++) verify(fn(), n);
  const trials = [];
  for (let run = 0; run < repeats; run++) {
    const trial = measure(fn); verify(trial.result, n); trials.push(trial.metrics);
  }
  const usage = process.resourceUsage();
  console.log(JSON.stringify({ kind: 'isolated JavaScript selection analogue', n, variant,
    rows: n * observations, fixtureBuild: setup.metrics, firstInvocation: first.metrics,
    trials, maxRSSKiB: usage.maxRSS, verified: 'exact count, sum and final price per ticker',
    financialMetricsVerified: false, appCachingExercised: false,
    inputArrayBytes: setup.result.ids.byteLength + setup.result.prices.byteLength }));
} else {
  const results = [];
  for (const n of sizes) {
    for (const variant of ['baseline', 'partitioned']) {
      if (variant === 'baseline' && n * n * observations > 1e9) {
        results.push({ n, variant, rows: n * observations,
          skipped: 'Bounded local CPU budget: more than 1 billion predicate checks per invocation',
          predicateChecks: n * n * observations });
        continue;
      }
      const child = spawnSync(process.execPath, ['--expose-gc', fileURLToPath(import.meta.url),
        '--worker', String(n), variant], { encoding: 'utf8', timeout: 45000 });
      if (child.status !== 0) throw Error(child.stderr || String(child.error));
      const row = JSON.parse(child.stdout);
      results.push(row);
      console.error(`${n} tickers / ${variant}: ${row.trials.map(x => x.wallMs.toFixed(3)).join(', ')} ms`);
    }
  }
  const payload = { capturedAt: new Date().toISOString(), node: process.version,
    platform: process.platform, cpu: os.cpus()[0].model, logicalCpuCount: os.cpus().length,
    totalMemoryBytes: os.totalmem(), observations, repeats,
    limitations: ['Not application/Python/pandas runtime', 'First invocation is JIT-cold, not an OS or Streamlit cold cache',
      'Memory is whole child process; maxRSS is cumulative high-water mark, not a per-function allocation peak',
      'No rankings, valuations, SQL, external providers, serialization, or full-screen analysis exercised'], results };
  fs.writeFileSync(path.join(here, 'selection_results.json'), JSON.stringify(payload, null, 2) + '\n');
  console.log(JSON.stringify({ output: 'selection_results.json', cases: results.length }));
}
