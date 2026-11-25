// k6 load test for POST /v1/answer.
//   VUS=500 BASE_URL=http://127.0.0.1:8080 k6 run load_tests/k6_load.js            (local)
//   VUS=50000 BASE_URL=https://api.example.com k6 run ...                          (cluster; see README for distribution)
// Env: VUS (peak virtual users, default 500)  HOLD (steady-state seconds, default 120)  RAMP (seconds per ramp, default 60)
//      THINK (seconds between requests per VU, default 1)  SUMMARY_OUT (default load_tests/out/k6_summary.json)
// Thresholds are the design KPI-31..34 values.
import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';
import { Trend, Rate } from 'k6/metrics';

const VUS = parseInt(__ENV.VUS || '500');
const HOLD = parseInt(__ENV.HOLD || '120');
const RAMP = parseInt(__ENV.RAMP || '60');
const THINK = parseFloat(__ENV.THINK || '1');
const BASE = __ENV.BASE_URL || 'http://127.0.0.1:8080';

const tokens = new SharedArray('tokens', () => JSON.parse(open('./tokens.json')));
const QUERIES = [
  'Who has to approve a wire transfer above 10,000 USD?', 'What is the minimum password length?',
  'How often are user access rights reviewed?', 'Is SKU-099 refundable once it has been activated?',
  'When is the holiday change freeze for production deployments?', 'How long is the battery warranty for SKU-205?',
  'Which certification must tier 1 vendors hold?', 'How many days of remote work are allowed per week?',
  'How long are expense reports retained?', 'What is the incident severity 1 response time?',
  'How do I bake sourdough bread?', 'Hello',
];

// Per-stage server timings from the response body (timings_ms), so the KPI converter can fill KPI-01/05 under load.
const tTotal = new Trend('srv_total_ms', true), tPre = new Trend('srv_hook_pre_ms', true), tTriage = new Trend('srv_triage_ms', true);
const tRetr = new Trend('srv_retrieval_ms', true), tGen = new Trend('srv_generation_ms', true);
const server5xx = new Rate('http_5xx');
const unexpected = new Rate('http_unexpected_status');   // anything other than 200 (and 429, which is the designed overload answer)

export const options = {
  scenarios: {
    ramp: {
      executor: 'ramping-vus', startVUs: 0,
      stages: [
        { duration: `${RAMP}s`, target: Math.ceil(VUS * 0.25) },
        { duration: `${RAMP}s`, target: VUS },
        { duration: `${HOLD}s`, target: VUS },
        { duration: `${Math.ceil(RAMP / 2)}s`, target: 0 },
      ],
      gracefulRampDown: '10s',
    },
  },
  thresholds: {
    http_req_duration: ['p(50)<250', 'p(95)<450', 'p(99)<600'],
    http_5xx: ['rate<0.0005'],            // 0.05%
    http_req_failed: ['rate<0.0005'],
    http_unexpected_status: ['rate<0.0005'],
  },
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

export default function () {
  const tok = tokens[(__VU - 1) % tokens.length];
  const q = QUERIES[Math.floor(Math.random() * QUERIES.length)];
  const res = http.post(`${BASE}/v1/answer`, JSON.stringify({ query: q }), {
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${tok}` },
    tags: { name: 'answer' },
  });
  server5xx.add(res.status >= 500);
  unexpected.add(res.status !== 200 && res.status !== 429);
  if (res.status === 200) {
    const t = res.json('timings_ms');
    if (t) {
      if (t.total !== undefined) tTotal.add(t.total);
      if (t['hook.pre'] !== undefined) tPre.add(t['hook.pre']);
      if (t.triage !== undefined) tTriage.add(t.triage);
      if (t.retrieval !== undefined) tRetr.add(t.retrieval);
      if (t.generation !== undefined) tGen.add(t.generation);
    }
  }
  check(res, { 'status 200': (r) => r.status === 200 });
  sleep(THINK);
}

export function handleSummary(data) {
  const out = __ENV.SUMMARY_OUT || 'load_tests/out/k6_summary.json';
  data.setup_data = { vus_target: VUS, hold_s: HOLD, ramp_s: RAMP, think_s: THINK, base_url: BASE };
  return { [out]: JSON.stringify(data, null, 2), stdout: textSummary(data) };
}

function textSummary(d) {
  const m = d.metrics, f = (x) => (x === undefined ? 'n/a' : x.toFixed(1));
  const dur = m.http_req_duration.values;
  return `\nreqs=${m.http_reqs.values.count} rps=${f(m.http_reqs.values.rate)} p50=${f(dur.med)} p95=${f(dur['p(95)'])} ` +
    `p99=${f(dur['p(99)'])} 5xx=${(m.http_5xx.values.rate * 100).toFixed(3)}%\n`;
}
