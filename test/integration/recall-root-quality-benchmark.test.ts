import { describe, expect, it, vi } from "vitest";
import { DEFAULTS } from "../../src/config";
import { graphSeedLimit, relatedSlotLimit } from "../../src/recall/neighborhood";
import { mmrRerank, rerankWithTimeDecay } from "../../src/recall/math";
import { localEvidenceOf } from "../../src/recall/root-candidate";
import type { RootCandidate } from "../../src/recall/root-selector";
import { recallEntries } from "../../src/recall/search";
import type { RecallDiagnostics } from "../../src/recall/types";
import {
  ROOT_QUALITY_CASES,
  type CandidateFixture,
  type RootQualityCase,
  type RootQualitySplit,
} from "../fixtures/recall-root-quality";
import { D1Mock } from "../helpers/d1-mock";
import { makeTestEnv, makeVectorizeMock } from "../helpers/make-env";

const TOP_K = 5;

interface CaseObservation {
  id: string;
  split: RootQualitySplit;
  candidateAvailable: boolean;
  fused: boolean;
  seed: boolean;
  expanded: boolean;
  selectedRelatedIds: string[];
  authoritative: boolean;
  baselineAuthoritative: boolean;
  directTopFourRegression: boolean;
  extraAiCalls: number;
  extraVectorizeQueries: number;
  diagnostics: RecallDiagnostics;
}

interface BenchmarkMetrics {
  cases: number;
  candidateAvailability: number;
  fusionSurvival: number;
  seedHits: number;
  neighborhoodReach: number;
  authoritativeAnswers: number;
  baselineAuthoritativeAnswers: number;
  improvement: number;
  usefulGraphPrecision: number;
  directTopFourRegressions: number;
  extraAiCalls: number;
  extraVectorizeQueries: number;
}

const caseId = (c: RootQualityCase) => `${c.domain}/${c.failureShape}`;
const rawCandidates = (c: RootQualityCase) => c.candidates.filter(candidate => candidate.baselineScore !== undefined);

function baselineRootIds(candidates: RootCandidate[], topK: number, lambda: number): string[] {
  return mmrRerank(candidates, lambda, graphSeedLimit(topK, candidates.length))
    .map(candidate => candidate.parentId);
}

function baselineLinkedEligible(content: string, tokens: string[]): boolean {
  const lower = content.toLowerCase();
  return tokens.some(token => lower.includes(token.toLowerCase()));
}

function baselineRecall(c: RootQualityCase, tokens: string[]): string[] {
  const fixtures = rawCandidates(c);
  const recallCounts = new Map(fixtures.map(candidate => [candidate.id, candidate.recallCount ?? 0]));
  const tags = new Map(fixtures.map(candidate => [candidate.id, candidate.tags ?? []]));
  const reranked = rerankWithTimeDecay(
    fixtures.map(candidate => ({
      id: candidate.id,
      score: candidate.baselineScore!,
      metadata: { parentId: candidate.id, created_at: candidate.createdAt ?? 1 },
    })),
    recallCounts,
    new Map(),
    [],
    new Map(),
    new Map(),
    tags,
    DEFAULTS,
  );
  const candidates: RootCandidate[] = reranked.map(match => ({
    ...match,
    parentId: match.id,
    rootScore: match.score,
    localEvidence: c.candidates.find(candidate => candidate.id === match.id)?.vectorContent ?? "",
    tags: tags.get(match.id) ?? [],
    lexicalCoverage: 0,
    metadataAlignment: 0,
  }));
  const roots = new Set(baselineRootIds(candidates, TOP_K, DEFAULTS.MMR_LAMBDA));
  const directIds = mmrRerank(reranked, DEFAULTS.MMR_LAMBDA, TOP_K).map(candidate => candidate.id);
  const rows = new Map(c.candidates.map(candidate => [candidate.id, candidate]));
  const related = c.edges
    .flatMap(edge => {
      const linkedId = roots.has(edge.sourceId)
        ? edge.targetId
        : roots.has(edge.targetId)
          ? edge.sourceId
          : undefined;
      if (!linkedId || directIds.includes(linkedId)) return [];
      const linked = rows.get(linkedId);
      return linked && baselineLinkedEligible(linked.content, tokens) ? [linkedId] : [];
    })
    .slice(0, relatedSlotLimit(TOP_K));
  return [...directIds.slice(0, TOP_K - related.length), ...related];
}

function installControlledQueries(db: D1Mock, c: RootQualityCase): void {
  const prepare = db.prepare.bind(db);
  (db as unknown as { prepare: (sql: string) => unknown }).prepare = (sql: string) => {
    if (sql.includes("SELECT COUNT(*) AS total") && sql.includes("SUM(CASE WHEN content LIKE")) {
      return {
        bind: (...patterns: string[]) => ({
          first: async () => {
            // Keep the eight-token weak-neighborhood query intact so its two generic
            // matches clear the lexical-count gate but remain below the score threshold.
            if (c.failureShape === "weak-generic-neighbor") throw new Error("controlled corpus scan unavailable");
            return Object.fromEntries([
              ["total", 100],
              ...patterns.map((_, index) => [`d${index}`, 2]),
            ]);
          },
        }),
      };
    }
    if (sql.includes("WHERE content LIKE") && sql.includes("ORDER BY created_at DESC LIMIT")) {
      const results = c.candidates
        .filter(candidate => candidate.keywordCandidate)
        .map(candidate => ({
          id: candidate.id,
          content: candidate.content,
          tags: JSON.stringify(candidate.tags ?? []),
          source: "benchmark",
          created_at: candidate.createdAt ?? 1,
        }));
      return { bind: () => ({ all: async () => ({ results }) }) };
    }
    return prepare(sql);
  };
}

function buildFixture(c: RootQualityCase) {
  const db = new D1Mock();
  for (const candidate of c.candidates) {
    db.entries.push({
      id: candidate.id,
      content: candidate.content,
      tags: JSON.stringify(candidate.tags ?? []),
      source: "benchmark",
      created_at: candidate.createdAt ?? 1,
      updated_at: candidate.createdAt ?? 1,
      vector_ids: "[]",
      recall_count: candidate.recallCount ?? 0,
      importance_score: 0,
      contradiction_wins: 0,
      contradiction_losses: 0,
    });
  }
  for (const [index, edge] of c.edges.entries()) {
    db.edges.push({
      id: `${caseId(c)}-edge-${index}`,
      source_id: edge.sourceId,
      target_id: edge.targetId,
      type: edge.type,
      weight: edge.weight,
      provenance: edge.provenance,
      metadata: "{}",
      created_at: 1,
      updated_at: 1,
    });
  }
  installControlledQueries(db, c);
  const query = vi.fn().mockResolvedValue({
    matches: c.candidates
      .filter((candidate): candidate is CandidateFixture & { denseScore: number } => candidate.denseScore !== undefined)
      .sort((a, b) => b.denseScore - a.denseScore)
      .map(candidate => ({
        id: candidate.id,
        score: candidate.denseScore,
        metadata: {
          parentId: candidate.id,
          content: candidate.vectorContent,
          created_at: candidate.createdAt ?? 1,
        },
      })),
  });
  const env = makeTestEnv(db, { VECTORIZE: makeVectorizeMock({ query }) });
  const ctx = { waitUntil: (_promise: Promise<unknown>) => undefined } as unknown as ExecutionContext;
  return { env, ctx, query };
}

async function runCase(c: RootQualityCase): Promise<CaseObservation> {
  const direct = buildFixture(c);
  const graph = buildFixture(c);
  const diagnostics: RecallDiagnostics = {};
  const withoutGraph = await recallEntries(
    { query: c.query, topK: TOP_K, hops: 0, synthesize: false },
    direct.env,
    direct.ctx,
  );
  const withGraph = await recallEntries(
    { query: c.query, topK: TOP_K, hops: 1, synthesize: false },
    graph.env,
    graph.ctx,
    undefined,
    { diagnostics },
  );
  const acceptableRoots = new Set(c.acceptableRootIds);
  const authoritative = new Set(c.authoritativeIds);
  const candidateAvailable = rawCandidates(c).some(candidate => acceptableRoots.has(candidate.id));
  const fused = (diagnostics.fusedIds ?? []).some(id => acceptableRoots.has(id));
  const seed = (diagnostics.rootSelections ?? []).some(selection => acceptableRoots.has(selection.id));
  const expanded = (diagnostics.expandedIds ?? []).some(id => authoritative.has(id));
  const outputIds = withGraph.matches.map(match => match.id);
  const baselineIds = baselineRecall(c, withGraph.queryTokens ?? []);
  const graphAiCalls = (graph.env.AI.run as ReturnType<typeof vi.fn>).mock.calls.length;

  return {
    id: caseId(c),
    split: c.split,
    candidateAvailable,
    fused,
    seed,
    expanded,
    selectedRelatedIds: diagnostics.selectedRelatedIds ?? [],
    authoritative: outputIds.some(id => authoritative.has(id)),
    baselineAuthoritative: baselineIds.some(id => authoritative.has(id)),
    directTopFourRegression: JSON.stringify(outputIds.slice(0, 4)) !== JSON.stringify(withoutGraph.matches.map(match => match.id).slice(0, 4)),
    extraAiCalls: Math.max(0, graphAiCalls - 1),
    extraVectorizeQueries: Math.max(0, graph.query.mock.calls.length - 1),
    diagnostics,
  };
}

function summarize(observations: CaseObservation[]): BenchmarkMetrics {
  const selectedRelatedIds = observations.flatMap(observation => observation.selectedRelatedIds.map(id => ({ observation, id })));
  const usefulRelated = selectedRelatedIds.filter(({ observation, id }) => {
    const c = ROOT_QUALITY_CASES.find(candidate => caseId(candidate) === observation.id)!;
    return c.authoritativeIds.includes(id);
  }).length;
  const authoritativeAnswers = observations.filter(observation => observation.authoritative).length;
  const baselineAuthoritativeAnswers = observations.filter(observation => observation.baselineAuthoritative).length;
  return {
    cases: observations.length,
    candidateAvailability: observations.filter(observation => observation.candidateAvailable).length,
    fusionSurvival: observations.filter(observation => observation.fused).length,
    seedHits: observations.filter(observation => observation.candidateAvailable && observation.seed).length,
    neighborhoodReach: observations.filter(observation => observation.expanded).length,
    authoritativeAnswers,
    baselineAuthoritativeAnswers,
    improvement: authoritativeAnswers - baselineAuthoritativeAnswers,
    usefulGraphPrecision: selectedRelatedIds.length ? usefulRelated / selectedRelatedIds.length : 1,
    directTopFourRegressions: observations.filter(observation => observation.directTopFourRegression).length,
    extraAiCalls: observations.reduce((sum, observation) => sum + observation.extraAiCalls, 0),
    extraVectorizeQueries: observations.reduce((sum, observation) => sum + observation.extraVectorizeQueries, 0),
  };
}

async function evaluate(cases: readonly RootQualityCase[]) {
  const observations = [] as CaseObservation[];
  for (const c of cases) observations.push(await runCase(c));
  return { observations, metrics: summarize(observations) };
}

function reportMetrics(label: string, metrics: BenchmarkMetrics): void {
  if (process.env.RECALL_BENCHMARK_REPORT === "1") {
    console.info(`RECALL_ROOT_QUALITY ${label} ${JSON.stringify(metrics)}`);
  }
}

function expectSplitGates(split: RootQualitySplit, metrics: BenchmarkMetrics, observations: CaseObservation[]) {
  const details = JSON.stringify({ split, metrics, observations }, null, 2);
  expect(metrics.cases, details).toBe(10);
  expect(metrics.candidateAvailability, details).toBe(8);
  expect(metrics.fusionSurvival, details).toBe(8);
  expect(metrics.seedHits, details).toBeGreaterThanOrEqual(split === "development" ? 7 : 6);
  expect(metrics.authoritativeAnswers, details).toBeGreaterThanOrEqual(7);
  expect(metrics.usefulGraphPrecision, details).toBeGreaterThanOrEqual(0.7);
  expect(metrics.directTopFourRegressions, details).toBe(0);
  expect(metrics.extraAiCalls, details).toBe(0);
  expect(metrics.extraVectorizeQueries, details).toBe(0);
}

describe("frozen recall root-quality fixture", () => {
  it("contains 20 generic, alternating development/holdout cases with literal controls", () => {
    expect(ROOT_QUALITY_CASES).toHaveLength(20);
    expect(ROOT_QUALITY_CASES.filter(c => c.split === "development")).toHaveLength(10);
    expect(ROOT_QUALITY_CASES.filter(c => c.split === "holdout")).toHaveLength(10);
    expect(ROOT_QUALITY_CASES.filter(c => !c.candidateAvailable)).toHaveLength(4);
    for (const c of ROOT_QUALITY_CASES) {
      const observed = rawCandidates(c).some(candidate => c.acceptableRootIds.includes(candidate.id));
      expect(observed, caseId(c)).toBe(c.candidateAvailable);
    }
    for (const domain of ["personal", "enterprise", "product", "architecture"] as const) {
      const cases = ROOT_QUALITY_CASES.filter(c => c.domain === domain);
      expect(cases).toHaveLength(5);
      expect(new Set(cases.map(c => c.failureShape)).size).toBe(5);
    }
    for (const shape of new Set(ROOT_QUALITY_CASES.map(c => c.failureShape))) {
      expect(new Set(ROOT_QUALITY_CASES.filter(c => c.failureShape === shape).map(c => c.split))).toEqual(new Set(["development", "holdout"]));
    }
  });
});

describe("frozen recall root-quality benchmark", () => {
  it("development aggregate meets the frozen gates", async () => {
    const { observations, metrics } = await evaluate(ROOT_QUALITY_CASES.filter(c => c.split === "development"));
    reportMetrics("development", metrics);
    expectSplitGates("development", metrics, observations);
  });

  it("holdout aggregate meets the frozen gates", async () => {
    const { observations, metrics } = await evaluate(ROOT_QUALITY_CASES.filter(c => c.split === "holdout"));
    reportMetrics("holdout", metrics);
    expectSplitGates("holdout", metrics, observations);
  });

  it("20-case aggregate improves authoritative answers by at least four", async () => {
    const { observations, metrics } = await evaluate(ROOT_QUALITY_CASES);
    reportMetrics("overall", metrics);
    const details = JSON.stringify({ metrics, observations }, null, 2);
    expect(metrics.candidateAvailability, details).toBe(16);
    expect(metrics.seedHits, details).toBeGreaterThanOrEqual(13);
    expect(metrics.authoritativeAnswers, details).toBeGreaterThanOrEqual(14);
    expect(metrics.usefulGraphPrecision, details).toBeGreaterThanOrEqual(0.7);
    expect(metrics.authoritativeAnswers - metrics.baselineAuthoritativeAnswers, details).toBeGreaterThanOrEqual(4);
    expect(metrics.directTopFourRegressions, details).toBe(0);
    expect(metrics.extraAiCalls, details).toBe(0);
    expect(metrics.extraVectorizeQueries, details).toBe(0);
  });

  it("frequency-neutral popularity sentinel keeps the specific root", async () => {
    const c = ROOT_QUALITY_CASES.find(candidate => candidate.failureShape === "popular-broad-summary" && candidate.split === "development")!;
    const observation = await runCase(c);
    expect(observation.diagnostics.rootSelections?.map(selection => selection.id)).toContain(c.acceptableRootIds[0]);
    expect(observation.authoritative).toBe(true);
  });

  it("query-local long-parent sentinel preserves complementary evidence", async () => {
    const c = ROOT_QUALITY_CASES.find(candidate => candidate.failureShape === "long-parent-pollution" && candidate.split === "development")!;
    const root = c.candidates.find(candidate => candidate.id === c.acceptableRootIds[0])!;
    expect(localEvidenceOf(
      { id: root.id, score: root.denseScore!, metadata: { parentId: root.id, content: root.vectorContent } },
      root.content,
      ["maple", "booking", "change"],
    )).toBe("maple planning context");
    const observation = await runCase(c);
    expect(observation.expanded).toBe(true);
    expect(observation.authoritative).toBe(true);
  });

  it("reserved lexical sentinel reaches a crowded specific root", async () => {
    const c = ROOT_QUALITY_CASES.find(candidate => candidate.failureShape === "crowded-lexical-root" && candidate.split === "development")!;
    const observation = await runCase(c);
    expect(observation.diagnostics.rootSelections).toContainEqual({ id: c.acceptableRootIds[0], selectedBy: "lexical" });
    expect(observation.authoritative).toBe(true);
  });

  it("neighborhood threshold backfill sentinel rejects weak generic evidence", async () => {
    const c = ROOT_QUALITY_CASES.find(candidate => candidate.failureShape === "weak-generic-neighbor" && candidate.split === "development")!;
    const observation = await runCase(c);
    expect(observation.diagnostics.selectedRelatedIds).toEqual([]);
    expect(observation.diagnostics.rejections).toContainEqual({ id: `${c.domain}-weak-neighbor`, reason: "weak-neighborhood" });
    expect(observation.authoritative).toBe(true);
    expect(observation.baselineAuthoritative).toBe(false);
  });

  it("AI and Vectorize parity sentinel keeps the controlled path at one call", async () => {
    const c = ROOT_QUALITY_CASES.find(candidate => candidate.failureShape === "crowded-lexical-root" && candidate.split === "development")!;
    const fixture = buildFixture(c);
    await recallEntries({ query: c.query, topK: TOP_K, hops: 1, synthesize: false }, fixture.env, fixture.ctx);
    expect((fixture.env.AI.run as ReturnType<typeof vi.fn>).mock.calls.map(call => call[0])).toEqual([DEFAULTS.EMBEDDING_MODEL]);
    expect(fixture.query).toHaveBeenCalledTimes(1);
  });
});
