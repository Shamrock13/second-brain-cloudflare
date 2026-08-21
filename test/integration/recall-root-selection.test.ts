import { describe, expect, it, vi } from "vitest";
import { DEFAULTS } from "../../src/config";
import { DEFAULT_EMBEDDING_QUERY_MODE } from "../../src/recall/query-profile";
import { recallEntries } from "../../src/recall/search";
import type { RecallDiagnostics } from "../../src/recall/types";
import { D1Mock } from "../helpers/d1-mock";
import { makeTestEnv, makeVectorizeMock } from "../helpers/make-env";

const ctx = { waitUntil: (_: Promise<unknown>) => {} } as unknown as ExecutionContext;
const seed = (db: D1Mock, id: string, content: string, tags: string[] = []) =>
  db.entries.push({ id, content, tags: JSON.stringify(tags), source: "api", created_at: 1000, vector_ids: "[]", recall_count: 0, importance_score: 0 });
const edge = (db: D1Mock, source_id: string, target_id: string) =>
  db.edges.push({ id: `${source_id}-${target_id}`, source_id, target_id, type: "decided", weight: 1, provenance: "explicit", metadata: "{}", created_at: 1, updated_at: 1 });

describe("recall root selection", () => {
  it("selects a rare lexical root crowded below semantic leaders", async () => {
    const db = new D1Mock();
    for (let i = 0; i < 16; i++) seed(db, `semantic-${i}`, `general summary ${i}`);
    seed(db, "ledger-root", "ledger status changed", ["work"]);
    seed(db, "ledger-answer", "The ledger reconciliation decision fixed the status", ["work"]);
    edge(db, "ledger-root", "ledger-answer");
    const prepare = db.prepare.bind(db);
    (db as any).prepare = (sql: string) => sql.includes("WHERE content LIKE") && sql.includes("ORDER BY created_at DESC LIMIT")
      ? { bind: () => ({ all: async () => ({ results: [] }) }) }
      : prepare(sql);
    const diagnostics: RecallDiagnostics = {};
    const env = makeTestEnv(db, { VECTORIZE: makeVectorizeMock({ query: vi.fn().mockResolvedValue({ matches: [
      ...Array.from({ length: 16 }, (_, i) => ({ id: `semantic-${i}`, score: .95 - i * .01, metadata: { parentId: `semantic-${i}` } })),
      { id: "ledger-root", score: .5, metadata: { parentId: "ledger-root" } },
    ] }) }) });

    const result = await recallEntries({ query: "why did the ledger reconciliation status change", topK: 5, hops: 1, synthesize: false }, env, ctx, undefined, { diagnostics });
    expect(diagnostics.rootSelections).toContainEqual({ id: "ledger-root", selectedBy: "lexical" });
    expect(result.matches.map(x => x.id)).toContain("ledger-answer");
  });

  it("backfills the fifth direct result when every neighborhood abstains", async () => {
    const db = new D1Mock();
    for (let i = 0; i < 5; i++) seed(db, `d${i}`, `direct status result ${i}`, ["work"]);
    seed(db, "unrelated-neighbor", "grocery list");
    edge(db, "d0", "unrelated-neighbor");
    const diagnostics: RecallDiagnostics = {};
    const env = makeTestEnv(db, { VECTORIZE: makeVectorizeMock({ query: vi.fn().mockResolvedValue({ matches: Array.from({ length: 5 }, (_, i) => ({ id: `d${i}`, score: .9 - i * .01, metadata: { parentId: `d${i}` } })) }) }) });

    const result = await recallEntries({ query: "status work", topK: 5, hops: 1, synthesize: false }, env, ctx, undefined, { diagnostics });
    expect(result.matches.map(x => x.id)).toEqual(["d0", "d1", "d2", "d3", "d4"]);
    expect(diagnostics.selectedRelatedIds).toEqual([]);
    expect(result.matches.map(x => x.id)).not.toContain("unrelated-neighbor");
  });

  it("does not populate root-selection diagnostics at hops:0", async () => {
    const db = new D1Mock();
    seed(db, "direct", "direct status", ["work"]);
    const diagnostics: RecallDiagnostics = {};
    const env = makeTestEnv(db, { VECTORIZE: makeVectorizeMock({ query: vi.fn().mockResolvedValue({ matches: [{ id: "direct", score: .9, metadata: { parentId: "direct" } }] }) }) });

    await recallEntries({ query: "direct status", topK: 5, hops: 0, synthesize: false }, env, ctx, undefined, { diagnostics });
    expect(diagnostics.rootSelections).toBeUndefined();
    expect(diagnostics.expandedIds).toBeUndefined();
  });

  it("never returns a direct candidate again as a graph-hop result", async () => {
    const db = new D1Mock();
    seed(db, "d0", "alpha root");
    seed(db, "d1", "other");
    seed(db, "d2", "alpha beta evidence");
    seed(db, "d3", "other three");
    seed(db, "d4", "other four");
    edge(db, "d0", "d2");
    const env = makeTestEnv(db, { VECTORIZE: makeVectorizeMock({ query: vi.fn().mockResolvedValue({ matches: [
      ...["d0", "d1", "d2", "d3", "d4"].map((id, i) => ({ id, score: .9 - i * .01, metadata: { parentId: id } })),
    ] }) }) });

    const result = await recallEntries({ query: "alpha beta", topK: 5, hops: 1, synthesize: false }, env, ctx);
    expect(new Set(result.matches.map(match => match.id)).size).toBe(result.matches.length);
    expect(result.matches.filter(match => match.hop > 0).map(match => match.id)).not.toContain("d2");
  });

  it("normalizes a selected root against the actual positive root maximum", async () => {
    const db = new D1Mock();
    seed(db, "low-root", "alpha beta gamma");
    seed(db, "normalized-answer", "delta epsilon evidence");
    edge(db, "low-root", "normalized-answer");
    const env = makeTestEnv(db, { VECTORIZE: makeVectorizeMock({ query: vi.fn().mockResolvedValue({ matches: [{ id: "low-root", score: .1, metadata: { parentId: "low-root" } }] }) }) });

    const result = await recallEntries({ query: "alpha beta gamma delta epsilon", topK: 5, hops: 1, synthesize: false }, env, ctx);
    expect(result.matches.map(match => match.id)).toContain("normalized-answer");
  });

  it("keeps a specific root when direct frequency favors a popular summary", async () => {
    const db = new D1Mock();
    seed(db, "popular-summary", "summary", []); (db.entries.at(-1) as any).recall_count = 10_000;
    seed(db, "specific-root", "alpha beta context");
    seed(db, "specific-answer", "alpha beta decision evidence");
    edge(db, "specific-root", "specific-answer");
    const diagnostics: RecallDiagnostics = {};
    const env = makeTestEnv(db, { VECTORIZE: makeVectorizeMock({ query: vi.fn().mockResolvedValue({ matches: [
      { id: "specific-root", score: .8, metadata: { parentId: "specific-root" } },
      { id: "popular-summary", score: .7, metadata: { parentId: "popular-summary" } },
    ] }) }) });

    await recallEntries({ query: "alpha beta", topK: 5, hops: 1, synthesize: false }, env, ctx, undefined, { diagnostics });
    expect(diagnostics.rootSelections?.map(selection => selection.id)).toContain("specific-root");
    expect(diagnostics.rootSelections?.[0].id).not.toBe("popular-summary");
  });

  it("uses a query-relevant local chunk instead of an unrelated parent section", async () => {
    const db = new D1Mock();
    seed(db, "chunk-root", "unrelated parent section");
    seed(db, "chunk-answer", "gamma delta evidence");
    seed(db, "unrelated-neighbor", "grocery list");
    edge(db, "chunk-root", "chunk-answer");
    edge(db, "chunk-root", "unrelated-neighbor");
    for (let i = 0; i < 5; i++) seed(db, `direct-${i}`, "alpha beta gamma direct evidence");
    const prepare = db.prepare.bind(db);
    (db as any).prepare = (sql: string) => sql.includes("WHERE content LIKE") && sql.includes("ORDER BY created_at DESC LIMIT")
      ? { bind: () => ({ all: async () => ({ results: [] }) }) }
      : prepare(sql);
    const env = makeTestEnv(db, { VECTORIZE: makeVectorizeMock({ query: vi.fn().mockResolvedValue({ matches: [
      ...Array.from({ length: 5 }, (_, i) => ({ id: `direct-${i}`, score: .9 - i * .01, metadata: { parentId: `direct-${i}` } })),
      { id: "chunk-root", score: .5, metadata: { parentId: "chunk-root", content: "alpha beta local chunk" } },
    ] }) }) });

    const diagnostics: RecallDiagnostics = {};
    const result = await recallEntries({ query: "alpha beta gamma delta", topK: 5, hops: 1, synthesize: false }, env, ctx, undefined, { diagnostics });
    expect(diagnostics.rejections).not.toContainEqual({ id: "chunk-answer", reason: "no-evidence-gain" });
    expect(result.matches.map(match => match.id)).toContain("chunk-answer");
    expect(result.matches.map(match => match.id)).not.toContain("unrelated-neighbor");
  });

  it("uses the configured default embedding mode when the internal override is absent", async () => {
    const db = new D1Mock();
    seed(db, "root", "ledger");
    const query = vi.fn().mockResolvedValue({ matches: [{ id: "root", score: .9, metadata: { parentId: "root" } }] });
    const env = makeTestEnv(db, { VECTORIZE: makeVectorizeMock({ query }) });
    const diagnostics: RecallDiagnostics = {};
    const expectedInput = {
      distilled: "ledger",
      semantic: "why ledger",
      hybrid: "why ledger ledger",
    } as const;

    await recallEntries({ query: "why ledger", topK: 5, synthesize: false }, env, ctx, undefined, { diagnostics });

    expect(env.AI.run).toHaveBeenCalledTimes(1);
    expect(env.AI.run).toHaveBeenCalledWith(DEFAULTS.EMBEDDING_MODEL, {
      text: [expectedInput[DEFAULT_EMBEDDING_QUERY_MODE]],
    });
    expect(diagnostics.embeddingMode).toBe(DEFAULT_EMBEDDING_QUERY_MODE);
  });

  it.each([
    ["distilled", "ledger"],
    ["semantic", "why ledger"],
    ["hybrid", "why ledger ledger"],
  ] as const)("uses one embedding call in %s mode", async (embeddingQueryMode, expectedInput) => {
    const db = new D1Mock();
    seed(db, "root", "ledger");
    const query = vi.fn().mockResolvedValue({ matches: [{ id: "root", score: .9, metadata: { parentId: "root" } }] });
    const env = makeTestEnv(db, { VECTORIZE: makeVectorizeMock({ query }) });

    await recallEntries({ query: "why ledger", topK: 5, synthesize: false }, env, ctx, undefined, { embeddingQueryMode });
    expect(env.AI.run).toHaveBeenCalledWith(DEFAULTS.EMBEDDING_MODEL, { text: [expectedInput] });
    expect(query).toHaveBeenCalledTimes(1);
    expect((env.AI.run as ReturnType<typeof vi.fn>).mock.calls.map(call => call[0])).toEqual([DEFAULTS.EMBEDDING_MODEL]);
  });
});
