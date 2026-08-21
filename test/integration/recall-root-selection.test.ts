import { describe, expect, it, vi } from "vitest";
import { DEFAULTS } from "../../src/config";
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
    for (let i = 0; i < 5; i++) seed(db, `semantic-${i}`, `general summary ${i}`);
    seed(db, "ledger-root", "ledger status changed", ["work"]);
    seed(db, "ledger-answer", "The ledger reconciliation decision fixed the status", ["work"]);
    edge(db, "ledger-root", "ledger-answer");
    const prepare = db.prepare.bind(db);
    (db as any).prepare = (sql: string) => sql.includes("WHERE content LIKE") && sql.includes("ORDER BY created_at DESC LIMIT")
      ? { bind: () => ({ all: async () => ({ results: [] }) }) }
      : prepare(sql);
    const diagnostics: RecallDiagnostics = {};
    const env = makeTestEnv(db, { VECTORIZE: makeVectorizeMock({ query: vi.fn().mockResolvedValue({ matches: [
      ...Array.from({ length: 5 }, (_, i) => ({ id: `semantic-${i}`, score: .95 - i * .01, metadata: { parentId: `semantic-${i}` } })),
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

  it.each(["distilled", "semantic", "hybrid"] as const)("uses one embedding call in %s mode", async (embeddingQueryMode) => {
    const db = new D1Mock();
    seed(db, "root", "ledger status change", ["work"]);
    const query = vi.fn().mockResolvedValue({ matches: [{ id: "root", score: .9, metadata: { parentId: "root" } }] });
    const env = makeTestEnv(db, { VECTORIZE: makeVectorizeMock({ query }) });

    await recallEntries({ query: "ledger status work", topK: 5, synthesize: false }, env, ctx, undefined, { embeddingQueryMode });
    expect(query).toHaveBeenCalledTimes(1);
    expect((env.AI.run as ReturnType<typeof vi.fn>).mock.calls.map(call => call[0])).toEqual([DEFAULTS.EMBEDDING_MODEL]);
  });
});
