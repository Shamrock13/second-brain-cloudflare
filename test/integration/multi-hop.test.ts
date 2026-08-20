import { describe, it, expect, vi, beforeEach } from "vitest";
import { recallEntries } from "../../src/recall/search";
import { makeTestEnv, makeTestDb, makeVectorizeMock } from "../helpers/make-env";
import type { Env } from "../../src/env";
import { D1Mock } from "../helpers/d1-mock";

function makeCtx() {
  const pending: Promise<any>[] = [];
  return {
    ctx: { waitUntil: (p: Promise<any>) => pending.push(p) } as any as ExecutionContext,
    drain: () => Promise.allSettled(pending),
  };
}

function seed(db: D1Mock, id: string, content: string, tags: string[] = []) {
  db.entries.push({ id, content, tags: JSON.stringify(tags), source: "api", created_at: 1000, vector_ids: "[]", recall_count: 0, importance_score: 0 });
}

function pushEdge(db: D1Mock, source_id: string, target_id: string, weight = 0.8) {
  db.edges.push({ id: `${source_id}-${target_id}`, source_id, target_id, type: "relates_to", weight, provenance: "inferred", metadata: "{}", created_at: 1, updated_at: 1 });
}

function denseEnv(db: D1Mock, matches: { id: string; score: number }[]) {
  return makeTestEnv(db, {
    VECTORIZE: makeVectorizeMock({
      query: vi.fn().mockResolvedValue({ matches: matches.map(m => ({ id: m.id, score: m.score, metadata: { parentId: m.id, isUpdate: false } })) }),
    }),
  });
}

describe("multi-hop recall (issue #16)", () => {
  let db: D1Mock;

  beforeEach(() => {
    db = makeTestDb();
  });

  it("hops:0 returns only direct matches even when a graph exists (no regression)", async () => {
    seed(db, "seed", "Direct match");
    seed(db, "neighbor", "Related context");
    pushEdge(db, "seed", "neighbor");
    const env = denseEnv(db, [{ id: "seed", score: 0.9 }]);
    const { ctx } = makeCtx();

    const res = await recallEntries({ query: "direct", topK: 5, hops: 0 }, env, ctx);
    expect(res.matches.map(m => m.id)).toEqual(["seed"]);
  });

  it("hops:1 surfaces a 1-hop neighbor that hops:0 misses, with direct matches still first", async () => {
    seed(db, "seed", "Direct match");
    seed(db, "neighbor", "Related context");
    pushEdge(db, "seed", "neighbor");
    const env = denseEnv(db, [{ id: "seed", score: 0.9 }]);
    const { ctx } = makeCtx();

    const res = await recallEntries({ query: "direct", topK: 5, hops: 1 }, env, ctx);
    expect(res.matches.map(m => m.id)).toEqual(["seed", "neighbor"]);
    expect(res.matches[0].hop).toBe(0);
    expect(res.matches[1].hop).toBe(1);
  });

  it("carries edge provenance, timestamp, and parent onto a graph-expanded match (#225)", async () => {
    seed(db, "seed", "Direct match");
    seed(db, "neighbor", "Related context");
    pushEdge(db, "seed", "neighbor"); // pushEdge sets provenance "inferred", created_at 1
    const env = denseEnv(db, [{ id: "seed", score: 0.9 }]);
    const { ctx } = makeCtx();

    const res = await recallEntries({ query: "direct", topK: 5, hops: 1 }, env, ctx);
    const hop = res.matches.find(m => m.id === "neighbor")!;
    expect(hop.viaProvenance).toBe("inferred");
    expect(hop.viaFrom).toBe("seed");
    expect(hop.viaLinkedAt).toBe(1);
    // a direct seed match carries no via* fields
    expect(res.matches.find(m => m.id === "seed")!.viaProvenance).toBeUndefined();
  });

  it("does not traverse into a status:deprecated neighbor", async () => {
    seed(db, "seed", "Direct match");
    seed(db, "neighbor", "Related context", ["status:deprecated"]);
    pushEdge(db, "seed", "neighbor");
    const env = denseEnv(db, [{ id: "seed", score: 0.9 }]);
    const { ctx } = makeCtx();

    const res = await recallEntries({ query: "direct", topK: 5, hops: 1 }, env, ctx);
    expect(res.matches.map(m => m.id)).toEqual(["seed"]);
  });

  it("bumps recall_count for direct seeds only, not graph-expanded neighbors", async () => {
    seed(db, "seed", "Direct match");
    seed(db, "neighbor", "Related context");
    pushEdge(db, "seed", "neighbor");
    const env = denseEnv(db, [{ id: "seed", score: 0.9 }]);
    const { ctx, drain } = makeCtx();

    await recallEntries({ query: "direct", topK: 5, hops: 1 }, env, ctx);
    await drain();

    expect(db.entries.find((e: any) => e.id === "seed").recall_count).toBe(1);
    expect(db.entries.find((e: any) => e.id === "neighbor").recall_count).toBe(0);
  });

  it("does not let expanded neighbors push out direct matches when topK is full", async () => {
    for (let i = 0; i < 5; i++) seed(db, `d${i}`, "Direct match");
    seed(db, "neighbor", "Related context");
    pushEdge(db, "d0", "neighbor");
    const env = denseEnv(db, [0, 1, 2, 3, 4].map(i => ({ id: `d${i}`, score: 0.9 - i * 0.05 })));
    const { ctx } = makeCtx();

    const res = await recallEntries({ query: "direct", topK: 5, hops: 1 }, env, ctx);
    expect(res.matches).toHaveLength(5);
    expect(res.matches.map(m => m.id)).not.toContain("neighbor"); // direct matches fill topK
  });

  it("surfaces relevant linked evidence from a root just outside direct topK", async () => {
    for (let i = 0; i < 5; i++) {
      seed(db, `direct-${i}`, `why anniversary changed distraction ${i}`);
      (db.entries.at(-1) as any).created_at = 2000 - i;
    }
    seed(db, "candidate-root", "why anniversary changed because the original trip was impractical");
    seed(db, "linked-answer", "Chateau Elan solved the sitter constraint");
    db.edges.push({
      id: "candidate-answer",
      source_id: "candidate-root",
      target_id: "linked-answer",
      type: "decided",
      weight: 1,
      provenance: "explicit",
      metadata: "{}",
      created_at: 1,
      updated_at: 1,
    });
    const env = denseEnv(db, [
      ...[0, 1, 2, 3, 4].map(i => ({ id: `direct-${i}`, score: 0.95 - i * 0.04 })),
      { id: "candidate-root", score: 0.7 },
    ]);
    const { ctx, drain } = makeCtx();

    const res = await recallEntries({ query: "why anniversary changed", topK: 5, hops: 1, synthesize: false }, env, ctx);
    await drain();

    expect(res.matches).toHaveLength(5);
    expect(res.matches[0].id).toBe("direct-0");
    expect(res.matches.map(m => m.id)).toContain("linked-answer");
    expect(res.matches.find(m => m.id === "linked-answer")?.viaFrom).toBe("candidate-root");
    expect(db.entries.find((e: any) => e.id === "direct-0").recall_count).toBe(1);
    expect(db.entries.find((e: any) => e.id === "direct-4").recall_count).toBe(0);
    expect(db.entries.find((e: any) => e.id === "candidate-root").recall_count).toBe(0);
    expect(db.entries.find((e: any) => e.id === "linked-answer").recall_count).toBe(0);
  });

  it("uses the same Workers AI models with and without graph traversal", async () => {
    const directDb = makeTestDb();
    seed(directDb, "seed", "Direct match");
    seed(directDb, "neighbor", "Related context");
    pushEdge(directDb, "seed", "neighbor");
    const graphDb = makeTestDb();
    seed(graphDb, "seed", "Direct match");
    seed(graphDb, "neighbor", "Related context");
    pushEdge(graphDb, "seed", "neighbor");
    const directEnv = denseEnv(directDb, [{ id: "seed", score: 0.9 }]);
    const graphEnv = denseEnv(graphDb, [{ id: "seed", score: 0.9 }]);

    await recallEntries({ query: "direct", topK: 5, hops: 0, synthesize: false }, directEnv, makeCtx().ctx);
    await recallEntries({ query: "direct", topK: 5, hops: 1, synthesize: false }, graphEnv, makeCtx().ctx);

    const models = (testEnv: Env) => (testEnv.AI.run as ReturnType<typeof vi.fn>).mock.calls.map(call => call[0]);
    expect(models(graphEnv)).toEqual(models(directEnv));
    expect(models(graphEnv)).toEqual(["@cf/baai/bge-small-en-v1.5"]);
  });
});
