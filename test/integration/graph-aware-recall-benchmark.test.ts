import { describe, expect, it, vi } from "vitest";
import { recallEntries } from "../../src/recall/search";
import { makeTestDb, makeTestEnv, makeVectorizeMock } from "../helpers/make-env";

const cases = [
  { domain: "personal", query: "anniversary childcare", answer: "Chateau Elan solved the sitter constraint" },
  { domain: "enterprise", query: "backend dotnet", answer: "The Microsoft stack aligned with enterprise support skills" },
  { domain: "governance", query: "review cycles", answer: "Separate passes kept critique independent from approval" },
  { domain: "operations", query: "issue ledger", answer: "Closed items were still counted as open" },
  { domain: "product", query: "learn council", answer: "Navigation followed user intent while risk governed oversight" },
  { domain: "launch", query: "product hunt", answer: "Fabian Merian requested the earlier Sunday" },
  { domain: "architecture", query: "okf retrieval", answer: "It is a portable document format rather than a query engine" },
  { domain: "insights", query: "derivepattern rebuilt", answer: "Zero of 135 proposals were accepted" },
  { domain: "platform", query: "graph fanout", answer: "The ceiling bounds database reads and response growth" },
  { domain: "operating-model", query: "advisor partner", answer: "Search changed from a gate into guided recommendations" },
] as const;

function fixture(c: typeof cases[number]) {
  const db = makeTestDb();
  for (let i = 0; i < 5; i++) {
    db.entries.push({
      id: `${c.domain}-direct-${i}`,
      content: `${c.query} unrelated direct note ${i}`,
      tags: "[]",
      source: "api",
      created_at: 2000 - i,
      vector_ids: "[]",
      recall_count: 0,
      importance_score: 0,
    });
  }
  const rootId = `${c.domain}-root`;
  const answerId = `${c.domain}-answer`;
  db.entries.push({
    id: rootId,
    content: `${c.query} decision root`,
    tags: "[]",
    source: "api",
    created_at: 1000,
    vector_ids: "[]",
    recall_count: 0,
    importance_score: 0,
  });
  db.entries.push({
    id: answerId,
    content: c.answer,
    tags: "[]",
    source: "api",
    created_at: 900,
    vector_ids: "[]",
    recall_count: 0,
    importance_score: 0,
  });
  db.edges.push({
    id: `${c.domain}-edge`,
    source_id: rootId,
    target_id: answerId,
    type: "decided",
    weight: 1,
    provenance: "explicit",
    metadata: "{}",
    created_at: 1,
    updated_at: 1,
  });

  const matches = [
    ...[0, 1, 2, 3, 4].map(i => ({
      id: `${c.domain}-direct-${i}`,
      score: 0.95 - i * 0.04,
      metadata: { parentId: `${c.domain}-direct-${i}`, isUpdate: false },
    })),
    { id: rootId, score: 0.7, metadata: { parentId: rootId, isUpdate: false } },
  ];
  const env = makeTestEnv(db, {
    VECTORIZE: makeVectorizeMock({
      query: vi.fn().mockResolvedValue({ matches }),
    }),
  });
  const ctx = { waitUntil: () => undefined } as any as ExecutionContext;
  return { env, ctx, answerId };
}

describe("graph-aware recall multi-domain benchmark", () => {
  for (const c of cases) {
    it(`${c.domain}: linked causal evidence enters Recall@5 only when graph recall is requested`, async () => {
      const direct = fixture(c);
      const graph = fixture(c);

      const withoutGraph = await recallEntries(
        { query: c.query, topK: 5, hops: 0, synthesize: false },
        direct.env,
        direct.ctx,
      );
      const withGraph = await recallEntries(
        { query: c.query, topK: 5, hops: 1, synthesize: false },
        graph.env,
        graph.ctx,
      );

      expect(withoutGraph.matches.map(m => m.id)).not.toContain(direct.answerId);
      expect(withGraph.matches.map(m => m.id)).toContain(graph.answerId);
      expect(withGraph.matches[0].id).toBe(`${c.domain}-direct-0`);
      expect(withGraph.matches).toHaveLength(5);
    });
  }
});
