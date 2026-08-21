import type { Env } from "../env";
import {
  D1_MAX_BOUND_PARAMS,
  KEYWORD_MAX_TOKENS,
  VECTORIZE_GET_BY_IDS_BATCH,
  VECTORIZE_TOP_K_MULTIPLIER,
} from "../constants";
import { resolveConfig, type Config } from "../config";
import { embed } from "../lib/ai";
import { expandGraph } from "../graph/traverse";
import type { GraphNeighbor } from "../graph/types";
import { KIND_VALUES, type MemoryKind } from "../memory/kind";
import { parseTimePhrase } from "../text/temporal";
import { tokenizeQuery } from "../text/tokenize";
import { distillToRareTerms, inferQueryTags, type DistilledQuery } from "./distill";
import { synthesizeInsight } from "./insight";
import { hasStaleAsOf } from "../memory/stale";
import { cosineSim, mmrRerank, rerankWithTimeDecay, type VectorizeMatch } from "./math";
import { rrfFuse } from "./rrf";
import { computeCompoundStale } from "./compound-stale";
import { graphSeedLimit, relatedSlotLimit, scoreLinkedEvidence } from "./neighborhood";
import { queryCoverage } from "./neighborhood";
import { buildQueryProfile, embeddingInput } from "./query-profile";
import { localEvidenceOf } from "./root-candidate";
import { selectGraphRoots, type RootCandidate } from "./root-selector";
import type { KeywordRow, RecallInternalOptions, RecallMatch, RecallSearchResult } from "./types";
import { TAG_LIKE_ESCAPE, tagLikePattern } from "../memory/tag-sql";

async function keywordSearch(tokens: string[], env: Env, limit: number): Promise<KeywordRow[]> {
  if (!tokens.length) return [];
  // Capped here rather than at distillation's uncapped exits because this is
  // the only place a token count becomes SQL, and there are two such exits —
  // one of which needs nothing worse than an empty corpus to fire (#276). Query
  // order is the only ordering available on those paths: they are exactly the
  // paths where the frequencies that would rank the terms are missing.
  const terms = tokens.slice(0, KEYWORD_MAX_TOKENS);
  const where = terms.map(() => "content LIKE ?").join(" OR ");
  const { results } = await env.DB.prepare(
    `SELECT id, content, tags, source, created_at FROM entries WHERE ${where} ORDER BY created_at DESC LIMIT ?`
  ).bind(...terms.map(t => `%${t}%`), limit).all();
  return results as unknown as KeywordRow[];
}

const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function fuseDenseAndKeyword(
  denseMatches: VectorizeMatch[],
  keywordRows: KeywordRow[],
  tokens: string[],
  allowKeywordOnly: boolean,
  corpus: Pick<DistilledQuery, "df" | "total">,
  substringWeight: number
): VectorizeMatch[] {
  const denseByParent = new Map<string, VectorizeMatch>();
  for (const m of [...denseMatches].sort((a, b) => b.score - a.score)) {
    const pid = ((m.metadata as any)?.parentId ?? m.id) as string;
    if (!denseByParent.has(pid)) denseByParent.set(pid, m);
  }
  const denseRanked = [...denseByParent.keys()];

  const kwLower = keywordRows.map(r => ({ row: r, lc: r.content.toLowerCase() }));

  // IDF from the corpus-wide frequencies distillToRareTerms already computed,
  // when they cover every token; otherwise the old estimate from the fetched
  // rows. All-or-nothing rather than per-token, because the two denominators
  // (corpus size vs fetch-window size) are different scales — mixing them in
  // one weight sum would let the source of a token's IDF, not its rarity,
  // decide the ranking.
  let idf: (t: string) => number;
  if (corpus.df && corpus.total && tokens.every(t => corpus.df!.has(t))) {
    const { df, total } = corpus;
    idf = t => Math.log(1 + total / ((df.get(t) ?? 0) + 1));
  } else {
    const kwN = kwLower.length || 1;
    const kwDf = new Map(tokens.map(t => [t, kwLower.reduce((n, x) => n + (x.lc.includes(t) ? 1 : 0), 0)]));
    idf = t => Math.log(1 + kwN / ((kwDf.get(t) ?? 0) + 1));
  }

  // A token found at a word boundary earns full IDF; found only inside a longer
  // word ("cat" in "concatenate") it earns a configured fraction. Lookarounds
  // rather than \b so identifier-shaped tokens ("#149", "v1.9") keep matching —
  // \b treats their punctuation as the boundary itself.
  const boundary = new Map(tokens.map(t => [t, new RegExp(`(?<![\\w])${escapeRegExp(t)}(?![\\w])`)]));
  const tokenWeight = (lc: string, t: string) => {
    if (!lc.includes(t)) return 0;
    return boundary.get(t)!.test(lc) ? idf(t) : idf(t) * substringWeight;
  };

  const keywordRanked = kwLower
    .map(x => ({ row: x.row, weight: tokens.reduce((s, t) => s + tokenWeight(x.lc, t), 0) }))
    .filter(x => x.weight > 0 && (allowKeywordOnly || denseByParent.has(x.row.id)))
    .sort((a, b) => b.weight - a.weight || b.row.created_at - a.row.created_at || (a.row.id < b.row.id ? -1 : 1));

  const fused = rrfFuse(denseRanked, keywordRanked.map(x => ({ id: x.row.id, weight: x.weight })));
  const keywordRowById = new Map(keywordRows.map(r => [r.id, r]));

  const out: VectorizeMatch[] = [];
  for (const [pid, score] of fused) {
    const dm = denseByParent.get(pid);
    if (dm) {
      out.push({ id: dm.id, score, metadata: dm.metadata, values: dm.values });
    } else {
      const r = keywordRowById.get(pid)!;
      out.push({ id: pid, score, metadata: { parentId: pid, created_at: r.created_at, tags: JSON.parse(r.tags ?? "[]"), content: r.content, source: r.source } });
    }
  }
  return out;
}

export async function recallEntries(
  params: { query: string; topK: number; tag?: string; after?: number; before?: number; kind?: MemoryKind; hops?: number; synthesize?: boolean },
  env: Env,
  ctx: ExecutionContext,
  // Resolved once at request entry by the route/MCP caller and threaded down.
  // Optional so this stays callable without a config in tests and any future
  // internal caller; the fallback costs one KV read.
  config?: Readonly<Config>,
  internal: RecallInternalOptions = {},
): Promise<RecallSearchResult> {
  const cfg = config ?? await resolveConfig(env);
  const { query, topK } = params;
  const synthesize = params.synthesize ?? true;
  let { tag, after, before, kind } = params;
  const hops = Math.max(0, Math.min(cfg.GRAPH_MAX_HOPS, params.hops ?? cfg.DEFAULT_HOPS));
  const now = Date.now();
  let semanticUnavailable = false;

  let semanticQuery = query;
  if (after === undefined && before === undefined) {
    const parsed = parseTimePhrase(query, now);
    after = parsed.after;
    before = parsed.before;
    semanticQuery = parsed.cleanQuery;
  }
  const distilled = await distillToRareTerms(semanticQuery, env, cfg);
  const profile = buildQueryProfile(semanticQuery, distilled);
  const embedQuery = embeddingInput(profile, internal.embeddingQueryMode ?? "distilled");
  const lexicalQuery = profile.lexicalQuery;
  internal.diagnostics && (internal.diagnostics.embeddingMode = internal.embeddingQueryMode ?? "distilled");

  const tokens = profile.lexicalTokens;
  const [values, queryTags] = await Promise.all([
    embed(embedQuery, env, cfg),
    inferQueryTags(lexicalQuery, env, cfg, ctx),
  ]);

  let keywordRows: KeywordRow[] = [];
  let results: { matches: VectorizeMatch[] };
  if (tag) {
    // Escaped: a tag is user data and LIKE reads _ and % as wildcards. This is a read, so
    // the failure is over-broad results rather than the permanent rollup the same bug
    // caused in compressTag — but `?tag=%` silently defeats the filter entirely and
    // returns the whole brain, which is not a recoverable-looking answer either.
    const { results: tagRows } = await env.DB.prepare(
      `SELECT id, vector_ids, content, tags, source, created_at FROM entries WHERE tags LIKE ? ${TAG_LIKE_ESCAPE}`
    ).bind(tagLikePattern(tag)).all();
    if (!tagRows.length) return { matches: [], insight: "", semanticUnavailable };
    keywordRows = tagRows as unknown as KeywordRow[];

    const vectorIds = [...new Set(
      (tagRows as any[]).flatMap(r => JSON.parse((r.vector_ids as string) ?? "[]") as string[])
    )];
    if (!vectorIds.length) return { matches: [], insight: "", semanticUnavailable };

    const vectors: VectorizeVector[] = [];
    try {
      for (let i = 0; i < vectorIds.length; i += VECTORIZE_GET_BY_IDS_BATCH) {
        vectors.push(...await env.VECTORIZE.getByIds(vectorIds.slice(i, i + VECTORIZE_GET_BY_IDS_BATCH)));
      }
    } catch (e) {
      console.error("Vectorize getByIds failed (degrading to keyword-only):", e);
      semanticUnavailable = true;
    }

    results = {
      matches: vectors.map(v => ({
        id: v.id,
        score: cosineSim(values, v.values as number[]),
        metadata: v.metadata,
        values: v.values as number[],
      })) as VectorizeMatch[],
    };
  } else {
    const vectorizeTopK = Math.min(topK * VECTORIZE_TOP_K_MULTIPLIER, 50);
    const denseQuery = async (): Promise<{ matches: VectorizeMatch[] }> => {
      try {
        return await env.VECTORIZE.query(values, { topK: vectorizeTopK, returnMetadata: "all", returnValues: true });
      } catch (e) {
        console.error("Vectorize query failed (degrading to keyword-only):", e);
        semanticUnavailable = true;
        return { matches: [] as VectorizeMatch[] };
      }
    };
    const [denseResults, kwRows] = await Promise.all([denseQuery(), keywordSearch(tokens, env, cfg.KEYWORD_CANDIDATE_LIMIT)]);
    results = denseResults;
    keywordRows = kwRows;

    // Governed by its own threshold, not the write-path duplicate flag: the two
    // shared a constant until #245, so retuning duplicate detection silently
    // retuned recall widening.
    if (!semanticUnavailable && results.matches.length && results.matches[0].score < cfg.RECALL_WIDEN_THRESHOLD) {
      try {
        results = await env.VECTORIZE.query(values, { topK: 50, returnMetadata: "all", returnValues: true });
      } catch (e) {
        console.error("Vectorize widen-query failed (non-fatal, keeping narrow results):", e);
      }
    }
  }

  const fusedMatches = fuseDenseAndKeyword(results.matches as VectorizeMatch[], keywordRows, tokens, !tag || semanticUnavailable, distilled, cfg.SUBSTRING_MATCH_WEIGHT);
  if (!fusedMatches.length) return { matches: [], insight: "", semanticUnavailable };

  const candidateIds = [...new Set(fusedMatches.map(m => (m.metadata as any)?.parentId ?? m.id))] as string[];
  internal.diagnostics && (internal.diagnostics.fusedIds = candidateIds);
  const rcRows: { id: string; content: string; recall_count: number; importance_score: number; contradiction_wins: number; contradiction_losses: number; tags: string }[] = [];
  for (let i = 0; i < candidateIds.length; i += D1_MAX_BOUND_PARAMS) {
    const batch = candidateIds.slice(i, i + D1_MAX_BOUND_PARAMS);
    const rcPlaceholders = batch.map(() => "?").join(", ");
    const { results: rows } = await env.DB.prepare(
      `SELECT id, content, recall_count, importance_score, contradiction_wins, contradiction_losses, tags FROM entries WHERE id IN (${rcPlaceholders})`
    ).bind(...batch).all() as { results: { id: string; content: string; recall_count: number; importance_score: number; contradiction_wins: number; contradiction_losses: number; tags: string }[] };
    rcRows.push(...rows);
  }
  const recallCounts = new Map(rcRows.map(r => [r.id, r.recall_count ?? 0]));
  const importanceScores = new Map(rcRows.map(r => [r.id, r.importance_score ?? 0]));
  const contradictionWins = new Map(rcRows.map(r => [r.id, r.contradiction_wins ?? 0]));
  const contradictionLosses = new Map(rcRows.map(r => [r.id, r.contradiction_losses ?? 0]));
  const d1Tags = new Map(rcRows.map(r => [r.id, JSON.parse(r.tags ?? "[]") as string[]]));
  const candidateContent = new Map(rcRows.map(r => [r.id, r.content ?? ""]));

  const directReranked = rerankWithTimeDecay(fusedMatches, recallCounts, importanceScores, queryTags, contradictionWins, contradictionLosses, d1Tags, cfg);
  const rootReranked = rerankWithTimeDecay(fusedMatches, recallCounts, importanceScores, queryTags, contradictionWins, contradictionLosses, d1Tags, cfg, { useRecallFrequency: false });
  internal.diagnostics && (internal.diagnostics.candidateIds = directReranked.map(m => ((m.metadata as any)?.parentId ?? m.id) as string));

  const seen = new Set<string>();
  const dedupedAll = directReranked.filter((m) => {
    const parentId = (m.metadata as any)?.parentId ?? m.id;
    if (seen.has(parentId)) return false;
    seen.add(parentId);
    return true;
  });
  const directCandidates = mmrRerank(dedupedAll, cfg.MMR_LAMBDA, topK);

  if (!directCandidates.length) return { matches: [], insight: "", semanticUnavailable };

  const directParentIds = directCandidates.map((m) => (m.metadata as any)?.parentId ?? m.id);
  const rootSeen = new Set<string>();
  const rootCandidates: RootCandidate[] = rootReranked.flatMap(match => {
    const parentId = ((match.metadata as any)?.parentId ?? match.id) as string;
    if (rootSeen.has(parentId)) return [];
    rootSeen.add(parentId);
    const tags = d1Tags.get(parentId) ?? [];
    const localEvidence = localEvidenceOf(match, candidateContent.get(parentId) ?? "", tokens);
    const tagAlignment = queryTags.length ? tags.filter(value => queryTags.includes(value)).length / queryTags.length : 0;
    const episodicAlignment = ["causal", "chronology"].includes(profile.intent) && tags.includes("kind:episodic") ? 1 : 0;
    const authorityAlignment = ["current", "direct"].includes(profile.intent) && tags.includes("status:canonical") ? 1 : 0;
    return [{ ...match, parentId, rootScore: match.score, localEvidence, tags,
      lexicalCoverage: queryCoverage(localEvidence, tokens, distilled).score,
      metadataAlignment: Math.min(1, .6 * tagAlignment + .2 * episodicAlignment + .2 * authorityAlignment) }];
  });
  const selectedRoots = hops > 0
    ? selectGraphRoots(rootCandidates, graphSeedLimit(topK, rootCandidates.length), cfg.MMR_LAMBDA)
    : [];
  const graphSeedIds = selectedRoots.map(x => x.candidate.parentId);
  if (internal.diagnostics) {
    internal.diagnostics.rootSelections = selectedRoots.map(x => ({ id: x.candidate.parentId, selectedBy: x.selectedBy }));
    internal.diagnostics.rejections = [];
  }

  let expanded: GraphNeighbor[] = [];
  if (hops > 0) {
    expanded = await expandGraph(graphSeedIds, { hops }, env, cfg);
  }
  internal.diagnostics && (internal.diagnostics.expandedIds = expanded.map(x => x.id));

  // The graph view can include up to 50 roots and 50 expanded nodes in addition
  // to direct candidates. Keep the union unique and chunked: with a topK above
  // the public route's cap this can span multiple D1 statements, and time
  // filters consume bindings in every statement.
  const allParentIds = [...new Set([
    ...directParentIds,
    ...graphSeedIds,
    ...expanded.map(e => e.id),
  ])];
  let d1Filters = ` AND tags NOT LIKE '%"auto-pattern"%' AND tags NOT LIKE '%"auto-insight"%' AND tags NOT LIKE '%"status:deprecated"%'`;
  const filterBindings: number[] = [];
  if (kind && (KIND_VALUES as readonly string[]).includes(kind)) {
    d1Filters += ` AND tags LIKE '%"kind:${kind}"%'`;
  }
  if (after !== undefined) { d1Filters += ` AND created_at >= ?`; filterBindings.push(after); }
  if (before !== undefined) { d1Filters += ` AND created_at <= ?`; filterBindings.push(before); }
  const d1Rows: Record<string, any>[] = [];
  const idBatchSize = D1_MAX_BOUND_PARAMS - filterBindings.length;
  for (let i = 0; i < allParentIds.length; i += idBatchSize) {
    const batch = allParentIds.slice(i, i + idBatchSize);
    const placeholders = batch.map(() => "?").join(", ");
    const { results } = await env.DB.prepare(
      `SELECT id, content, tags, source, created_at, updated_at FROM entries WHERE id IN (${placeholders})${d1Filters}`
    ).bind(...batch, ...filterBindings).all() as { results: Record<string, any>[] };
    d1Rows.push(...results);
  }

  const d1Map = new Map(d1Rows.map((r) => [r.id as string, r]));

  const directMatches: RecallMatch[] = directCandidates.flatMap((m) => {
    const meta = m.metadata as Record<string, any>;
    const parentId = (meta?.parentId ?? m.id) as string;
    const row = d1Map.get(parentId);
    if (!row) return [];
    return [{
      id: parentId,
      content: row.content as string,
      score: m.score,
      createdAt: row.created_at as number,
      updatedAt: (row.updated_at as number | null) ?? (row.created_at as number),
      tags: JSON.parse(row.tags ?? "[]"),
      source: row.source as string,
      isUpdate: !!meta?.isUpdate,
      hop: 0,
      staleAsOf: hasStaleAsOf(JSON.parse(row.tags ?? "[]")),
    }];
  }).sort((a, b) => b.score - a.score);

  const maximumRootScore = Math.max(...selectedRoots.map(x => x.candidate.rootScore), 1);
  const rootById = new Map(selectedRoots.map(x => [x.candidate.parentId, x.candidate]));
  const rootIdByNode = new Map(selectedRoots.map(x => [x.candidate.parentId, x.candidate.parentId]));
  const fallbackRootScore = directCandidates.at(-1)?.score ?? 0;
  for (const e of expanded) {
    rootIdByNode.set(e.id, rootIdByNode.get(e.viaFrom) ?? e.viaFrom);
  }
  const relatedLimit = relatedSlotLimit(topK);
  const replacement = directMatches[topK - relatedLimit];
  const replacementCoverage = replacement ? queryCoverage(replacement.content, tokens, distilled).score : 0;
  const expandedMatches: { match: RecallMatch; eligible: boolean }[] = expanded.flatMap((e) => {
    const row = d1Map.get(e.id);
    if (!row) return [];
    const root = rootById.get(rootIdByNode.get(e.id) ?? "");
    const rootScore = root ? root.rootScore / maximumRootScore : fallbackRootScore;
    const evidence = scoreLinkedEvidence({
      parentScore: rootScore,
      parentContent: root?.localEvidence ?? "",
      content: row.content as string,
      queryTokens: tokens,
      corpus: distilled,
      hop: e.hop,
      edgeWeight: e.viaWeight,
      provenance: e.viaProvenance,
      hopDecay: cfg.GRAPH_HOP_DECAY,
      replacementCoverage,
      intent: profile.intent,
      edgeType: e.viaType,
    });
    if (!evidence.eligible) internal.diagnostics?.rejections?.push({ id: e.id, reason: evidence.rejection ?? "weak-neighborhood" });
    return [{
      eligible: evidence.eligible,
      match: {
        id: e.id,
        content: row.content as string,
        score: evidence.score,
        createdAt: row.created_at as number,
        updatedAt: (row.updated_at as number | null) ?? (row.created_at as number),
        tags: JSON.parse(row.tags ?? "[]"),
        source: row.source as string,
        isUpdate: false,
        hop: e.hop,
        staleAsOf: hasStaleAsOf(JSON.parse(row.tags ?? "[]")),
        viaProvenance: e.viaProvenance,
        viaType: e.viaType,
        viaLinkedAt: e.viaLinkedAt,
        viaFrom: e.viaFrom,
      },
    }];
  });

  const sortedExpanded = expandedMatches
    .sort((a, b) => b.match.score - a.match.score || a.match.id.localeCompare(b.match.id));
  const selectedRelated = sortedExpanded
    .filter(e => e.eligible)
    .slice(0, relatedLimit)
    .map(e => e.match);
  if (internal.diagnostics) internal.diagnostics.selectedRelatedIds = selectedRelated.map(x => x.id);
  const selectedDirect = directMatches.slice(0, topK - selectedRelated.length);
  const matches: RecallMatch[] = [...selectedDirect, ...selectedRelated];

  const presentedDirectIds = new Set(selectedDirect.map(m => m.id));
  ctx.waitUntil(
    Promise.all(
      [...presentedDirectIds].map(id =>
        env.DB.prepare(`UPDATE entries SET recall_count = recall_count + 1 WHERE id = ?`).bind(id).run()
      )
    ).catch(e => console.error("recall_count update failed (non-fatal):", e))
  );

  const maxScore = matches.reduce((mx, m) => Math.max(mx, m.score), 0);
  if (maxScore > 0) for (const m of matches) m.score = m.score / maxScore;

  const compoundStale = computeCompoundStale(matches);

  const insight = synthesize && matches.length > 1
    ? await synthesizeInsight(lexicalQuery, matches.map(m => ({ id: m.id, content: m.content })), env, cfg)
    : "";

  return { matches, insight, semanticUnavailable, queryUsed: lexicalQuery, queryTokens: tokens, compoundStale };
}
