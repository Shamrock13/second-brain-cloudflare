import { mmrRerank, type VectorizeMatch } from "./math";

export type RootView = "semantic" | "lexical" | "metadata" | "diversity";

export interface RootCandidate extends VectorizeMatch {
  parentId: string;
  rootScore: number;
  localEvidence: string;
  tags: string[];
  lexicalCoverage: number;
  metadataAlignment: number;
}

export interface SelectedRoot {
  candidate: RootCandidate;
  selectedBy: RootView;
}

const VIEW_SHARE: Record<RootView, number> = {
  semantic: 0.4,
  lexical: 0.3,
  metadata: 0.15,
  diversity: 0.15,
};

const VIEWS: RootView[] = ["semantic", "lexical", "metadata", "diversity"];

function compareBy(
  score: (candidate: RootCandidate) => number,
  fallback?: (candidate: RootCandidate) => number,
): (a: RootCandidate, b: RootCandidate) => number {
  return (a, b) => score(b) - score(a)
    || (fallback ? fallback(b) - fallback(a) : 0)
    || (a.parentId < b.parentId ? -1 : a.parentId > b.parentId ? 1 : 0);
}

function allocateQuotas(available: RootView[], limit: number): Map<RootView, number> {
  const quotas = new Map<RootView, number>();
  for (const view of available) {
    quotas.set(view, Math.max(1, Math.floor(limit * VIEW_SHARE[view])));
  }

  let remaining = limit - [...quotas.values()].reduce((sum, quota) => sum + quota, 0);
  const byRemainder = available.slice().sort((a, b) => {
    const remainder = (limit * VIEW_SHARE[b]) % 1 - (limit * VIEW_SHARE[a]) % 1;
    return remainder || VIEWS.indexOf(a) - VIEWS.indexOf(b);
  });
  for (let index = 0; remaining > 0; index = (index + 1) % byRemainder.length, remaining--) {
    const view = byRemainder[index];
    quotas.set(view, (quotas.get(view) ?? 0) + 1);
  }
  return quotas;
}

export function selectGraphRoots(
  candidates: RootCandidate[],
  limit: number,
  mmrLambda: number,
): SelectedRoot[] {
  if (limit <= 0 || candidates.length === 0) return [];

  const semantic = candidates.slice().sort(compareBy(candidate => candidate.rootScore));
  const views: Record<RootView, RootCandidate[]> = {
    semantic,
    lexical: candidates.slice().sort(compareBy(candidate => candidate.lexicalCoverage, candidate => candidate.rootScore)),
    metadata: candidates.slice().sort(compareBy(candidate => candidate.metadataAlignment, candidate => candidate.rootScore)),
    diversity: mmrRerank(semantic, mmrLambda, semantic.length),
  };
  const available = VIEWS.filter(view => views[view].length > 0).slice(0, limit);
  const quotas = allocateQuotas(available, limit);
  const selected: SelectedRoot[] = [];
  const selectedIds = new Set<string>();

  for (const view of available) {
    const quota = quotas.get(view) ?? 0;
    for (const candidate of views[view]) {
      if (selected.length >= limit || selected.filter(root => root.selectedBy === view).length >= quota) break;
      if (selectedIds.has(candidate.parentId)) continue;
      selected.push({ candidate, selectedBy: view });
      selectedIds.add(candidate.parentId);
    }
  }

  for (const candidate of semantic) {
    if (selected.length >= limit) break;
    if (selectedIds.has(candidate.parentId)) continue;
    selected.push({ candidate, selectedBy: "semantic" });
    selectedIds.add(candidate.parentId);
  }
  return selected;
}
