/**
 * WASM Backend — Search & Query Operations
 *
 * Extracted from wasm.ts to keep file NLOC under 500.
 * All functions receive a WasmContext to access collections and the WASM module.
 */

import type {
  SearchOptions,
  SearchResult,
  MultiQuerySearchOptions,
  QueryOptions,
  QueryApiResponse,
  FusionParams,
} from '../types';
import type { FilterInput } from '../filter';
import { NotFoundError, VelesDBError } from '../types';
import { wasmNotSupported } from './shared';
import {
  isSet,
  requireWasmCapability,
  requireWasmFieldsListed,
  requireWasmFilterSupport,
} from './wasm-capability-guards';
import { sparseHits } from './wasm-sparse';
import type {
  WasmContext,
  WasmDenseResult,
  WasmSparseResult,
  WasmFilteredResult,
  WasmHybridResult,
  WasmSearchResultItem,
} from './wasm-types';

// ---------------------------------------------------------------------------
// Dense search (optionally with sparse/hybrid/filter)
// ---------------------------------------------------------------------------

function searchSparseOnly(
  ctx: WasmContext,
  collection: ReturnType<WasmContext['getCollection']>,
  indices: number[],
  values: number[],
  k: number
): SearchResult[] {
  return sparseHits(collection!.store, collection!.sparseIds, indices, values, k).map(
    ([id, score]) => ({
      id: String(id),
      score,
      payload: collection!.payloads.get(ctx.canonicalPayloadKeyFromResultId(id)),
    })
  );
}

function searchHybridFusion(
  ctx: WasmContext,
  collection: ReturnType<WasmContext['getCollection']>,
  queryVector: Float32Array,
  indices: number[],
  values: number[],
  k: number
): SearchResult[] {
  const denseResults: WasmDenseResult[] = collection!.store.search(queryVector, k);
  const denseForFuse: Array<[number, number]> = denseResults.map(
    ([id, score]) => [Number(id), score]
  );
  const sparseForFuse = sparseHits(collection!.store, collection!.sparseIds, indices, values, k);

  const fused: WasmSparseResult[] = ctx.wasmModule.hybrid_search_fuse(
    denseForFuse, sparseForFuse, 60, k
  );

  return fused.slice(0, k).map(r => ({
    id: String(r.doc_id),
    score: r.score,
    payload: collection!.payloads.get(ctx.canonicalPayloadKeyFromResultId(r.doc_id)),
  }));
}

function searchWithFilter(
  ctx: WasmContext,
  collection: ReturnType<WasmContext['getCollection']>,
  queryVector: Float32Array,
  k: number,
  filter: FilterInput
): SearchResult[] {
  const results: WasmFilteredResult[] = collection!.store.search_with_filter(
    queryVector, k, filter
  );

  return results.map(r => ({
    id: String(r.id),
    score: r.score,
    payload: r.payload || collection!.payloads.get(ctx.canonicalPayloadKeyFromResultId(r.id)),
  }));
}

function searchDenseOnly(
  ctx: WasmContext,
  collection: ReturnType<WasmContext['getCollection']>,
  queryVector: Float32Array,
  k: number
): SearchResult[] {
  const rawResults: WasmDenseResult[] = collection!.store.search(queryVector, k);

  return rawResults.map(([id, score]) => {
    const result: SearchResult = { id: String(id), score };
    const payload = collection!.payloads.get(ctx.canonicalPayloadKeyFromResultId(id));
    if (payload) {
      result.payload = payload;
    }
    return result;
  });
}

/**
 * Refuse the `SearchOptions` this backend cannot apply. `quality` is
 * accepted and has nothing to tune: WASM search scans every stored vector,
 * with no graph index whose recall a preset would trade for speed.
 */
function refuseUnhonouredSearchOptions(options: SearchOptions | undefined): void {
  if (options?.includeVectors === true) {
    requireWasmCapability('includeVectors', 'search with includeVectors: true');
  }
  if (isSet(options?.sparseIndexName)) {
    requireWasmCapability('namedSparseIndexes', 'search with a sparseIndexName');
  }
  if (options?.sparseVector) {
    requireWasmCapability('sparseSearch', 'search with a sparseVector');
    requireWasmFilterSupport('sparseSearch', options.filter);
  } else {
    requireWasmFilterSupport('search', options?.filter);
  }
}

// ---------------------------------------------------------------------------
// Exported search functions
// ---------------------------------------------------------------------------

export async function wasmSearch(
  ctx: WasmContext,
  collectionName: string,
  query: number[] | Float32Array,
  options?: SearchOptions
): Promise<SearchResult[]> {
  const collection = ctx.getCollection(collectionName);
  if (!collection) {
    throw new NotFoundError(`Collection '${collectionName}'`);
  }

  const queryVector = query instanceof Float32Array ? query : new Float32Array(query);
  if (queryVector.length !== collection.config.dimension) {
    throw new VelesDBError(
      `Query dimension mismatch: expected ${collection.config.dimension}, got ${queryVector.length}`,
      'DIMENSION_MISMATCH'
    );
  }

  const k = options?.k ?? 10;
  refuseUnhonouredSearchOptions(options);

  if (options?.sparseVector) {
    const { indices, values } = ctx.sparseVectorToArrays(options.sparseVector);
    const hasDense = queryVector.length > 0
      && collection.config.dimension !== undefined
      && collection.config.dimension > 0;

    return hasDense
      ? searchHybridFusion(ctx, collection, queryVector, indices, values, k)
      : searchSparseOnly(ctx, collection, indices, values, k);
  }

  if (options?.filter) {
    return searchWithFilter(ctx, collection, queryVector, k, options.filter);
  }

  return searchDenseOnly(ctx, collection, queryVector, k);
}

export async function wasmSearchBatch(
  ctx: WasmContext,
  collectionName: string,
  searches: Array<{
    vector: number[] | Float32Array;
    k?: number;
    filter?: FilterInput;
    /**
     * Search quality preset, forwarded to `wasmSearch`. It has nothing to
     * tune there: WASM search scans every stored vector, which meets the
     * recall of any preset.
     */
    quality?: import('../types').SearchQuality;
  }>
): Promise<SearchResult[][]> {
  for (const s of searches) {
    requireWasmFilterSupport('searchBatch', s.filter);
  }
  const results: SearchResult[][] = [];
  for (const s of searches) {
    results.push(
      await wasmSearch(ctx, collectionName, s.vector, {
        k: s.k,
        filter: s.filter,
        quality: s.quality,
      })
    );
  }
  return results;
}

// ---------------------------------------------------------------------------
// Text / Hybrid search
// ---------------------------------------------------------------------------

/** Map a WASM search result (tuple or object) to a SearchResult. */
function mapWasmResult(
  ctx: WasmContext,
  collection: ReturnType<WasmContext['getCollection']>,
  r: WasmSearchResultItem
): SearchResult {
  if (Array.isArray(r)) {
    const key = ctx.canonicalPayloadKeyFromResultId(r[0]);
    return { id: String(r[0]), score: r[1], payload: collection!.payloads.get(key) };
  }
  const key = ctx.canonicalPayloadKeyFromResultId(r.id);
  return { id: String(r.id), score: r.score, payload: r.payload ?? collection!.payloads.get(key) };
}

export async function wasmTextSearch(
  ctx: WasmContext,
  collectionName: string,
  query: string,
  options?: { k?: number; filter?: FilterInput }
): Promise<SearchResult[]> {
  const collection = ctx.getCollection(collectionName);
  if (!collection) {
    throw new NotFoundError(`Collection '${collectionName}'`);
  }
  requireWasmFilterSupport('textSearch', options?.filter);
  const k = options?.k ?? 10;
  // The binding's third argument names one payload field to match. It is
  // not a filter, which is why a filter is refused above.
  const raw: WasmSearchResultItem[] = collection.store.text_search(query, k, null);
  return raw.map(r => mapWasmResult(ctx, collection, r));
}

export async function wasmHybridSearch(
  ctx: WasmContext,
  collectionName: string,
  vector: number[] | Float32Array,
  textQuery: string,
  options?: { k?: number; vectorWeight?: number; filter?: FilterInput }
): Promise<SearchResult[]> {
  const collection = ctx.getCollection(collectionName);
  if (!collection) {
    throw new NotFoundError(`Collection '${collectionName}'`);
  }
  requireWasmFilterSupport('hybridSearch', options?.filter);
  const queryVector = vector instanceof Float32Array ? vector : new Float32Array(vector);
  const k = options?.k ?? 10;
  const vectorWeight = options?.vectorWeight ?? 0.5;
  const raw: WasmHybridResult[] = collection.store.hybrid_search(
    queryVector, textQuery, k, vectorWeight
  );
  return raw.map(r => {
    const key = ctx.canonicalPayloadKeyFromResultId(r.id);
    return { id: String(r.id), score: r.score, payload: r.payload ?? collection.payloads.get(key) };
  });
}

// ---------------------------------------------------------------------------
// Multi-query search
// ---------------------------------------------------------------------------

/** The weighted-fusion fields velesdb-wasm takes as one `[avg, max, hit]` argument. */
const WEIGHTED_TRIPLE = ['avgWeight', 'maxWeight', 'hitWeight'] as const;

/**
 * How far from 1.0 a weighted triple may sum: core's `validate_weight_sum`
 * (`crates/velesdb-core/src/fusion/strategy.rs`). The binding checks it
 * again, but reports a failure as a bare string instead of an error.
 */
const WEIGHTED_SUM_TOLERANCE = 0.001;

/**
 * Refuse, as core does, a weighted triple with a negative or non-finite
 * weight, or one that does not sum to 1.0.
 */
function validateWeightedTriple(weights: readonly number[]): void {
  const sum = weights.reduce((total, weight) => total + weight, 0);
  const invalid = weights.some((weight) => !Number.isFinite(weight) || weight < 0);
  if (invalid || Math.abs(sum - 1) > WEIGHTED_SUM_TOLERANCE) {
    throw new VelesDBError(
      'multiQuerySearch weighted fusion: avgWeight, maxWeight and hitWeight must be ' +
        `finite, non-negative and sum to 1.0 within ${WEIGHTED_SUM_TOLERANCE}; ` +
        `got ${weights.join(', ')}`,
      'BAD_REQUEST'
    );
  }
}

/**
 * Translate `fusionParams` into velesdb-wasm's `multi_query_search`
 * arguments, refusing what the binding cannot apply.
 *
 * The three weighted-fusion weights travel as one argument, and the binding
 * applies core's defaults only when that argument is absent. A partial
 * triple is therefore refused rather than completed with guessed values, and
 * a complete one is checked against core's rule.
 */
function wasmFusionArgs(params: FusionParams | undefined): {
  rrfK: number;
  weights: Float32Array | null;
} {
  requireWasmFieldsListed('multiQueryFusionParams', 'multiQuerySearch fusionParams', params);
  const rrfK = params?.k ?? 60;
  const weights = WEIGHTED_TRIPLE.map((name) => params?.[name]).filter(isSet);
  if (weights.length === 0) {
    return { rrfK, weights: null };
  }
  if (weights.length !== WEIGHTED_TRIPLE.length) {
    wasmNotSupported(
      'multiQuerySearch with only some of fusionParams avgWeight, maxWeight and ' +
        'hitWeight (velesdb-wasm takes the three together)'
    );
  }
  validateWeightedTriple(weights);
  return { rrfK, weights: new Float32Array(weights) };
}

export async function wasmMultiQuerySearch(
  ctx: WasmContext,
  collectionName: string,
  vectors: Array<number[] | Float32Array>,
  options?: MultiQuerySearchOptions
): Promise<SearchResult[]> {
  const collection = ctx.getCollection(collectionName);
  if (!collection) {
    throw new NotFoundError(`Collection '${collectionName}'`);
  }
  requireWasmFilterSupport('multiQuerySearch', options?.filter);
  const { rrfK, weights } = wasmFusionArgs(options?.fusionParams);
  if (vectors.length === 0) {
    return [];
  }

  const numVectors = vectors.length;
  const dimension = collection.config.dimension ?? 0;
  const flat = new Float32Array(numVectors * dimension);
  vectors.forEach((vector, idx) => {
    const src = vector instanceof Float32Array ? vector : new Float32Array(vector);
    flat.set(src, idx * dimension);
  });

  const strategy = options?.fusion ?? 'rrf';
  const raw: WasmSearchResultItem[] = collection.store.multi_query_search(
    flat,
    numVectors,
    options?.k ?? 10,
    strategy,
    rrfK,
    weights
  );

  return raw.map(r => mapWasmResult(ctx, collection, r));
}

// ---------------------------------------------------------------------------
// Query (VelesQL over WASM)
// ---------------------------------------------------------------------------

/**
 * The only VelesQL shape the WASM backend can execute faithfully: a pure
 * top-k NEAR scan — `SELECT * FROM <collection> WHERE vector NEAR $param
 * [LIMIT n]` (case-insensitive, optional trailing semicolon).
 *
 * `vector` is the literal keyword from the grammar
 * (`vector_search = { ^"vector" ~ ^"NEAR" ~ vector_value }`), not a column
 * name — any other identifier left of NEAR is a parse error on
 * velesdb-server, so it must be rejected here too or the query would work
 * in WASM and break on REST.
 *
 * `VectorStore.query()` is a brute-force k-NN that evaluates no other
 * clause. Anything else (WHERE predicates, JOIN, GROUP BY, MATCH, set
 * operations, FUSION, …) must be rejected loudly instead of silently
 * dropping clauses and returning unfiltered neighbours.
 */
const PURE_NEAR_QUERY =
  /^\s*select\s+\*\s+from\s+([a-z_]\w*)\s+where\s+vector\s+near\s+\$([a-z_]\w*)\s*(?:limit\s+(\d+))?\s*;?\s*$/i;

interface PureNearQuery {
  /** Collection named in the FROM clause. */
  from: string;
  /** Name of the `$param` holding the query embedding. */
  param: string;
  /** `LIMIT n` value when present. */
  limit?: number;
}

/** Parse `queryString` against the pure-NEAR shape or throw `NOT_SUPPORTED`. */
function parsePureNearQuery(queryString: string): PureNearQuery {
  const match = PURE_NEAR_QUERY.exec(queryString);
  if (!match) {
    throw new VelesDBError(
      'The WASM backend only executes pure top-k NEAR queries of the form ' +
        '"SELECT * FROM <collection> WHERE vector NEAR $param [LIMIT n]". ' +
        'WHERE predicates, JOIN, GROUP BY, MATCH, set operations and FUSION ' +
        'are not evaluated in WASM — use the REST backend (velesdb-server) ' +
        `for full VelesQL. Received: ${queryString}`,
      'NOT_SUPPORTED'
    );
  }
  const parsed: PureNearQuery = { from: match[1]!, param: match[2]! };
  if (match[3] !== undefined) {
    parsed.limit = Number(match[3]);
  }
  return parsed;
}

/** Resolve top-k: `LIMIT` from the query wins, then `params.k`, then 10. */
function resolveQueryK(limit: number | undefined, requestedK: unknown): number {
  if (limit !== undefined) {
    return limit;
  }
  return typeof requestedK === 'number' && Number.isInteger(requestedK) && requestedK > 0
    ? requestedK
    : 10;
}

export async function wasmQuery(
  ctx: WasmContext,
  collectionName: string,
  queryString: string,
  params?: Record<string, unknown>,
  options?: QueryOptions
): Promise<QueryApiResponse> {
  const collection = ctx.getCollection(collectionName);
  if (!collection) {
    throw new NotFoundError(`Collection '${collectionName}'`);
  }
  requireWasmFieldsListed('queryOptions', 'query', options);
  const parsed = parsePureNearQuery(queryString);
  if (parsed.from !== collectionName) {
    throw new VelesDBError(
      `Query targets collection '${parsed.from}' but was executed against '${collectionName}'.`,
      'BAD_REQUEST'
    );
  }
  const paramsVector = params?.[parsed.param];
  if (!Array.isArray(paramsVector) && !(paramsVector instanceof Float32Array)) {
    throw new VelesDBError(
      `WASM query() expects params.${parsed.param} to contain the query embedding vector.`,
      'BAD_REQUEST'
    );
  }
  const k = resolveQueryK(parsed.limit, params?.k);
  const raw: Record<string, unknown>[] = collection.store.query(
    paramsVector instanceof Float32Array ? paramsVector : new Float32Array(paramsVector),
    k
  );

  return {
    results: raw,
    stats: {
      executionTimeMs: 0,
      strategy: 'wasm-query',
      scannedNodes: raw.length,
    },
  };
}
