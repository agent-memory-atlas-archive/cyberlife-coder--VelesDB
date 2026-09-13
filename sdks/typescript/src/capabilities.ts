/**
 * VelesDB Backend Capability Map
 *
 * Static, per-backend description of which features the currently
 * connected backend supports. Callers use this to gracefully degrade
 * their UI / plan / workflow when a feature is not available instead
 * of catching a runtime `NOT_SUPPORTED` error after the fact.
 *
 * The map is **frozen at backend construction** — it does not round-
 * trip to the server. The REST map assumes a `velesdb-server` of the
 * same minor version; if the server does not ship a given feature,
 * the individual call will still surface a typed `VelesError` at
 * runtime.
 *
 * @example
 * ```typescript
 * import { VelesDB } from '@wiscale/velesdb-sdk';
 *
 * const db = new VelesDB({ backend: 'wasm' });
 * await db.init();
 *
 * if (db.capabilities().graphTraversal) {
 *   await db.traverseGraph('kg', { source: 1, direction: 'out' });
 * } else {
 *   // fall back to REST or a pure in-memory traversal
 * }
 * ```
 *
 * @packageDocumentation
 */

import type { FusionParamName } from './types';

/**
 * Search entry points that accept a `filter`. `'sparseSearch'` is `search`
 * called with a `sparseVector`.
 */
export type FilteredSearchOperation =
  | 'search'
  | 'sparseSearch'
  | 'textSearch'
  | 'hybridSearch'
  | 'multiQuerySearch';

/**
 * Capability map surfaced by `VelesDB.capabilities()`.
 *
 * Feature fields are `boolean` so that callers can write
 * `if (caps.feature) { ... }` without `?.` chaining; list fields name
 * the values the backend honours. A missing backend must still expose
 * the full set of keys with `false` or empty values — we prefer
 * explicit "unsupported" over "unknown".
 */
export interface CapabilityMap {
  /** Dense vector similarity search (`search`, `searchBatch`). */
  vectorSearch: boolean;
  /** BM25 full-text search (`textSearch`). */
  textSearch: boolean;
  /** Combined dense + BM25 search (`hybridSearch`). */
  hybridSearch: boolean;
  /** Multi-query fusion search (`multiQuerySearch`). */
  multiQuerySearch: boolean;
  /** Sparse vector search: `search` with a `sparseVector`, alone or fused with the dense query. */
  sparseSearch: boolean;
  /**
   * Search entry points whose `filter` the backend applies. A backend that
   * cannot apply a filter to one of them refuses it with `NOT_SUPPORTED`
   * rather than returning rows the filter excludes.
   */
  filteredSearch: readonly FilteredSearchOperation[];
  /**
   * `multiQuerySearch` `fusionParams` fields the backend applies. A field
   * not listed is refused with `NOT_SUPPORTED` rather than ignored.
   */
  multiQueryFusionParams: readonly FusionParamName[];
  /** Named sparse indexes: `search({ sparseIndexName })` and `sparseSearchNamed`. */
  namedSparseIndexes: boolean;
  /** `search({ includeVectors: true })` returns each hit's vector. */
  includeVectors: boolean;
  /** ID-and-score searches that skip payloads (`searchIds`, `multiQuerySearchIds`). */
  idOnlySearch: boolean;
  /** Cursor-based scroll pagination over a collection (`scroll`). */
  scroll: boolean;
  /** Knowledge graph edge CRUD + traversal (`addEdge`, `traverseGraph`, `traverseParallel`, `getNodeDegree`). */
  graphTraversal: boolean;
  /** Secondary property indexes (`createIndex`, `listIndexes`, `hasIndex`, `dropIndex`). */
  secondaryIndexes: boolean;
  /** Agent Memory SDK (semantic, episodic, procedural). */
  agentMemory: boolean;
  /** Enable the bounded streaming-ingestion channel (`enableStreaming`). */
  enableStreaming: boolean;
  /** Streaming insert with backpressure (`streamInsert`). */
  streamInsert: boolean;
  /** Product quantization training (`trainPq`). */
  pqTraining: boolean;
  /** VelesQL multi-model query + EXPLAIN (`query`, `queryExplain`). */
  velesqlQuery: boolean;
  /** Collection introspection endpoints (`collectionSanity`, `getCollectionStats`, `analyzeCollection`, `getCollectionConfig`). */
  collectionIntrospection: boolean;
  /**
   * `USING FUSION(strategy='...')` strategies the backend's query path
   * accepts. Empty when `velesqlQuery` is `false`. The core SQL parser
   * accepts `rrf`, `weighted`, `maximum`, `rsf`, `average`.
   */
  velesqlFusionStrategies: readonly string[];
  /**
   * `MATCH (...) RETURN ... ORDER BY ... [LIMIT n]` is honored end-to-end
   * (sorted, then limited) by the backend's query path.
   */
  velesqlMatchOrderBy: boolean;
  /**
   * `ALTER COLLECTION <name> SET(...)` is supported via the typed
   * {@link VelesDB.alterCollection} / {@link VelesDB.setAutoReindex} helpers.
   */
  velesqlAlterCollection: boolean;
}

/**
 * Capability map for the REST backend — assumes a server of the
 * same minor version as the SDK. Every feature the SDK wraps is
 * advertised; individual endpoints may still surface a typed
 * `VelesError` at runtime if the server was built with a feature
 * flag disabled.
 */
export const REST_CAPABILITIES: Readonly<CapabilityMap> = Object.freeze({
  vectorSearch: true,
  textSearch: true,
  hybridSearch: true,
  multiQuerySearch: true,
  sparseSearch: true,
  filteredSearch: Object.freeze<FilteredSearchOperation[]>([
    'search',
    'sparseSearch',
    'textSearch',
    'hybridSearch',
    'multiQuerySearch',
  ]),
  multiQueryFusionParams: Object.freeze<FusionParamName[]>([
    'k',
    'avgWeight',
    'maxWeight',
    'hitWeight',
    'denseWeight',
    'sparseWeight',
  ]),
  namedSparseIndexes: true,
  includeVectors: true,
  idOnlySearch: true,
  scroll: true,
  graphTraversal: true,
  secondaryIndexes: true,
  agentMemory: true,
  enableStreaming: true,
  streamInsert: true,
  pqTraining: true,
  velesqlQuery: true,
  collectionIntrospection: true,
  velesqlFusionStrategies: Object.freeze(['rrf', 'weighted', 'maximum', 'rsf', 'average']),
  velesqlMatchOrderBy: true,
  velesqlAlterCollection: true,
});

/**
 * Capability map for the WASM backend.
 *
 * The WASM build ships a focused subset: the dense, sparse, text,
 * hybrid and multi-query search paths. Everything that relies on
 * persistent on-disk structures (secondary indexes, graph, streaming,
 * PQ training, agent memory, introspection) is explicitly `false`;
 * `backends/wasm-stubs.ts` holds those throw sites.
 *
 * This table is the one place WASM support is stated. The WASM search
 * paths read it before they use an option
 * (`backends/wasm-capability-guards.ts`) and refuse with `NOT_SUPPORTED`
 * what it withholds: a `filter` on an operation `filteredSearch` does
 * not list, a `fusionParams` field outside `multiQueryFusionParams`,
 * `sparseIndexName`, `includeVectors: true`. None is dropped. The map
 * once said `sparseSearch: false` while sparse search ran (#2095);
 * `tests/wasm-capabilities-conformance.test.ts` now probes every key
 * against the backend, so the two cannot drift apart unnoticed.
 *
 * `velesqlQuery` is `false`: `query()` only executes pure top-k NEAR
 * statements (`SELECT * FROM <collection> WHERE vector NEAR $param
 * [LIMIT n]`) and throws `NOT_SUPPORTED` for any other VelesQL clause
 * (WHERE predicates, JOIN, GROUP BY, MATCH, set operations, FUSION),
 * so full VelesQL is not advertised.
 */
export const WASM_CAPABILITIES: Readonly<CapabilityMap> = Object.freeze({
  vectorSearch: true,
  textSearch: true,
  hybridSearch: true,
  multiQuerySearch: true,
  sparseSearch: true,
  // Dense search filters through the binding's `search_with_filter`; its
  // `sparse_search`, `text_search`, `hybrid_search` and
  // `multi_query_search` take no filter.
  filteredSearch: Object.freeze<FilteredSearchOperation[]>(['search']),
  // velesdb-wasm's `multi_query_search` takes `rrf_k` and the weighted
  // `[avg, max, hit]` triple. Its `relative_score` averages the query
  // branches with equal weight, so it has no use for dense/sparse weights.
  multiQueryFusionParams: Object.freeze<FusionParamName[]>([
    'k',
    'avgWeight',
    'maxWeight',
    'hitWeight',
  ]),
  namedSparseIndexes: false,
  includeVectors: false,
  idOnlySearch: false,
  scroll: false,
  graphTraversal: false,
  secondaryIndexes: false,
  agentMemory: false,
  enableStreaming: false,
  streamInsert: false,
  pqTraining: false,
  velesqlQuery: false,
  collectionIntrospection: false,
  // `velesqlQuery` is false on this backend, so the VelesQL sub-capabilities
  // are all unavailable.
  velesqlFusionStrategies: Object.freeze([]),
  velesqlMatchOrderBy: false,
  velesqlAlterCollection: false,
});
