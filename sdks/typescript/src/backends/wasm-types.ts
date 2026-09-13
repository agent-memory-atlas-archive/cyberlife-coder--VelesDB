/**
 * WASM Backend — Shared type definitions
 *
 * Internal context interface used by wasm-search.ts and wasm-stubs.ts
 * to access WasmBackend internals without circular dependencies.
 */

import type { CollectionConfig } from '../types';
import type { SparseVector } from '../types';
import type { VectorStore as BindingVectorStore } from '@wiscale/velesdb-wasm';

/**
 * Parameter list of a velesdb-wasm `VectorStore` method, read from the
 * binding's declaration file. A hand-copied list went stale: it omitted the
 * `weights` argument `multi_query_search` has taken since velesdb-wasm
 * 4.0.0, so the SDK never passed it (#2095).
 */
type BindingParams<M extends keyof BindingVectorStore> =
  BindingVectorStore[M] extends (...args: infer P) => unknown ? P : never;

// ---------------------------------------------------------------------------
// WASM result types — mirror the shapes returned by velesdb-wasm
// ---------------------------------------------------------------------------

/** Dense search result: [id, score] tuple returned by VectorStore.search(). */
export type WasmDenseResult = [bigint, number];

/** Sparse/hybrid search result returned by sparse_search / hybrid_search_fuse. */
export interface WasmSparseResult {
  doc_id: bigint | number;
  score: number;
}

/** Filtered search result returned by VectorStore.search_with_filter(). */
export interface WasmFilteredResult {
  id: bigint;
  score: number;
  payload?: Record<string, unknown> | null;
}

/** Hybrid search result returned by VectorStore.hybrid_search(). */
export interface WasmHybridResult {
  id: bigint | number;
  score: number;
  payload?: Record<string, unknown>;
}

/** Point returned by VectorStore.get(). */
export interface WasmPoint {
  id: bigint | number;
  vector: number[] | Float32Array;
  payload?: Record<string, unknown> | null;
}

/** Generic search result (tuple or object) returned by text_search / multi_query_search. */
export type WasmSearchResultItem =
  | WasmDenseResult
  | WasmHybridResult;

// ---------------------------------------------------------------------------
// VectorStore — typed interface for the WASM VectorStore class
// ---------------------------------------------------------------------------

/** Typed interface for the velesdb-wasm VectorStore class instance. */
export interface WasmVectorStore {
  /** Release WASM memory. */
  free(): void;

  /** Insert a vector by ID. */
  insert(id: bigint, vector: Float32Array): void;

  /** Insert a vector with JSON payload. */
  insert_with_payload(id: bigint, vector: Float32Array, payload: unknown): void;

  /** Batch insert: array of [id, vector] pairs. */
  insert_batch(batch: Array<[bigint, number[]]>): void;

  /** Pre-allocate memory for additional vectors. */
  reserve(additional: number): void;

  /** Remove a vector by ID. Returns true if found. */
  remove(id: bigint): boolean;

  /** Get a point by ID. Returns point object or null. */
  get(id: bigint): WasmPoint | null;

  /** Whether the store is empty (getter property). */
  readonly is_empty: boolean;

  /** Number of vectors in the store (getter property). */
  readonly len: number;

  // The search methods take their parameter lists from the binding's own
  // declaration file (`BindingParams`); only the result shapes, which the
  // binding types as `any`, are stated here.

  /** k-NN dense search. Returns array of [id, score] tuples. */
  search(...args: BindingParams<'search'>): WasmDenseResult[];

  /** k-NN search with metadata filter. Returns array of {id, score, payload}. */
  search_with_filter(...args: BindingParams<'search_with_filter'>): WasmFilteredResult[];

  /** Sparse index search. Returns array of {doc_id, score}. */
  sparse_search(...args: BindingParams<'sparse_search'>): WasmSparseResult[];

  /**
   * Text search on payload fields. The third argument names one payload
   * field to match; this method takes no filter.
   */
  text_search(...args: BindingParams<'text_search'>): WasmSearchResultItem[];

  /** Hybrid vector + text search. Returns array of {id, score, payload}. */
  hybrid_search(...args: BindingParams<'hybrid_search'>): WasmHybridResult[];

  /**
   * Multi-query search with fusion: `(vectors, num_vectors, k, strategy,
   * rrf_k?, weights?)`, `weights` being the weighted strategy's
   * `[avg, max, hit]`. Returns mixed result items.
   */
  multi_query_search(...args: BindingParams<'multi_query_search'>): WasmSearchResultItem[];

  /** VelesQL-style query returning multi-model results. */
  query(...args: BindingParams<'query'>): Record<string, unknown>[];
}

// ---------------------------------------------------------------------------
// WasmModule — typed interface for the imported WASM package
// ---------------------------------------------------------------------------

/** Constructor signature for the VectorStore class exported by velesdb-wasm. */
export interface WasmVectorStoreConstructor {
  new (dimension: number, metric: string): WasmVectorStore;
}

/** Typed interface for the @wiscale/velesdb-wasm module. */
export interface WasmModule {
  /** WASM initialization function (must be called once before use).
   *
   * Accepts an optional argument that wasm-bindgen forwards to its loader:
   *  - browser: omit, the loader will fetch the .wasm next to the JS module.
   *  - Node.js: pass a `BufferSource` (e.g. `await fs.readFile(...)`) because
   *    Node's stdlib fetch has no `file://` scheme handler. The WasmBackend
   *    helper does this transparently when running under Node.
   */
  default(moduleOrPath?: Uint8Array | URL | string): Promise<void>;

  /** VectorStore class constructor. */
  VectorStore: WasmVectorStoreConstructor;

  /** Fuse dense + sparse search results via Reciprocal Rank Fusion. */
  hybrid_search_fuse(
    denseResults: Array<[number, number]>,
    sparseResults: Array<[number, number]>,
    rrfK: number,
    k?: number
  ): WasmSparseResult[];
}

/** In-memory collection storage */
export interface CollectionData {
  config: CollectionConfig;
  store: WasmVectorStore;
  payloads: Map<string, Record<string, unknown>>;
  createdAt: Date;
}

/**
 * Internal context passed from WasmBackend to extracted search/stub modules.
 *
 * Exposes the minimum surface needed by helper functions without leaking the
 * full class. All methods mirror private WasmBackend helpers.
 */
export interface WasmContext {
  wasmModule: WasmModule;
  getCollection(name: string): CollectionData | undefined;
  canonicalPayloadKeyFromResultId(id: bigint | number | string): string;
  canonicalPayloadKey(id: string | number): string;
  sparseVectorToArrays(sv: SparseVector): { indices: number[]; values: number[] };
  toNumericId(id: string | number): number;
}
