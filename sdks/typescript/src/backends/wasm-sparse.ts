/**
 * WASM Backend — sparse vector bookkeeping
 *
 * velesdb-wasm's sparse index keys postings by a document id and has no way
 * to delete a document's postings, nor to drop the terms a re-insert leaves
 * out: `VectorStore.remove` leaves them in place, and an insert overwrites
 * only the terms it repeats. Indexed under the point's own id, a deleted
 * point, or a term a point no longer has, would keep matching sparse queries.
 *
 * Each sparse upsert is therefore indexed under a fresh sparse id. Replacing
 * or deleting the point retires that id: its postings stay in the binding
 * but map to no point, and searches skip them, over-fetching by the number
 * of retired ids so the dead cannot crowd live points out of the top `k`.
 * As in core, an upsert without a sparse vector keeps the point's current one.
 */

import type { SparseVector } from '../types';
import type { SparseIds, WasmSparseResult, WasmVectorStore } from './wasm-types';
import { sparseVectorToArrays } from './wasm-helpers';

/** Empty bookkeeping for a new collection. */
export function newSparseIds(): SparseIds {
  return { byPoint: new Map(), byId: new Map(), dead: 0, next: 1n };
}

/** Retire `pointId`'s sparse id, if it has one: its postings stop matching. */
export function retireSparse(ids: SparseIds, pointId: number): void {
  const current = ids.byPoint.get(pointId);
  if (current === undefined) {
    return;
  }
  ids.byPoint.delete(pointId);
  ids.byId.delete(current);
  ids.dead += 1;
}

/** Index `vector` as `pointId`'s sparse vector, retiring the one it had. */
export function indexSparse(
  store: WasmVectorStore,
  ids: SparseIds,
  pointId: number,
  vector: SparseVector
): void {
  const { indices, values } = sparseVectorToArrays(vector);
  const sparseId = ids.next;
  store.sparse_insert(sparseId, new Uint32Array(indices), new Float32Array(values));
  ids.next += 1n;
  retireSparse(ids, pointId);
  ids.byPoint.set(pointId, sparseId);
  ids.byId.set(sparseId, pointId);
}

/** The top `k` sparse hits as `[pointId, score]`, live points only. */
export function sparseHits(
  store: WasmVectorStore,
  ids: SparseIds,
  indices: number[],
  values: number[],
  k: number
): Array<[number, number]> {
  const raw: WasmSparseResult[] = store.sparse_search(
    new Uint32Array(indices),
    new Float32Array(values),
    k + ids.dead
  );
  const hits: Array<[number, number]> = [];
  for (const { doc_id, score } of raw) {
    const pointId = ids.byId.get(BigInt(doc_id));
    if (pointId !== undefined) {
      hits.push([pointId, score]);
      if (hits.length === k) {
        break;
      }
    }
  }
  return hits;
}
