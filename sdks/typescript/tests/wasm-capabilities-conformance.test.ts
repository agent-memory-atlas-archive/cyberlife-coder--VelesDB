/**
 * WASM capability map ⇔ WASM backend conformance (#2095).
 *
 * `WASM_CAPABILITIES` is what a caller reads to decide whether a call will
 * work. It was written by hand beside the backend and drifted from it:
 * `sparseSearch` said `false` while `search({ sparseVector })` ran. This file
 * holds every key to the backend's behaviour rather than to a second list.
 *
 * For each key, probes call the real `WasmBackend` over a mocked binding.
 * Each comes out `honoured`, `refused` (NOT_SUPPORTED) or `dropped`: a probe
 * for an option that resolves while the option's value never reached the
 * binding, nor shows in the result, was dropped, the failure #2095 is about.
 * A probe must be `honoured` where the map grants the capability and
 * `refused` where it does not, so `dropped` fails either way. A key with no
 * probe fails the completeness test, so a capability cannot be added without
 * saying how it is observed.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { WasmBackend } from '../src/backends/wasm';
import { VelesDBError } from '../src/types';
import { REST_CAPABILITIES, WASM_CAPABILITIES } from '../src/capabilities';

class MockVectorStore {
  insert = vi.fn();
  insert_with_payload = vi.fn();
  insert_batch = vi.fn();
  reserve = vi.fn();
  remove = vi.fn(() => true);
  get = vi.fn(() => null);
  free = vi.fn();
  // One hit, so a probe can tell whether an option shaped the result.
  search = vi.fn(() => [[1n, 0.9]]);
  search_with_filter = vi.fn(() => []);
  sparse_search = vi.fn(() => []);
  text_search = vi.fn(() => []);
  hybrid_search = vi.fn(() => []);
  multi_query_search = vi.fn(() => []);
  query = vi.fn(() => []);
  len = 0;
  is_empty = true;
  constructor(public dimension: number, _metric: string) {}
}

const mockWasmModule = {
  default: vi.fn(() => Promise.resolve()),
  VectorStore: MockVectorStore,
  hybrid_search_fuse: vi.fn(() => []),
};

vi.mock('@wiscale/velesdb-wasm', () => mockWasmModule);

const C = 'c';
const V = [0.1, 0.2];
const FILTER = { condition: { type: 'eq', field: 'tenant', value: 'mine' } };

type Call = (backend: WasmBackend) => Promise<unknown>;
/** Whether the option under test took effect, judged from the binding's calls or the result. */
type Applied = (result: unknown, store: MockVectorStore) => boolean;
interface Probe {
  call: Call;
  applied?: Applied;
}

/** A probe for a whole operation: running it is honouring it. */
const operation = (call: Call): Probe => ({ call });
/** A probe for an option: running it is not enough, the option must take effect. */
const option = (call: Call, applied: Applied): Probe => ({ call, applied });

/** Every argument the mocked binding received, typed arrays flattened. */
function bindingArgs(store: MockVectorStore): unknown[] {
  const mocks = [
    store.search,
    store.search_with_filter,
    store.sparse_search,
    store.text_search,
    store.hybrid_search,
    store.multi_query_search,
    store.query,
    mockWasmModule.hybrid_search_fuse,
  ];
  return mocks
    .flatMap((mock) => mock.mock.calls.flat())
    .flatMap((arg) => (ArrayBuffer.isView(arg) ? Array.from(arg as Float32Array) : [arg]));
}

const reachesBinding =
  (value: unknown): Applied =>
  (_result, store) =>
    bindingArgs(store).includes(value);

const returnsVectors: Applied = (result) =>
  Array.isArray(result) &&
  result.length > 0 &&
  result.every((row) => (row as { vector?: unknown }).vector !== undefined);

/** A filtered call for each `filteredSearch` value. */
const FILTERED_CALLS: Record<string, Call> = {
  search: (b) => b.search(C, V, { filter: FILTER }),
  sparseSearch: (b) => b.search(C, V, { sparseVector: { 1: 0.5 }, filter: FILTER }),
  textSearch: (b) => b.textSearch(C, 'q', { filter: FILTER }),
  hybridSearch: (b) => b.hybridSearch(C, V, 'q', { filter: FILTER }),
  multiQuerySearch: (b) => b.multiQuerySearch(C, [V], { filter: FILTER }),
};

/**
 * `fusionParams` exercising each field; the weighted triple only travels
 * whole. The values are exact in f32 and differ from every other argument a
 * multi-query call passes, so `reachesBinding` can only find them.
 */
const WEIGHTED_VALUES = { avgWeight: 0.5, maxWeight: 0.375, hitWeight: 0.125 };
const FUSION_PARAMS: Record<string, Record<string, number>> = {
  k: { k: 37 },
  avgWeight: WEIGHTED_VALUES,
  maxWeight: WEIGHTED_VALUES,
  hitWeight: WEIGHTED_VALUES,
  denseWeight: { denseWeight: 0.75 },
  sparseWeight: { sparseWeight: 0.625 },
};

/** `table[value]`, or an error naming the list value that has no probe. */
function entryFor<T>(table: Record<string, T>, key: string, value: string): T {
  const entry = table[value];
  if (entry === undefined) {
    throw new Error(`no probe for ${key} value '${value}'`);
  }
  return entry;
}

/** Probes for each boolean capability. */
const BOOLEAN_PROBES: Record<string, readonly Probe[]> = {
  vectorSearch: [
    operation((b) => b.search(C, V)),
    operation((b) => b.searchBatch(C, [{ vector: V }])),
  ],
  textSearch: [operation((b) => b.textSearch(C, 'q'))],
  hybridSearch: [operation((b) => b.hybridSearch(C, V, 'q'))],
  multiQuerySearch: [operation((b) => b.multiQuerySearch(C, [V]))],
  sparseSearch: [operation((b) => b.search(C, V, { sparseVector: { 1: 0.5 } }))],
  namedSparseIndexes: [
    option(
      (b) => b.search(C, V, { sparseVector: { 1: 0.5 }, sparseIndexName: 'splade_v2' }),
      reachesBinding('splade_v2')
    ),
    operation((b) => b.sparseSearchNamed(C, { 1: 0.5 }, 'splade_v2')),
  ],
  includeVectors: [option((b) => b.search(C, V, { includeVectors: true }), returnsVectors)],
  idOnlySearch: [
    operation((b) => b.searchIds(C, V)),
    operation((b) => b.multiQuerySearchIds(C, [V])),
  ],
  scroll: [operation((b) => b.scroll(C))],
  graphTraversal: [
    operation((b) => b.addEdge(C, { id: 1, source: 1, target: 2, label: 'R' })),
    operation((b) => b.getEdges(C)),
    operation((b) => b.traverseGraph(C, { source: 1 })),
    operation((b) => b.traverseParallel(C, { sources: [1] })),
    operation((b) => b.getNodeDegree(C, 1)),
  ],
  secondaryIndexes: [
    operation((b) => b.createIndex(C, { label: 'Doc', property: 'x' })),
    operation((b) => b.listIndexes(C)),
    operation((b) => b.hasIndex(C, 'Doc', 'x')),
    operation((b) => b.dropIndex(C, 'Doc', 'x')),
  ],
  agentMemory: [
    operation((b) => b.storeSemanticFact(C, { id: 1, text: 't', embedding: V })),
    operation((b) => b.searchSemanticMemory(C, V)),
    operation((b) => b.recordEpisodicEvent(C, { eventType: 'e', data: {}, embedding: V })),
    operation((b) => b.recallEpisodicEvents(C, V)),
    operation((b) => b.storeProceduralPattern(C, { name: 'p', steps: [] })),
    operation((b) => b.matchProceduralPatterns(C, V)),
  ],
  enableStreaming: [operation((b) => b.enableStreaming(C))],
  streamInsert: [operation((b) => b.streamInsert(C, [{ id: 1, vector: V }]))],
  pqTraining: [operation((b) => b.trainPq(C))],
  velesqlQuery: [operation((b) => b.query(C, "SELECT * FROM c WHERE tenant = 'mine' LIMIT 5"))],
  collectionIntrospection: [
    operation((b) => b.collectionSanity(C)),
    operation((b) => b.getCollectionStats(C)),
    operation((b) => b.analyzeCollection(C)),
    operation((b) => b.getCollectionConfig(C)),
  ],
  velesqlMatchOrderBy: [
    operation((b) => b.query(C, 'MATCH (d:Doc) RETURN d.id ORDER BY d.id LIMIT 1')),
  ],
  velesqlAlterCollection: [
    operation((b) => b.query(C, 'ALTER COLLECTION c SET(auto_reindex=true)')),
  ],
};

/**
 * For each list-valued capability, the probe for one value. The candidate
 * values are REST's list, since REST honours every one of them.
 */
const LIST_PROBES: Record<string, (value: string) => Probe> = {
  velesqlFusionStrategies: (strategy) =>
    operation((b) =>
      b.query(
        C,
        `SELECT * FROM c WHERE vector NEAR $v USING FUSION(strategy='${strategy}') LIMIT 5`,
        { v: V }
      )
    ),
  filteredSearch: (op) =>
    option(entryFor(FILTERED_CALLS, 'filteredSearch', op), reachesBinding(FILTER)),
  multiQueryFusionParams: (name) => {
    const fusionParams = entryFor(FUSION_PARAMS, 'multiQueryFusionParams', name);
    return option(
      (b) => b.multiQuerySearch(C, [V], { fusionParams }),
      reachesBinding(fusionParams[name])
    );
  },
};

type Outcome = 'honoured' | 'refused' | 'dropped';

function storeOf(backend: WasmBackend): MockVectorStore {
  const internals = backend as unknown as { collections: Map<string, { store: MockVectorStore }> };
  return internals.collections.get(C)!.store;
}

async function outcomeOf(probe: Probe, backend: WasmBackend): Promise<Outcome> {
  let result: unknown;
  try {
    result = await probe.call(backend);
  } catch (error) {
    if (error instanceof VelesDBError && error.code === 'NOT_SUPPORTED') {
      return 'refused';
    }
    throw error;
  }
  if (probe.applied && !probe.applied(result, storeOf(backend))) {
    return 'dropped';
  }
  return 'honoured';
}

const expectedOutcome = (granted: boolean): Outcome => (granted ? 'honoured' : 'refused');

const booleanCases = Object.entries(BOOLEAN_PROBES).flatMap(([key, probes]) =>
  probes.map((probe, index) => [key, index, probe] as const)
);

const listCases = Object.entries(LIST_PROBES).flatMap(([key, probeFor]) =>
  (REST_CAPABILITIES[key as keyof typeof REST_CAPABILITIES] as readonly string[]).map(
    (value) => [key, value, probeFor(value)] as const
  )
);

describe('WASM_CAPABILITIES matches what WasmBackend does (#2095)', () => {
  let backend: WasmBackend;

  beforeEach(async () => {
    vi.clearAllMocks();
    backend = new WasmBackend();
    await backend.init();
    await backend.createCollection(C, { dimension: V.length, metric: 'cosine' });
  });

  it('has a probe for every capability key', () => {
    const probed = [...Object.keys(BOOLEAN_PROBES), ...Object.keys(LIST_PROBES)].sort();
    expect(probed).toEqual(Object.keys(WASM_CAPABILITIES).sort());
  });

  it.each(Object.keys(LIST_PROBES))(
    '%s: every value WASM grants is one REST lists, so each is probed',
    (key) => {
      const rest = REST_CAPABILITIES[key as keyof typeof REST_CAPABILITIES] as readonly string[];
      const wasm = WASM_CAPABILITIES[key as keyof typeof WASM_CAPABILITIES] as readonly string[];
      expect(rest).toEqual(expect.arrayContaining([...wasm]));
    }
  );

  it.each(booleanCases)(
    '%s (probe %i): the backend does what the map says',
    async (key, _index, probe) => {
      const granted = WASM_CAPABILITIES[key as keyof typeof WASM_CAPABILITIES];
      expect(typeof granted).toBe('boolean');
      expect(await outcomeOf(probe, backend)).toBe(expectedOutcome(granted as boolean));
    }
  );

  it.each(listCases)(
    '%s lists %s exactly when the backend honours it',
    async (key, value, probe) => {
      const granted = WASM_CAPABILITIES[key as keyof typeof WASM_CAPABILITIES] as readonly string[];
      expect(await outcomeOf(probe, backend)).toBe(expectedOutcome(granted.includes(value)));
    }
  );
});
