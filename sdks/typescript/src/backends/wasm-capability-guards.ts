/**
 * WASM Backend — capability guards
 *
 * The WASM search paths call these before they use an option, so what
 * `WASM_CAPABILITIES` tells a caller and what the backend does are decided
 * in one place. A capability the map withholds is refused through
 * `wasmNotSupported`, the SDK's standard `NOT_SUPPORTED` error, and never
 * silently dropped (#2095).
 */

import { WASM_CAPABILITIES } from '../capabilities';
import type { CapabilityMap, FilteredSearchOperation } from '../capabilities';
import type { FusionParams, FusionParamName } from '../types';
import { wasmNotSupported } from './shared';

/** `CapabilityMap` keys whose value is a `boolean`. */
export type BooleanCapability = {
  [K in keyof CapabilityMap]: CapabilityMap[K] extends boolean ? K : never;
}[keyof CapabilityMap];

/** Whether an option was given: `undefined` and `null` ask for nothing. */
export function isSet<T>(value: T): value is NonNullable<T> {
  return value !== undefined && value !== null;
}

/** Refuse `feature` unless `WASM_CAPABILITIES` grants `capability`. */
export function requireWasmCapability(capability: BooleanCapability, feature: string): void {
  if (!WASM_CAPABILITIES[capability]) {
    wasmNotSupported(`${feature} (capability '${capability}' is false)`);
  }
}

/** Refuse a `filter` on `operation` unless `WASM_CAPABILITIES.filteredSearch` lists it. */
export function requireWasmFilterSupport(
  operation: FilteredSearchOperation,
  filter: unknown
): void {
  if (isSet(filter) && !WASM_CAPABILITIES.filteredSearch.includes(operation)) {
    wasmNotSupported(
      `${operation} with a filter (capability 'filteredSearch' does not list '${operation}')`
    );
  }
}

/** Refuse every `fusionParams` field `WASM_CAPABILITIES.multiQueryFusionParams` does not list. */
export function requireWasmFusionParams(params: FusionParams | undefined): void {
  for (const [name, value] of Object.entries(params ?? {})) {
    if (
      isSet(value) &&
      !WASM_CAPABILITIES.multiQueryFusionParams.includes(name as FusionParamName)
    ) {
      wasmNotSupported(
        `multiQuerySearch fusionParams.${name} (capability 'multiQueryFusionParams' does not list it)`
      );
    }
  }
}
