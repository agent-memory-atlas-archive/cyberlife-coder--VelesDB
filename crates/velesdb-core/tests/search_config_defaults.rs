#![cfg(feature = "persistence")]
#![allow(clippy::cast_precision_loss)] // indices into f32 coordinates, deliberately

//! `[search]` reaches an unqualified `search()` — asserted on results, not on a
//! getter.
//!
//! The section was parsed, validated and ignored until #2087. Proving it is now
//! applied means observing the SEARCH, because a test reading back the resolved
//! quality would pass just as well if nothing consumed it.
//!
//! Every test here carries a control that makes its main assertion falsifiable:
//! comparing two searches only means something once the fixture is shown to be
//! hard enough that `ef_search` changes the answer at all.

use std::collections::HashMap;
use velesdb_core::{Database, DistanceMetric, Point, SearchMode, VelesConfig};

const DIM: usize = 32;
const POINTS: usize = 3_000;
// A larger corpus for the batch/multi-query and filtered-rerank tests below:
// at `POINTS`, their 4x-overfetched candidate pool already recovers the
// exact top-k at `LOW_EF` just as reliably as at `HIGH_EF` on this random
// fixture, so comparing the two configured defaults would prove nothing.
const HARD_POINTS: usize = 60_000;
const K: usize = 10;
// A larger k for `search_batch_with_filters` and `multi_query_search`: at
// `K`, each one's internal overfetch window agrees on the FINAL top-`K` for
// both `LOW_EF` and `HIGH_EF` (the obvious, easy matches), even while their
// wider internal candidate pools disagree further down; asking for more
// final results surfaces that disagreement. See the call sites.
const WIDE_K: usize = 50;
// `multi_query_search`'s own overfetch tiers (`Collection::overfetch_factor`)
// give a *smaller* candidate window to a *larger* top_k once top_k > 100
// (x2, vs x10 for the 11..=50 bucket `WIDE_K` falls in) -- 101 lands just
// past that boundary, keeping the window (202) close enough to the output
// size (101) to surface a LOW_EF vs HIGH_EF disagreement that a wider
// window (500 at `WIDE_K`) saturates away on this fixture.
const MULTI_K: usize = 101;
const LOW_EF: usize = 16; // the minimum `validate_search` accepts
const HIGH_EF: usize = 4_096; // the maximum

/// Deterministic pseudo-random coordinates: a fixture that changes between runs
/// would make a recall-sensitive assertion flap.
fn vector(seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..DIM)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 10_000) as f32 / 10_000.0
        })
        .collect()
}

fn seeded(dir: &tempfile::TempDir, config: VelesConfig) -> velesdb_core::VectorCollection {
    seeded_n(dir, config, POINTS)
}

fn seeded_n(
    dir: &tempfile::TempDir,
    config: VelesConfig,
    n: usize,
) -> velesdb_core::VectorCollection {
    let db = Database::open_with_config(dir.path(), config).expect("test: open");
    db.create_vector_collection("docs", DIM, DistanceMetric::Euclidean)
        .expect("test: create");
    let collection = db.get_vector_collection("docs").expect("test: collection");
    let points: Vec<Point> = (0..n as u64)
        .map(|id| Point::new(id, vector(id), None))
        .collect();
    collection.upsert(points).expect("test: upsert");
    collection
}

fn ids(results: &[velesdb_core::SearchResult]) -> Vec<u64> {
    results.iter().map(|r| r.point.id).collect()
}

fn config_with_ef(ef: usize) -> VelesConfig {
    let mut config = VelesConfig::default();
    config.search.ef_search = Some(ef);
    config
}

/// `search.ef_search` reaches `search()`.
///
/// The control comes first: unless `LOW_EF` and `HIGH_EF` disagree on this
/// fixture, comparing anything to the low-ef answer proves nothing. If that
/// assertion ever fails the fixture got easy, and the message says so rather
/// than leaving a green test that checks nothing.
#[test]
fn a_configured_ef_search_reaches_an_unqualified_search() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");
    let collection = seeded(&dir, config_with_ef(LOW_EF));
    let query = vector(POINTS as u64 + 1);

    let low = ids(&collection
        .search_with_ef(&query, K, LOW_EF)
        .expect("test: low ef"));
    let high = ids(&collection
        .search_with_ef(&query, K, HIGH_EF)
        .expect("test: high ef"));
    assert_ne!(
        low, high,
        "CONTROL: ef must change the answer on this fixture, or the assertion below is vacuous"
    );

    assert_eq!(
        ids(&collection.search(&query, K).expect("test: search")),
        low,
        "an unqualified search must use the configured ef, not the built-in Balanced"
    );
}

/// A default config leaves `search()` exactly where it was.
///
/// The regression that matters most: every existing caller must see the
/// built-in `Balanced` it saw before the wiring.
#[test]
fn a_default_config_leaves_search_on_the_built_in_quality() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");
    let collection = seeded(&dir, VelesConfig::default());
    let query = vector(POINTS as u64 + 1);
    assert_eq!(
        ids(&collection.search(&query, K).expect("test: search")),
        ids(&collection
            .search_with_quality(&query, K, velesdb_core::SearchQuality::Balanced)
            .expect("test: balanced")),
        "an unconfigured collection must answer exactly as Balanced does"
    );
}

fn reopened(
    dir: &tempfile::TempDir,
    config: VelesConfig,
) -> (Database, velesdb_core::VectorCollection) {
    let db = Database::open_with_config(dir.path(), config).expect("test: reopen");
    let collection = db.get_vector_collection("docs").expect("test: collection");
    (db, collection)
}

/// A per-query `ef` does not depend on the configured default.
///
/// One persisted index, reopened under two DIFFERENT configured defaults, must
/// give the same answer to the same explicit call -- if the default leaked into
/// explicit searches the two would disagree. Same index on purpose: two indexes
/// built separately can differ by construction and would make the equality
/// flap for reasons unrelated to configuration.
///
/// The first draft compared an explicit call to ITSELF and labelled it a
/// control, which could not fail, and asserted only that explicit and default
/// answers differ, which a clamped `ef` satisfies (#2246, P2-b and P2-e). That
/// an explicit `ef` is honoured at all is proven by the control of
/// `a_configured_ef_search_reaches_an_unqualified_search`, which fails if
/// `search_with_ef` ignores its argument.
#[test]
fn a_per_query_ef_is_independent_of_the_configured_default() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");
    seeded(&dir, VelesConfig::default())
        .flush_full()
        .expect("test: persist the index both opens will read");
    let query = vector(POINTS as u64 + 1);

    let (db_low, low) = reopened(&dir, config_with_ef(LOW_EF));
    let low_default = ids(&low.search(&query, K).expect("test: low default"));
    let low_explicit = ids(&low
        .search_with_ef(&query, K, HIGH_EF)
        .expect("test: explicit"));
    drop(low);
    drop(db_low);

    let (_db_high, high) = reopened(&dir, config_with_ef(HIGH_EF));
    let high_default = ids(&high.search(&query, K).expect("test: high default"));
    let high_explicit = ids(&high
        .search_with_ef(&query, K, HIGH_EF)
        .expect("test: explicit"));

    assert_ne!(
        low_default, high_default,
        "CONTROL: the two configured defaults must answer differently, or the \
         equality below could not tell a leak from a default with no effect"
    );
    assert_eq!(
        low_explicit, high_explicit,
        "an explicit ef must give the same answer whatever default the collection was opened with"
    );
}

/// `perfect` as a global default still loads, and is applied as `accurate`.
///
/// This refused at load until the seven-lens review (#2246): a TOML accepted by
/// v6.0.0 then failed `Database::open`, a breaking change shipped under
/// `### Added`. The concern behind the refusal is kept -- a filtered search's
/// bitmap pre-filter never reads the configured quality, so a global `Perfect`
/// would scan on some queries and traverse on others -- by never letting the
/// default resolve to it.
///
/// Asserted on the resolution because
/// `a_configured_ef_search_reaches_an_unqualified_search` already proves the
/// resolved quality is what `search()` runs; together they cover the path.
#[test]
fn perfect_as_a_global_default_still_opens_and_resolves_to_accurate() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");
    let mut opening = VelesConfig::default();
    opening.search.default_mode = SearchMode::Perfect;
    Database::open_with_config(dir.path(), opening)
        .expect("a config v6.0.0 accepted must keep opening a database");

    let mut config = VelesConfig::default();
    config.search.default_mode = SearchMode::Perfect;
    assert_eq!(
        config.search.resolved_quality(),
        velesdb_core::SearchQuality::Accurate,
        "a global `perfect` must resolve to `accurate`: the bitmap pre-filter never reads it"
    );

    config.search.default_mode = SearchMode::Balanced;
    assert_eq!(
        config.search.resolved_quality(),
        velesdb_core::SearchQuality::Balanced,
        "CONTROL: only `perfect` is downgraded; every other mode resolves to itself"
    );
}

/// A rerank-only `WITH` option -- naming no `mode`/`ef_search` of its own --
/// still reaches the configured `[search]` quality, on both the plain and
/// the metadata-filtered vector path (#2399).
///
/// Comparing `WITH (rerank=...)` alone against an explicit `WITH (ef_search=
/// LOW_EF, rerank=...)` is itself the control: if the rerank-only path fell
/// back to a hard-coded `Balanced` (the bug #2399 fixes), it would never
/// match the LOW_EF-explicit answer and would stay constant regardless of
/// LOW_EF vs HIGH_EF -- so this also proves the two explicit answers differ
/// on this fixture, without a separate control block.
#[test]
fn a_configured_ef_search_reaches_a_rerank_only_with_clause() {
    let dir = tempfile::TempDir::new().expect("test: tempdir");
    let collection = seeded(&dir, config_with_ef(LOW_EF));
    let query = vector(POINTS as u64 + 1);
    let mut params = HashMap::new();
    params.insert("v".to_string(), serde_json::json!(query));

    for rerank in [true, false] {
        let default_rerank = ids(&collection
            .execute_query_str(
                &format!(
                    "SELECT * FROM docs WHERE vector NEAR $v LIMIT {K} WITH (rerank={rerank})"
                ),
                &params,
            )
            .expect("test: rerank-only query"));
        let low_explicit = ids(&collection
            .execute_query_str(
                &format!(
                    "SELECT * FROM docs WHERE vector NEAR $v LIMIT {K} WITH (ef_search={LOW_EF}, rerank={rerank})"
                ),
                &params,
            )
            .expect("test: low-ef explicit query"));
        let high_explicit = ids(&collection
            .execute_query_str(
                &format!(
                    "SELECT * FROM docs WHERE vector NEAR $v LIMIT {K} WITH (ef_search={HIGH_EF}, rerank={rerank})"
                ),
                &params,
            )
            .expect("test: high-ef explicit query"));

        assert_ne!(
            low_explicit, high_explicit,
            "CONTROL: ef must change the rerank={rerank} answer on this fixture"
        );
        assert_eq!(
            default_rerank, low_explicit,
            "WITH (rerank={rerank}) alone must follow the configured ef_search, \
             not a hard-coded Balanced"
        );
    }
}

/// Same proof, through the metadata-filtered vector path
/// (`vector_filter.rs`'s `search_with_filter_and_opts`).
///
/// Uses `HARD_POINTS` and several query vectors, not one -- see the comment
/// on `a_configured_ef_search_reaches_the_batch_and_multi_query_entry_points`:
/// at `POINTS`, or for an unlucky single query, this filtered path's
/// oversampled candidate pool can recover the exact top-k at `LOW_EF` just
/// as reliably as at `HIGH_EF`, making the control pass vacuously. Checking
/// several queries and requiring the control to hold for at least one keeps
/// the equality assertion meaningful without depending on any single query
/// landing on the hard side of this fixture's recall boundary.
#[test]
fn a_configured_ef_search_reaches_a_rerank_only_with_clause_filtered() {
    const QUERIES: usize = 8;

    let dir = tempfile::TempDir::new().expect("test: tempdir");
    let db = Database::open_with_config(dir.path(), config_with_ef(LOW_EF)).expect("test: open");
    db.create_vector_collection("docs", DIM, DistanceMetric::Euclidean)
        .expect("test: create");
    let collection = db.get_vector_collection("docs").expect("test: collection");
    let points: Vec<Point> = (0..HARD_POINTS as u64)
        .map(|id| Point::new(id, vector(id), Some(serde_json::json!({ "cat": id % 2 }))))
        .collect();
    collection.upsert(points).expect("test: upsert");

    let mut control_held = false;
    for qi in 0..QUERIES as u64 {
        let query = vector(HARD_POINTS as u64 + 1 + qi);
        let mut params = HashMap::new();
        params.insert("v".to_string(), serde_json::json!(query));

        for rerank in [true, false] {
            let default_rerank = ids(&collection
                .execute_query_str(
                    &format!(
                        "SELECT * FROM docs WHERE vector NEAR $v AND cat = 0 LIMIT {K} \
                         WITH (rerank={rerank})"
                    ),
                    &params,
                )
                .expect("test: filtered rerank-only query"));
            let low_explicit = ids(&collection
                .execute_query_str(
                    &format!(
                        "SELECT * FROM docs WHERE vector NEAR $v AND cat = 0 LIMIT {K} \
                         WITH (ef_search={LOW_EF}, rerank={rerank})"
                    ),
                    &params,
                )
                .expect("test: filtered low-ef explicit query"));
            let high_explicit = ids(&collection
                .execute_query_str(
                    &format!(
                        "SELECT * FROM docs WHERE vector NEAR $v AND cat = 0 LIMIT {K} \
                         WITH (ef_search={HIGH_EF}, rerank={rerank})"
                    ),
                    &params,
                )
                .expect("test: filtered high-ef explicit query"));

            if low_explicit != high_explicit {
                control_held = true;
                assert_eq!(
                    default_rerank, low_explicit,
                    "a filtered WITH (rerank={rerank}) alone must follow the configured \
                     ef_search, not a hard-coded Balanced (query {qi})"
                );
            }
        }
    }
    assert!(
        control_held,
        "CONTROL: ef must change the filtered rerank-only answer for at least one of \
         {QUERIES} queries on this fixture"
    );
}

/// The configured `[search]` quality also reaches the batch/multi-query
/// entry points, which take no per-call quality of their own
/// (`search_batch_parallel`, `search_batch_with_filters`,
/// `multi_query_search`): each hard-coded `SearchQuality::Balanced` until
/// #2427's round-2 review found the same bug #2399 fixed for single-query
/// search, just unreached by #2399's own fix.
///
/// Reopening one persisted index under two different configured defaults and
/// finding the SAME answer every time would mean the config never reached
/// the call -- so "at least one of several queries disagrees" is both the
/// control and the proof, the same shape as
/// `a_configured_ef_search_reaches_an_unqualified_search`.
///
/// Checks several query vectors, not one: `HnswIndex` assigns random graph
/// levels per build (unseeded), so a single query can land in a lucky spot
/// where even `LOW_EF`'s narrow beam already recovers the exact answer
/// `HIGH_EF` would -- observed directly while writing this test, on the same
/// fixture, across different `cargo test` runs. If the fix were reverted
/// (both configs running the same hard-coded `Balanced`), EVERY query would
/// agree, every run, regardless of this per-build randomness.
///
/// Uses `HARD_POINTS`, not `POINTS` -- see the comment on
/// `a_configured_ef_search_reaches_a_rerank_only_with_clause_filtered`: the
/// 4x-overfetched candidate pool these entry points request saturates recall
/// at `POINTS` regardless of `ef`, making `LOW_EF` vs `HIGH_EF` answer
/// identically whether or not the configured quality reached the index.
#[test]
fn a_configured_ef_search_reaches_the_batch_and_multi_query_entry_points() {
    const QUERIES: usize = 8;

    let dir = tempfile::TempDir::new().expect("test: tempdir");
    seeded_n(&dir, VelesConfig::default(), HARD_POINTS)
        .flush_full()
        .expect("test: persist the index both opens will read");
    let queries: Vec<Vec<f32>> = (0..QUERIES as u64)
        .map(|i| vector(HARD_POINTS as u64 + 1 + i))
        .collect();

    // `search_batch_with_filters` overfetches `4 * k` candidates internally
    // before truncating back to `k`; at `K` that floor (40) is small enough
    // to still saturate recall on this fixture regardless of `ef`, so this
    // call alone asks for a bigger `k` to push its overfetch floor past the
    // point where `LOW_EF` and `HIGH_EF` agree.
    let (db_low, low) = reopened(&dir, config_with_ef(LOW_EF));
    let low_batch_parallel: Vec<_> = queries
        .iter()
        .map(|q| {
            ids(&low
                .search_batch_parallel(&[q], K)
                .expect("test: low batch_parallel")[0])
        })
        .collect();
    let low_batch_filtered: Vec<_> = queries
        .iter()
        .map(|q| {
            ids(&low
                .search_batch_with_filters(&[q], WIDE_K, &[None])
                .expect("test: low batch_with_filters")[0])
        })
        .collect();
    let low_multi: Vec<_> = queries
        .iter()
        .map(|q| {
            ids(&low
                .multi_query_search(
                    &[q],
                    MULTI_K,
                    velesdb_core::fusion::FusionStrategy::Maximum,
                    None,
                )
                .expect("test: low multi_query_search"))
        })
        .collect();
    drop(low);
    drop(db_low);

    let (_db_high, high) = reopened(&dir, config_with_ef(HIGH_EF));
    let high_batch_parallel: Vec<_> = queries
        .iter()
        .map(|q| {
            ids(&high
                .search_batch_parallel(&[q], K)
                .expect("test: high batch_parallel")[0])
        })
        .collect();
    let high_batch_filtered: Vec<_> = queries
        .iter()
        .map(|q| {
            ids(&high
                .search_batch_with_filters(&[q], WIDE_K, &[None])
                .expect("test: high batch_with_filters")[0])
        })
        .collect();
    let high_multi: Vec<_> = queries
        .iter()
        .map(|q| {
            ids(&high
                .multi_query_search(
                    &[q],
                    MULTI_K,
                    velesdb_core::fusion::FusionStrategy::Maximum,
                    None,
                )
                .expect("test: high multi_query_search"))
        })
        .collect();

    assert!(
        low_batch_parallel
            .iter()
            .zip(&high_batch_parallel)
            .any(|(l, h)| l != h),
        "search_batch_parallel must follow the configured ef_search, not a hard-coded Balanced \
         (all {QUERIES} queries agreed between LOW_EF and HIGH_EF)"
    );
    assert!(
        low_batch_filtered
            .iter()
            .zip(&high_batch_filtered)
            .any(|(l, h)| l != h),
        "search_batch_with_filters must follow the configured ef_search, not a hard-coded \
         Balanced (all {QUERIES} queries agreed between LOW_EF and HIGH_EF)"
    );
    assert!(
        low_multi.iter().zip(&high_multi).any(|(l, h)| l != h),
        "multi_query_search must follow the configured ef_search, not a hard-coded Balanced \
         (all {QUERIES} queries agreed between LOW_EF and HIGH_EF)"
    );
}
