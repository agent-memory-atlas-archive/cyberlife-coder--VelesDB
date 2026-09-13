"""Tests for scripts/bench-memory-extraction.py — above all, its refusals.

A bench reports a verdict on models. If its scorer cannot fail, every model
passes and the campaign publishes a number that means nothing. So each severity
has a vector here that MUST produce it, next to a positive control that must
produce nothing.

Three of these tests exist because the code they cover was already wrong, and
only running it showed that:

  * **The prompt parse stopped two thirds of the way in.** A lazy
    `format!\\(\\s*"(.*?)"\\s*\\)` ends on the `")` inside the prompt's own
    example, `(e.g. \\"bruno durand\\")`. It produced 904 characters of a
    2,696-character prompt, and the sanity guard missed it because every marker
    it checked — the passage, `STEP 0`, `"relations"` — appears in the surviving
    head. Guards now check the TAIL, and `test_prompt_is_whole` pins the closing
    JSON contract.
  * **The SSE decoder read the priming frame.** The daemon opens its stream with
    `data: ` carrying an empty payload before the reply, so taking the first
    `data:` line dies on empty input. Every frame is tried now.
  * **`memory_status` is not everywhere.** The daemon installed on this machine
    on 2026-08-15 exposes 20 tools and not that one — its binary predates it.
    Phase B calls it, so `probe` checks the tool list against the server instead
    of assuming, and reports "tool not found" as a missing capability rather
    than as a measurement.

The fixtures below are written by hand, which proves the scorer REACTS but not
that it reacts to what models produce. The campaign's raw answers are kept for
exactly that reason: once one has run, these fixtures get replaced by captured
ones.
"""

from __future__ import annotations

import contextlib
import copy
import datetime
import hashlib
import http.server
import importlib.util
import io
import json
import math
import platform
import re
import socket
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "bench-memory-extraction.py"
CASES_PATH = Path(__file__).resolve().parent.parent / "memory-extraction-cases.json"


def load_module():
    spec = importlib.util.spec_from_file_location("bench_memory_extraction", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load_module()


def case_by_id(case_id: str) -> dict:
    for case in bench.load_cases(CASES_PATH):
        if case["id"] == case_id:
            return case
    raise AssertionError(f"no case {case_id!r}")


def severities(failures: "list[dict]") -> "list[str]":
    return [failure["severity"] for failure in failures]


# ------------------------------------------------------------------- folding --


class FoldTest(unittest.TestCase):
    def test_ligature_expands_instead_of_vanishing(self):
        """NFKD does not decompose U+0153: the ASCII pass would drop it entirely.

        Without the explicit mapping `soeur` folds to `sur`, and every kinship
        check stops matching while still reporting success.
        """
        self.assertEqual(bench.fold("sœur"), "soeur")
        self.assertEqual(bench.fold("belle-sœur"), "belle-soeur")
        self.assertEqual(bench.fold("œuvre"), "oeuvre")
        self.assertEqual(bench.fold("Ex æquo"), "ex aequo")

    def test_folds_case_and_accents(self):
        self.assertEqual(bench.fold("Éco-Kérité"), "eco-kerite")
        self.assertEqual(bench.fold("Marie DUPONT"), "marie dupont")

    def test_folds_non_strings(self):
        self.assertEqual(bench.fold(15), "15")
        self.assertEqual(bench.fold(None), "none")


# ------------------------------------------------- the prompt, read from Rust --


class PromptSourcingTest(unittest.TestCase):
    def test_prompt_is_whole(self):
        """The parse must reach the closing JSON contract, not stop at an escaped quote."""
        template = bench.read_graph_prompt_template()
        prompt = bench.build_graph_prompt("Marie Dupont a une soeur, Camille Dupont.", template)
        self.assertIn("Marie Dupont a une soeur", prompt)
        self.assertIn('"attributes"', prompt)
        self.assertIn('"entity": string', prompt)
        self.assertTrue(prompt.rstrip().endswith("}"), prompt[-80:])
        # The truncated parse produced 904 characters; the whole prompt is ~2.7k.
        self.assertGreater(len(prompt), 2000)

    def test_braces_collapse_but_passage_braces_survive(self):
        template = bench.read_graph_prompt_template()
        prompt = bench.build_graph_prompt('Projet "Ardoise {beta}" chez Wiscale.', template)
        self.assertIn("Ardoise {beta}", prompt)
        self.assertIn('{"facts"', prompt)

    def test_refuses_a_truncated_template(self):
        """A template cut before the contract must raise, never be sent."""
        with self.assertRaises(RuntimeError):
            bench.build_graph_prompt("passage", "STEP 0 and \"relations\" but cut here")

    def test_refuses_a_source_without_the_function(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "extract.rs"
            empty.write_text("fn other() {}\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                bench.read_graph_prompt_template(empty)
            with self.assertRaises(RuntimeError):
                bench.read_generation_cap(empty)

    def test_generation_cap_matches_the_crate(self):
        self.assertEqual(bench.read_generation_cap(), 512)


# ------------------------------------------------------- scorer: the control --


GOOD_POSSESSIVE = {
    "facts": [{"fact": "Camille Dupont a 15 ans.", "entities": ["camille dupont"]}],
    "relations": [{"subject": "camille dupont", "predicate": "soeur de", "object": "marie dupont"}],
    "attributes": [
        {"entity": "camille dupont", "key": "age", "value": 15},
        {"entity": "camille dupont", "key": "employeur", "value": "Wiscale"},
    ],
}


class PositiveControlTest(unittest.TestCase):
    def test_a_correct_answer_scores_nothing(self):
        """Without this, a scorer that never fires would look like a strict one."""
        case = case_by_id("fr-possessive")
        failures = bench.score_passage(GOOD_POSSESSIVE, case["passages"][0]["checks"])
        self.assertEqual(failures, [], failures)

    def test_a_correct_empty_answer_scores_nothing(self):
        case = case_by_id("edge-no-relation")
        payload = {"facts": [{"fact": "Il pleuvait hier soir.", "entities": []}],
                   "relations": [], "attributes": []}
        self.assertEqual(bench.score_passage(payload, case["passages"][0]["checks"]), [])


# --------------------------------------------------- scorer: refusal vectors --


class RefusalVectorTest(unittest.TestCase):
    def test_relations_as_arrays_are_fatal(self):
        """The silent one: this JSON parses, RawRelation refuses it, enrichment vanishes."""
        payload = {"relations": [["camille dupont", "soeur de", "marie dupont"]]}
        failures = bench.score_passage(payload, case_by_id("fr-possessive")["passages"][0]["checks"])
        self.assertEqual(severities(failures), ["fatal"])
        self.assertEqual(failures[0]["type"], "schema")

    def test_unparsable_response_is_one_fatal_not_a_cascade(self):
        failures = bench.score_passage(None, case_by_id("fr-possessive")["passages"][0]["checks"])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["type"], "parse")

    def test_reversed_orientation_is_major(self):
        payload = {**GOOD_POSSESSIVE, "relations": [
            {"subject": "marie dupont", "predicate": "soeur de", "object": "camille dupont"},
        ]}
        failures = bench.score_passage(payload, case_by_id("fr-possessive")["passages"][0]["checks"])
        labels = [failure["label"] for failure in failures]
        self.assertIn("sibling edge missing or reversed", labels)
        self.assertIn("parasitic converse: both directions of one predicate", labels)

    def test_wrong_language_predicate_is_major(self):
        payload = {**GOOD_POSSESSIVE, "relations": [
            {"subject": "camille dupont", "predicate": "sister of", "object": "marie dupont"},
        ]}
        failures = bench.score_passage(payload, case_by_id("fr-possessive")["passages"][0]["checks"])
        self.assertIn("English predicate on a French passage",
                      [failure["label"] for failure in failures])

    def test_number_as_string_is_major(self):
        payload = {**GOOD_POSSESSIVE,
                   "attributes": [{"entity": "camille dupont", "key": "age", "value": "15"}]}
        failures = bench.score_passage(payload, case_by_id("fr-possessive")["passages"][0]["checks"])
        self.assertIn("age missing, misattributed, or emitted as a string",
                      [failure["label"] for failure in failures])

    def test_boolean_is_not_a_number(self):
        """`True` is an int in Python; an age of `true` must not pass as 15."""
        payload = {**GOOD_POSSESSIVE,
                   "attributes": [{"entity": "camille dupont", "key": "age", "value": True}]}
        failures = bench.score_passage(payload, case_by_id("fr-possessive")["passages"][0]["checks"])
        self.assertTrue(failures)

    def test_invented_relation_is_fatal(self):
        payload = {"facts": [], "relations": [
            {"subject": "la ville", "predicate": "a eu", "object": "pluie"},
        ], "attributes": []}
        failures = bench.score_passage(payload, case_by_id("edge-no-relation")["passages"][0]["checks"])
        self.assertEqual(severities(failures), ["fatal"])

    def test_truncation_is_fatal_and_comes_from_the_server(self):
        case = case_by_id("edge-verbose-truncation")
        payload = {"facts": [], "relations": [
            {"subject": "a", "predicate": "p", "object": "b"},
            {"subject": "c", "predicate": "p", "object": "d"},
            {"subject": "e", "predicate": "p", "object": "f"},
        ], "attributes": []}
        clean = bench.score_passage(payload, case["passages"][0]["checks"], truncated=False)
        cut = bench.score_passage(payload, case["passages"][0]["checks"], truncated=True)
        self.assertEqual(clean, [])
        self.assertEqual(severities(cut), ["fatal"])

    def test_cross_contamination_is_fatal(self):
        case = case_by_id("close-homonyms")
        payload = {"facts": [], "relations": [], "attributes": [
            {"entity": "marie dupont", "key": "ville", "value": "Lyon"},
            {"entity": "marie dupont", "key": "ville", "value": "Nantes"},
            {"entity": "marie dupond", "key": "ville", "value": "Nantes"},
        ]}
        failures = bench.score_passage(payload, case["passages"][0]["checks"])
        self.assertIn("fatal", severities(failures))

    def test_pronoun_left_unresolved_is_major(self):
        case = case_by_id("fr-pronoun")
        payload = {"facts": [{"fact": "Il habite a Lyon.", "entities": []}],
                   "relations": [], "attributes": [
                       {"entity": "bruno durand", "key": "ville", "value": "Lyon"}]}
        failures = bench.score_passage(payload, case["passages"][0]["checks"])
        self.assertIn("pronoun left unresolved in a fact meant to stand alone",
                      [failure["label"] for failure in failures])

    def test_both_directions_of_one_predicate_is_major(self):
        case = case_by_id("en-possessive")
        payload = {"facts": [], "relations": [
            {"subject": "tom miller", "predicate": "brother of", "object": "sarah miller"},
            {"subject": "sarah miller", "predicate": "brother of", "object": "tom miller"},
        ], "attributes": [{"entity": "tom miller", "key": "age", "value": 22}]}
        failures = bench.score_passage(payload, case["passages"][0]["checks"])
        self.assertIn("both directions over the same pair",
                      [failure["label"] for failure in failures])

    def test_unknown_check_type_raises(self):
        """A typo in the cases file must stop the campaign, not skip a check."""
        with self.assertRaises(RuntimeError):
            bench.score_passage({"relations": []}, [{"type": "nope", "severity": "major", "label": "x"}])


class CrossCheckTest(unittest.TestCase):
    def test_identical_predicates_across_a_close_pair_is_major(self):
        case = case_by_id("close-role")
        collapsed = {"relations": [
            {"subject": "alice martin", "predicate": "travaille chez", "object": "wiscale"}]}
        failures = bench.score_cross_checks([collapsed, collapsed], case["cross_checks"])
        self.assertEqual(severities(failures), ["major"])

    def test_distinct_predicates_pass(self):
        case = case_by_id("close-role")
        works = {"relations": [
            {"subject": "alice martin", "predicate": "travaille chez", "object": "wiscale"}]}
        leads = {"relations": [
            {"subject": "alice martin", "predicate": "dirige", "object": "wiscale"}]}
        self.assertEqual(bench.score_cross_checks([works, leads], case["cross_checks"]), [])

    def test_two_empty_answers_do_not_count_as_a_collapse(self):
        """Both empty is a different defect, already scored by the per-passage checks."""
        case = case_by_id("close-role")
        empty = {"relations": []}
        self.assertEqual(bench.score_cross_checks([empty, empty], case["cross_checks"]), [])


# ------------------------------------------------------ the same checks, graph --


class GraphViewTest(unittest.TestCase):
    def test_entity_response_becomes_scorable(self):
        profile = {
            "found": True, "name": "camille dupont",
            "attributes": {"age": 15},
            "relations": [{"predicate": "soeur de", "target": "Entity: marie dupont"}],
            "relations_in": [{"predicate": "employe", "target": "Entity: wiscale"}],
        }
        payload = bench.entity_as_payload("camille dupont", profile)
        triples = bench.relation_triples(payload)
        self.assertEqual(triples, [("camille dupont", "soeur de", "marie dupont")])
        self.assertEqual(payload["attributes"],
                         [{"entity": "camille dupont", "key": "age", "value": 15}])

    def test_incoming_edges_are_not_credited_to_this_entity(self):
        """An incoming edge belongs to its SOURCE; folding it in invents an edge."""
        profile = {"name": "wiscale", "attributes": {}, "relations": [],
                   "relations_in": [{"predicate": "travaille chez", "target": "Entity: alice martin"}]}
        payload = bench.entity_as_payload("wiscale", profile)
        self.assertEqual(payload["relations"], [])

    def test_graph_checks_keep_only_what_a_stored_entity_can_answer(self):
        case = case_by_id("fr-possessive")
        kept = {spec["type"] for spec in bench.graph_checks_for(case["passages"][0])}
        self.assertIn("relation_present", kept)
        self.assertIn("attribute_number", kept)
        # About the ANSWER, not the graph: scoring it here would double-count.
        self.assertNotIn("predicate_forbids", kept)

    def test_a_missing_entity_fails_its_graph_check(self):
        case = case_by_id("fr-possessive")
        empty = bench.entity_as_payload("camille dupont", {"found": False, "name": "camille dupont"})
        specs = bench.graph_checks_for(case["passages"][0])
        self.assertTrue(bench.score_passage(empty, specs))


# ---------------------------------------------------------------- transport --


class McpBodyTest(unittest.TestCase):
    def test_sse_priming_frame_is_skipped(self):
        """The exact body the daemon returned on 2026-08-15."""
        body = ('data: \nid: 0\nretry: 3000\n\n'
                'data: {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18"}}\n\n')
        decoded = bench.decode_mcp_body(body)
        self.assertEqual(decoded["result"]["protocolVersion"], "2025-06-18")

    def test_plain_json_body(self):
        self.assertEqual(bench.decode_mcp_body('{"jsonrpc":"2.0","id":1}')["id"], 1)

    def test_empty_body_is_none(self):
        self.assertIsNone(bench.decode_mcp_body("   "))

    def test_sse_without_any_payload_is_none(self):
        self.assertIsNone(bench.decode_mcp_body("data: \nid: 0\nretry: 3000\n\n"))


class ToolPayloadTest(unittest.TestCase):
    def test_refusal_inside_a_valid_result_is_not_a_success(self):
        """`isError` rides INSIDE a well-formed result; reading the outer reply lies."""
        refused = {"response": {"result": {"isError": True, "content": [
            {"type": "text", "text": "refused"}]}}}
        self.assertIsNone(bench.tool_payload(refused))

    def test_text_content_is_parsed_as_json(self):
        ok = {"response": {"result": {"content": [{"type": "text", "text": '{"found": true}'}]}}}
        self.assertEqual(bench.tool_payload(ok), {"found": True})

    def test_non_json_text_is_kept_verbatim(self):
        ok = {"response": {"result": {"content": [{"type": "text", "text": "plain"}]}}}
        self.assertEqual(bench.tool_payload(ok), {"text": "plain"})


# ------------------------------------------------------------ log digestion --


SAMPLE_LOG = """\
2026-08-15T09:23:21.000000Z  INFO velesdb_memory::mcp: tool=remember session=abc verdict=ok elapsed_ms=132033 "mcp tool call"
2026-08-15T09:24:00.000000Z  INFO velesdb_memory::mcp: tool=recall session=abc verdict=ok elapsed_ms=128 "mcp tool call"
2026-08-15T09:24:01.000000Z  INFO velesdb_memory::mcp: tool=recall session=abc verdict=tool_error elapsed_ms=5 "mcp tool call"
2026-08-15T09:24:02.000000Z  INFO velesdb_memory::http: mcp http request method=POST session=abc status=200 elapsed_ms=2
2026-08-14T23:40:41.000000Z  INFO velesdb_memory::mcp: tool=remember session=abc verdict=ok elapsed_ms=300403 "mcp tool call"
"""


class DigestTest(unittest.TestCase):
    def digest(self, since=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "daemon.err.log"
            path.write_text(SAMPLE_LOG, encoding="utf-8")
            return bench.digest_log(path, since)

    def test_tool_and_http_events_are_not_mixed(self):
        """Both carry `elapsed_ms`; averaging them buries a 132-second write."""
        digest = self.digest()
        self.assertEqual(digest["tools"]["remember"]["n"], 2)
        self.assertEqual(digest["http_post"]["n"], 1)
        self.assertNotIn("mcp", digest["tools"])

    def test_verdicts_separate_refusals_from_successes(self):
        digest = self.digest()
        self.assertEqual(digest["verdicts"]["recall:tool_error"], 1)
        self.assertEqual(digest["verdicts"]["recall:ok"], 1)

    def test_since_filters_by_timestamp(self):
        digest = self.digest(since="2026-08-15T00")
        self.assertEqual(digest["tools"]["remember"]["n"], 1)
        self.assertEqual(digest["tools"]["remember"]["max_ms"], 132033)


class PercentileTest(unittest.TestCase):
    def test_nearest_rank(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(bench.percentile(values, 0.50), 5)
        self.assertEqual(bench.percentile(values, 0.95), 10)
        self.assertEqual(bench.percentile([7], 0.95), 7)

    def test_empty_is_nan_not_a_crash(self):
        self.assertTrue(math.isnan(bench.percentile([], 0.5)))


# ------------------------------------------------------- the cases file itself --


class CasesFileTest(unittest.TestCase):
    """The cases are data, so their integrity is a test, not a runtime surprise.

    A mistyped check type raises mid-campaign otherwise — after the model has
    been loaded, warmed and half-scored.
    """

    def setUp(self):
        self.cases = bench.load_cases(CASES_PATH)

    def test_every_check_type_is_implemented(self):
        for case in self.cases:
            for passage in case["passages"]:
                for spec in passage["checks"]:
                    self.assertIn(spec["type"], bench.CHECKS, f"{case['id']}: {spec['type']}")
            for spec in case.get("cross_checks") or []:
                self.assertEqual(spec["type"], "predicates_differ", case["id"])

    def test_every_check_declares_a_known_severity_and_a_label(self):
        for case in self.cases:
            specs = [s for p in case["passages"] for s in p["checks"]]
            specs += case.get("cross_checks") or []
            for spec in specs:
                self.assertIn(spec["severity"], bench.SEVERITIES, case["id"])
                self.assertTrue(spec["label"].strip(), case["id"])

    def test_the_suite_covers_every_family_the_plan_names(self):
        families = {case["family"] for case in self.cases}
        self.assertEqual(families, {"nominal-fr", "nominal-en", "edge", "close-pair"})

    def test_french_and_english_nominals_mirror_each_other(self):
        """A model passing one language and failing the other must be visible."""
        fr = sum(1 for case in self.cases if case["family"] == "nominal-fr")
        en = sum(1 for case in self.cases if case["family"] == "nominal-en")
        self.assertEqual(fr, en)

    def test_close_pairs_carry_two_passages_and_a_cross_check(self):
        pairs = [case for case in self.cases
                 if case["family"] == "close-pair" and len(case["passages"]) == 2]
        self.assertTrue(pairs)
        for case in pairs:
            self.assertTrue(case.get("cross_checks"), case["id"])

    def test_case_ids_are_unique(self):
        ids = [case["id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_passage_builds_a_valid_prompt(self):
        """Cheap, and it proves no passage breaks the format! substitution."""
        template = bench.read_graph_prompt_template()
        for case in self.cases:
            for passage in case["passages"]:
                prompt = bench.build_graph_prompt(passage["text"], template)
                self.assertIn(passage["text"], prompt)


# ---------------------------------------------------------------- language ----


def scored_case(case_id, family, lang, fatal=0, major=0, seconds=1.0):
    """A minimal scored case, shaped as `screen_case` returns one."""
    failures = ([{"severity": "fatal", "label": "f", "type": "t"}] * fatal
                + [{"severity": "major", "label": "m", "type": "t"}] * major)
    return {
        "id": case_id, "family": family, "lang": lang,
        "passages": [{"seconds": seconds, "parse_ok": True, "truncated": False,
                      "failures": failures}],
        "cross_failures": [], "counts": bench.tally(failures),
    }


class LanguageVerdictTest(unittest.TestCase):
    """The verdict a global score cannot give.

    velesdb-memory is used in whatever language its user writes in, and the
    extractor model is theirs to choose. A model strong in English and weak in
    French does not merely score slightly lower — `works at` and `travaille
    chez` become two graph predicates for one relation, and the graph fragments.
    """

    def test_an_english_only_failure_names_english_as_weaker(self):
        results = [
            scored_case("fr1", "nominal-fr", "fr"),
            scored_case("fr2", "nominal-fr", "fr"),
            scored_case("en1", "nominal-en", "en", major=3),
            scored_case("en2", "nominal-en", "en"),
        ]
        gap = bench.mirror_gap(results)
        self.assertEqual(gap["weaker"], "en")
        self.assertEqual(gap["gap"], 3)

    def test_a_fatal_outweighs_majors_in_the_gap(self):
        """One fatal is not three majors: it means the graph is wrong, not poorer."""
        results = [
            scored_case("fr1", "nominal-fr", "fr", fatal=1),
            scored_case("en1", "nominal-en", "en", major=5),
        ]
        gap = bench.mirror_gap(results)
        self.assertEqual(gap["weaker"], "fr")

    def test_a_balanced_model_names_no_weaker_side(self):
        results = [
            scored_case("fr1", "nominal-fr", "fr", major=1),
            scored_case("en1", "nominal-en", "en", major=1),
        ]
        self.assertIsNone(bench.mirror_gap(results)["weaker"])

    def test_the_gap_ignores_the_french_only_families(self):
        """The suite is unbalanced on purpose; only the mirrors are comparable.

        Edge and close-pair cases are French, so counting them would report
        every model as 'weaker in French' regardless of what it did.
        """
        results = [
            scored_case("fr1", "nominal-fr", "fr"),
            scored_case("en1", "nominal-en", "en"),
            scored_case("edge1", "edge", "fr", fatal=2),
            scored_case("close1", "close-pair", "fr", major=4),
        ]
        gap = bench.mirror_gap(results)
        self.assertEqual(gap["gap"], 0)
        self.assertIsNone(gap["weaker"])

    def test_by_language_still_reports_both_sides(self):
        results = [scored_case("fr1", "nominal-fr", "fr", major=2),
                   scored_case("en1", "nominal-en", "en")]
        by_language = bench.totals_by_language(results)
        self.assertEqual(by_language["fr"]["major"], 2)
        self.assertEqual(by_language["en"]["major"], 0)

    def test_the_report_carries_the_language_table(self):
        results = {"campaign": "x", "configurations": {
            "m": {"totals": {"fatal": 0, "major": 3, "minor": 0, "parse_rate": 1.0,
                             "truncated": 0, "p50_seconds": 1.0, "p95_seconds": 1.0},
                  "mirror_gap": bench.mirror_gap([
                      scored_case("fr1", "nominal-fr", "fr"),
                      scored_case("en1", "nominal-en", "en", major=3)])}}}
        rendered = bench.render_report(results)
        self.assertIn("Language symmetry", rendered)
        self.assertIn("| `m` |", rendered.split("Language symmetry")[1])
        self.assertIn("en", rendered.split("Language symmetry")[1])


class OtherLanguageTest(unittest.TestCase):
    """A user writing in neither French nor English must be able to decide too.

    The model is their choice (`VELESDB_MEMORY_EXTRACTOR_MODEL`); this suite
    answers for French and English, and `--cases` is what makes the same verdict
    reachable for any other language.
    """

    def test_an_alternate_cases_file_is_accepted(self):
        cases = {"version": 1, "cases": [{
            "id": "de-possessive", "family": "nominal-de", "lang": "de",
            "passages": [{"text": "Marie Dupont hat eine Schwester, Camille Dupont.",
                          "checks": [{"type": "relation_present",
                                      "subject": "camille dupont",
                                      "predicate_any": ["schwester"],
                                      "object": "marie dupont",
                                      "severity": "major", "label": "sibling edge missing"}]}],
        }]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "de.json"
            path.write_text(json.dumps(cases), encoding="utf-8")
            loaded = bench.load_cases(path)
        self.assertEqual(loaded[0]["lang"], "de")
        payload = {"relations": [{"subject": "camille dupont", "predicate": "schwester von",
                                  "object": "marie dupont"}]}
        self.assertEqual(bench.score_passage(payload, loaded[0]["passages"][0]["checks"]), [])

    def test_a_suite_without_mirrors_reports_no_language_verdict(self):
        """No mirrored families means no comparison — and it must say so, not invent one."""
        gap = bench.mirror_gap([scored_case("de1", "nominal-de", "de", major=2)])
        self.assertIsNone(gap["weaker"])
        self.assertEqual(gap["gap"], 0)


# ----------------------------------------------------------------- storage ----


class ColdLoadConditionsTest(unittest.TestCase):
    """Cold-load time measures the disk and the page cache as much as the model.

    Both defects are real on this machine: weights are being split between an
    internal SSD and an external one measured at ~2.7 GB/s, and 64 GiB of RAM
    keeps a 28 GB model resident after its first load — a second "cold" load
    then reports a throughput no SSD can reach.
    """

    def test_a_speed_no_disk_can_reach_is_called_page_cache(self):
        # 28 GB in 2.5 s = 11 200 MB/s, against a 2 703 MB/s device.
        verdict = bench.page_cache_verdict(28_000_000_000, 2.5, 2703.0)
        self.assertEqual(verdict, "page-cache")

    def test_a_plausible_disk_speed_is_called_cold(self):
        verdict = bench.page_cache_verdict(28_000_000_000, 11.0, 2703.0)
        self.assertEqual(verdict, "cold")

    def test_an_unmeasured_device_yields_unknown_not_a_guess(self):
        """A mislabelled cold load is worse than an absent one."""
        self.assertEqual(bench.page_cache_verdict(28_000_000_000, 2.5, None), "unknown")

    def test_missing_size_yields_unknown(self):
        self.assertEqual(bench.page_cache_verdict(None, 2.5, 2703.0), "unknown")

    def test_the_report_marks_a_cached_figure(self):
        results = {"campaign": "x", "configurations": {
            "m": {"totals": {"fatal": 0, "major": 0, "minor": 0, "parse_rate": 1.0,
                             "truncated": 0, "p50_seconds": 1.0, "p95_seconds": 1.0},
                  "cold": {"cold_total_seconds": 3.0, "cache_state": "page-cache"}}}}
        self.assertIn("cache", bench.render_report(results))


class StorageBackingTest(unittest.TestCase):
    def test_a_real_path_reports_its_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            backing = bench.storage_backing(tmp)
        self.assertTrue(backing["reachable"])
        self.assertIn("device", backing)

    def test_a_dead_symlink_is_unreachable_with_a_reason(self):
        """A model behind a link to an unplugged volume is a cable problem."""
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / "model"
            link.symlink_to(Path(tmp) / "absent-volume" / "weights")
            backing = bench.storage_backing(str(link))
        self.assertFalse(backing["reachable"])
        self.assertIn("unplugged", backing["reason"])

    def test_no_path_is_unreachable_not_an_exception(self):
        self.assertFalse(bench.storage_backing(None)["reachable"])

    def test_preflight_refuses_unreachable_weights(self):
        """It must NOT be scored as a failing model."""
        residency = {"models": [{"id": "m", "model_path": "/Volumes/absent/m",
                                 "estimated_size": 1}]}
        with self.assertRaises(RuntimeError) as raised:
            bench.preflight(residency, "m")
        self.assertIn("storage problem, not a model result", str(raised.exception))

    def test_preflight_passes_a_reachable_model_and_carries_its_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            residency = {"models": [{"id": "m", "model_path": tmp, "estimated_size": 4242}]}
            checked = bench.preflight(residency, "m")
        self.assertEqual(checked["estimated_size"], 4242)
        self.assertTrue(checked["backing"]["reachable"])

    def test_a_backend_reporting_no_models_does_not_block_the_run(self):
        """Ollama has no such endpoint; absence of a record is not a failure."""
        self.assertEqual(bench.preflight({}, "m")["estimated_size"], None)


# --------------------------------------------------------------- warm-up ----


class WarmUpTest(unittest.TestCase):
    def test_stability_needs_a_full_window(self):
        self.assertFalse(bench.is_stable([10.0, 10.0]))

    def test_settled_latencies_are_stable(self):
        self.assertTrue(bench.is_stable([50.0, 10.1, 10.0, 9.9]))

    def test_a_drifting_model_is_not_stable(self):
        self.assertFalse(bench.is_stable([10.0, 13.0, 17.0]))

    def test_warm_up_reports_failure_to_settle_instead_of_raising(self):
        """Not stabilising is a finding about the model, not a bench error."""

        class Drifting:
            def __init__(self):
                self.calls = 0

            def generate(self, _prompt):
                self.calls += 1
                return {"seconds": float(self.calls) * 3}

        trace = bench.warm_up(Drifting(), "prompt")
        self.assertFalse(trace["stabilised"])
        self.assertEqual(trace["rounds"], bench.WARMUP_MAX_ROUNDS)


# ----------------------------------------------------------------- report ----


class ReportTest(unittest.TestCase):
    def test_report_is_derived_from_the_results(self):
        results = {
            "campaign": "2026-08-15",
            "configurations": {
                "default:fast": {
                    "totals": {"fatal": 0, "major": 1, "minor": 0, "parse_rate": 1.0,
                               "truncated": 0, "p50_seconds": 10.4, "p95_seconds": 11.2},
                    "cold": {"cold_total_seconds": 92.3},
                    "warmup": {"rounds": 3, "stabilised": True},
                    "origin": {"machine": "M5 Pro", "os": "macOS-26.5.2"},
                },
            },
        }
        rendered = bench.render_report(results)
        self.assertIn("`default:fast`", rendered)
        self.assertIn("10.4s", rendered)
        self.assertIn("92.3s", rendered)
        self.assertIn("M5 Pro", rendered)

    def test_a_missing_cold_load_is_reported_not_faked(self):
        results = {"campaign": "x", "configurations": {
            "m": {"totals": {"fatal": 0, "major": 0, "minor": 0, "parse_rate": 1.0,
                             "truncated": 0, "p50_seconds": 1.0, "p95_seconds": 1.0}}}}
        self.assertIn("n/a", bench.render_report(results))

    def test_an_unstable_warm_up_is_flagged_in_the_table(self):
        results = {"campaign": "x", "configurations": {
            "m": {"totals": {"fatal": 0, "major": 0, "minor": 0, "parse_rate": 1.0,
                             "truncated": 0, "p50_seconds": 1.0, "p95_seconds": 1.0},
                  "warmup": {"rounds": 6, "stabilised": False}}}}
        self.assertIn("unstable", bench.render_report(results))

    def test_report_is_reproducible(self):
        """Same input, same bytes — the check that the table was not hand-edited."""
        results = json.loads(json.dumps({"campaign": "x", "configurations": {}}))
        self.assertEqual(bench.render_report(results), bench.render_report(results))


# ---------------------------------------------------------------- provenance --


DIGEST = "bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8"
RECORDED_OPTIONS = {"num_ctx": 2048, "num_predict": 512, "temperature": 0, "constrained": False}
PUBLISHED_CAMPAIGN = SCRIPT_PATH.parents[1] / "benchmarks" / "results" / "2026-08-16-campagne"


class FakeOllama:
    """An Ollama that answers from a table and keeps every body it is sent.

    A real server would make these tests depend on what the machine running them
    has installed. What they pin is what the bench does with an answer, and with
    the absence of one. `version=None` turns `/api/version` into a 404: a proxy
    that serves generation and nothing else.
    """

    def __init__(self, test: unittest.TestCase, version: "str | None" = "0.34.0",
                 models: "list[dict] | None" = None) -> None:
        self.bodies: "list[dict]" = []
        routes = {"/api/ps": {"models": []}, "/api/tags": {"models": models or []}}
        if version is not None:
            routes["/api/version"] = {"version": version}
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), self._handler(routes))
        thread = threading.Thread(target=server.serve_forever,
                                  kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        # Cleanups run last-in first-out: stop the loop, close the socket, join.
        test.addCleanup(thread.join)
        test.addCleanup(server.server_close)
        test.addCleanup(server.shutdown)
        self.url = f"http://127.0.0.1:{server.server_address[1]}"

    def _handler(self, routes: dict) -> type:
        bodies = self.bodies

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self._answer(routes.get(self.path))

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                bodies.append(json.loads(self.rfile.read(length)))
                self._answer({"response": '{"relations": [], "attributes": []}',
                              "eval_count": 1, "done_reason": "stop"})

            def _answer(self, payload: "dict | None") -> None:
                body = b"" if payload is None else json.dumps(payload).encode()
                self.send_response(404 if payload is None else 200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                """Keep the per-request line off the test output."""

        return Handler


def closed_port_url() -> str:
    """A loopback URL nothing listens on: an Ollama that is not running."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def utc_millisecond() -> datetime.datetime:
    """Now, cut to the millisecond a result file records, so a bound compares like with like."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.replace(microsecond=now.microsecond - now.microsecond % 1000)


def recorded_start(test: unittest.TestCase, record: dict) -> datetime.datetime:
    """The start a result file records, checked to be ISO 8601 in UTC."""
    stamp = record.get("started_at")
    test.assertIsInstance(stamp, str, "the result file records no start time")
    started = datetime.datetime.fromisoformat(stamp)
    test.assertEqual(started.utcoffset(), datetime.timedelta(0))
    return started


def screen_via_main(fake: FakeOllama, cases: Path, out: Path, *extra: str) -> dict:
    """`screen` through `main` against `fake`, as a campaign runs it: the file it wrote."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        bench.main(["screen", "--backend", "ollama", "--url", fake.url,
                    "--config", "qwen3:14b", "--cases", str(cases),
                    "--out", str(out), *extra])
    return json.loads(out.read_text(encoding="utf-8"))


ENDTOEND_OUTCOME = {"phase": "endtoend", "write_p50_seconds": 0.1, "write_p95_seconds": 0.2,
                    "cases": [], "burst": {"count": 0, "p95_seconds": 0.1, "autograph_dropped": 0},
                    "totals": {"fatal": 0, "major": 0, "max_drain_seconds": 0.0}}


# What the stand-in daemon prints for `--version`, in velesdb-memory's own form.
FAKE_BINARY_VERSION = "velesdb-memory 0.0.0-bench-test"
FAKE_BINARY_SCRIPT = f"#!/bin/sh\necho '{FAKE_BINARY_VERSION}'\n"


def endtoend_via_main(out: Path, *extra: str, script: str = FAKE_BINARY_SCRIPT) -> dict:
    """`endtoend` through `main`, with no daemon spawned and no model run: the file it wrote.

    What is pinned is the record `endtoend` writes around the phase. The closed
    port would turn any request the bench made into a different reason. The
    binary it is given runs `script`, from outside the repository: `endtoend`
    only asks it `--version`.
    """
    with tempfile.TemporaryDirectory() as bin_dir, \
            mock.patch.object(bench, "DisposableDaemon"), \
            mock.patch.object(bench, "endtoend", return_value=copy.deepcopy(ENDTOEND_OUTCOME)), \
            contextlib.redirect_stdout(io.StringIO()):
        binary = Path(bin_dir) / "velesdb-memory"
        binary.write_text(script, encoding="utf-8")
        binary.chmod(0o755)
        bench.main(["endtoend", "--backend", "ollama", "--url", closed_port_url(),
                    "--config", "qwen3:14b", "--binary", str(binary),
                    "--out", str(out), *extra])
    return json.loads(out.read_text(encoding="utf-8"))


def launched_fake_binary() -> dict:
    """The `binary` record of a run that launched `FAKE_BINARY_SCRIPT` from outside the repository."""
    return {"path": None, "version": FAKE_BINARY_VERSION,
            "sha256": hashlib.sha256(FAKE_BINARY_SCRIPT.encode("utf-8")).hexdigest(),
            "missing": {"path": bench.OUTSIDE_REPOSITORY}}


def served_by(fake: "FakeOllama") -> dict:
    """The `binary` record of a run screened against `fake`: its URL and version, unhashed."""
    return {"url": fake.url, "version": "0.34.0", "sha256": None,
            "missing": {"sha256": bench.SERVED_OVER_HTTP}}


def checkout_origin() -> dict:
    """Where these tests run from, asked the way a result file must record it."""
    def git(*argv: str) -> str:
        return subprocess.run(["git", "-C", str(SCRIPT_PATH.parents[1]), *argv],
                              capture_output=True, text=True, check=True).stdout.strip()
    return {"machine": platform.machine(), "os": platform.platform(terse=True),
            "commit": git("rev-parse", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no"))}


def origin_cell(origin: dict) -> str:
    """The origin cell a row renders, restated: host, OS, the commit cut to 12, marked when
    tracked files had changed, the cases file, then a recorded binary by the version it
    reported and its sha256 cut to 12, or `unhashed`; `?` for what the run could not answer."""
    commit = origin.get("commit")
    mark = {True: " (modified)", None: " (modified ?)"}.get(origin.get("uncommitted_changes"), "")
    parts = [origin.get("machine") or "?", origin.get("os") or "?",
             f"{commit[:12]}{mark}" if commit else "?", origin.get("cases_file") or "?"]
    binary = origin.get("binary")
    if isinstance(binary, dict):
        digest = binary.get("sha256")
        parts.append(f"{binary.get('version') or '?'} ({digest[:12] if digest else 'unhashed'})")
    return " · ".join(parts)


class RuntimeRecordTest(unittest.TestCase):
    """A result names the server build and the weights that produced it (#1949).

    The 2026-08-16 campaign could not: `qwen3:4b-instruct` is a moving tag, the
    llama.cpp behind Ollama changes with its version, and when a re-run on
    another machine disagreed, no file could say which of the two had moved.
    """

    def test_the_build_and_the_digest_are_asked_of_the_server(self):
        fake = FakeOllama(self, models=[{"name": "qwen3:14b", "digest": DIGEST}])
        runtime = bench.OllamaBackend(fake.url, "qwen3:14b", 512).runtime()
        self.assertEqual(runtime, {"ollama_version": "0.34.0",
                                   "model_digests": {"qwen3:14b": DIGEST}, "missing": {}})

    def test_an_untagged_name_is_found_under_its_latest_tag(self):
        """Ollama lists a model pulled as `bge-m3` under `bge-m3:latest`."""
        fake = FakeOllama(self, models=[{"name": "bge-m3:latest", "digest": DIGEST}])
        self.assertEqual(bench.ollama_runtime(fake.url, ["bge-m3"])["model_digests"],
                         {"bge-m3": DIGEST})

    def test_a_server_that_does_not_answer_is_recorded_as_silent(self):
        runtime = bench.ollama_runtime(closed_port_url(), ["qwen3:14b"])
        self.assertIsNone(runtime["ollama_version"])
        self.assertEqual(runtime["model_digests"], {"qwen3:14b": None})
        self.assertIn("no answer from GET /api/version", runtime["missing"]["ollama_version"])
        self.assertIn("no answer from GET /api/tags", runtime["missing"]["qwen3:14b"])

    def test_one_unanswered_question_does_not_erase_the_other_answer(self):
        fake = FakeOllama(self, version=None, models=[{"name": "qwen3:14b", "digest": DIGEST}])
        runtime = bench.ollama_runtime(fake.url, ["qwen3:14b"])
        self.assertIsNone(runtime["ollama_version"])
        self.assertIn("404", runtime["missing"]["ollama_version"])
        self.assertEqual(runtime["model_digests"], {"qwen3:14b": DIGEST})

    def test_a_model_the_server_does_not_list_has_no_digest_and_a_reason(self):
        fake = FakeOllama(self, models=[{"name": "qwen3:8b", "digest": DIGEST}])
        runtime = bench.ollama_runtime(fake.url, ["qwen3:14b"])
        self.assertEqual(runtime["model_digests"], {"qwen3:14b": None})
        self.assertEqual(runtime["missing"]["qwen3:14b"], "not listed by GET /api/tags")

    def test_a_backend_the_bench_does_not_ask_says_so(self):
        """No request is made: a closed port would turn one into an error here."""
        for backend in (bench.GitHubModelsBackend("m", "token", 512),
                        bench.OpenAiBackend(closed_port_url(), "m", "token", 512)):
            with self.subTest(backend=type(backend).__name__):
                runtime = backend.runtime()
                self.assertIsNone(runtime["ollama_version"])
                self.assertEqual(runtime["model_digests"], {"m": None})
                self.assertTrue(runtime["missing"])

    def test_the_recorded_options_are_the_ones_the_request_carried(self):
        """`settings()` against the body on the wire, with and without `format`."""
        fake = FakeOllama(self)
        for schema in (None, bench.EXTRACTION_SCHEMA):
            with self.subTest(constrained=schema is not None):
                backend = bench.OllamaBackend(fake.url, "m", 512, num_ctx=4096, schema=schema)
                backend.generate("prompt")
                recorded, sent = backend.settings(), fake.bodies[-1]
                for option in ("num_ctx", "num_predict", "temperature"):
                    self.assertEqual(recorded[option], sent["options"][option])
                self.assertEqual(recorded["constrained"], "format" in sent)


class ProvenanceReportTest(unittest.TestCase):
    """A campaign whose files cannot say what produced them reads `unverified`."""

    def test_the_published_campaign_reads_unverified(self):
        """No 2026-08-16 file records a server build, a digest or its options."""
        configurations = bench.merge_configurations(PUBLISHED_CAMPAIGN)
        rendered = bench.render_report({"configurations": configurations})
        count = len(configurations)
        self.assertGreater(count, 0)
        self.assertIn(f"**Unverified: {count} of {count} rows**", rendered)
        rows = report_table(rendered, "## Configurations")
        self.assertEqual(len(rows), count)
        self.assertEqual({row.get("provenance") for row in rows.values()}, {"unverified"})

    def test_only_the_rows_without_a_record_are_unverified(self):
        recorded = {"config": "a", "settings": dict(RECORDED_OPTIONS),
                    "runtime": {"ollama_version": "0.34.0", "model_digests": {"a": DIGEST}}}
        # Options and no runtime: what a file written between #1955 and this carries.
        options_only = {"config": "b", "settings": dict(RECORDED_OPTIONS)}
        rendered = bench.render_report({"configurations": {"a": recorded, "b": options_only}})
        self.assertIn("**Unverified: 1 of 2 rows**", rendered)
        self.assertIn(f"| ollama 0.34.0 · {DIGEST[:12]} · ctx 2048 |", rendered)
        self.assertIn("`ollama_version`, `digest`", rendered)
        self.assertNotIn("`num_ctx`", rendered)


class ProvenanceWiringTest(unittest.TestCase):
    """`screen` writes the record and `report --from-dir` prints it, through `main`.

    A record no subcommand writes, or a label no report prints, protects
    nothing: both ends run here against a fake server, as a campaign runs them.
    """

    ONE_CASE = {"version": 1, "cases": [{
        "id": "one", "family": "nominal-en", "lang": "en",
        "passages": [{"text": "Nothing in this sentence relates to anything.",
                      "checks": [{"type": "relations_empty", "severity": "minor",
                                  "label": "a relation was invented"}]}],
    }]}

    def screen_into(self, fake: FakeOllama, root: Path, name: str, *extra: str) -> dict:
        """`screen` through `main`, into `root/campaign/name`: the file it wrote."""
        cases = root / "cases.json"
        cases.write_text(json.dumps(self.ONE_CASE), encoding="utf-8")
        return screen_via_main(fake, cases, root / "campaign" / name, *extra)

    @staticmethod
    def report_of(campaign: Path) -> str:
        """`report --from-dir` through `main`, as a campaign renders it."""
        report = io.StringIO()
        with contextlib.redirect_stdout(report):
            bench.main(["report", "--from-dir", str(campaign)])
        return report.getvalue()

    def screen_then_report(self, fake: FakeOllama, *extra: str) -> "tuple[dict, str]":
        with tempfile.TemporaryDirectory() as tmp:
            record = self.screen_into(fake, Path(tmp), "qwen3-14b.json", *extra)
            return record, self.report_of(Path(tmp) / "campaign")

    def test_a_screened_file_carries_its_runtime_and_the_report_prints_it(self):
        fake = FakeOllama(self, models=[{"name": "qwen3:14b", "digest": DIGEST}])
        record, rendered = self.screen_then_report(fake, "--num-ctx", "4096")
        self.assertEqual(record["runtime"], {"ollama_version": "0.34.0",
                                             "model_digests": {"qwen3:14b": DIGEST},
                                             "missing": {}})
        self.assertEqual(record["settings"]["num_ctx"], 4096)
        self.assertIn(f"| ollama 0.34.0 · {DIGEST[:12]} · ctx 4096 |", rendered)
        self.assertIn("Every row's result file records", rendered)
        # The bench's default, read when the report is rendered, must not stand
        # in for what a run sent: the 2026-08-16 report printed `num_ctx: 2048`
        # under Environment while none of its result files recorded one.
        self.assertNotIn("**num_ctx**", rendered)

    def test_a_screened_file_records_its_suite_and_the_report_prints_it(self):
        fake = FakeOllama(self, models=[{"name": "qwen3:14b", "digest": DIGEST}])
        record, rendered = self.screen_then_report(fake)
        digest = scored_digest(self.ONE_CASE["cases"], "screen")
        self.assertEqual(record.get("suite"), {"cases": 1, "definitions_sha256": digest})
        rows = report_table(rendered, "## Configurations")
        self.assertEqual(rows["qwen3:14b"].get("suite"), f"1 case · {digest[:12]}")

    def test_a_screened_file_records_where_it_ran_and_the_report_prints_it(self):
        """Host, commit and cases file are the run's own, never the renderer's (#1949).

        This cases file lies outside the repository: its local path is not
        recorded, and the gap says why.
        """
        fake = FakeOllama(self, models=[{"name": "qwen3:14b", "digest": DIGEST}])
        record, rendered = self.screen_then_report(fake)
        self.assertIsInstance(record.get("origin"), dict, "the result file records no origin")
        self.assertEqual(record["origin"], {**checkout_origin(), "cases_file": None,
                                            "binary": served_by(fake),
                                            "missing": {"cases_file": bench.OUTSIDE_REPOSITORY}})
        rows = report_table(rendered, "## Configurations")
        self.assertEqual(rows["qwen3:14b"].get("origin"), origin_cell(record["origin"]))
        self.assertNotIn("## Environment", rendered)

    def test_a_screened_file_records_the_server_it_reached_and_the_report_prints_it(self):
        """Screening launches no binary: it reaches a server over HTTP (#1949).

        That server's executable is out of the bench's reach, so the run records
        how it reached it and the version the server answered, and why it has
        no sha256.
        """
        fake = FakeOllama(self, models=[{"name": "qwen3:14b", "digest": DIGEST}])
        record, rendered = self.screen_then_report(fake)
        self.assertIsInstance(record["origin"].get("binary"), dict,
                              "the result file records no binary")
        self.assertEqual(record["origin"]["binary"], served_by(fake))
        cell = report_table(rendered, "## Configurations")["qwen3:14b"]["origin"]
        self.assertTrue(cell.endswith(" · 0.34.0 (unhashed)"), cell)

    def test_a_screened_file_records_its_start_and_the_report_orders_runs_by_it(self):
        """Screened into `99-…` first and `01-…` second, the replay is `01-…` (#1949).

        Before the start was recorded, the file names alone decided, and `01-…`
        would have read as the reference.
        """
        fake = FakeOllama(self, models=[{"name": "qwen3:14b", "digest": DIGEST}])
        with tempfile.TemporaryDirectory() as tmp:
            before = utc_millisecond()
            first = self.screen_into(fake, Path(tmp), "99-qwen3-14b.json")
            second = self.screen_into(fake, Path(tmp), "01-qwen3-14b.json")
            after = datetime.datetime.now(datetime.timezone.utc)
            rendered = self.report_of(Path(tmp) / "campaign")
        starts = [recorded_start(self, record) for record in (first, second)]
        self.assertLessEqual(before, starts[0])
        self.assertLess(starts[0], starts[1])
        self.assertLessEqual(starts[1], after)
        rows = report_table(rendered, "## Configurations")
        self.assertEqual(list(rows), ["qwen3:14b", "qwen3:14b (replay: 01-qwen3-14b.json)"])

    def test_a_run_the_server_would_not_identify_is_reported_unverified(self):
        fake = FakeOllama(self, version=None, models=[{"name": "qwen3:14b", "digest": DIGEST}])
        record, rendered = self.screen_then_report(fake)
        self.assertIn("404", record["runtime"]["missing"]["ollama_version"])
        # Asked once: the binary record carries the answer `runtime` got.
        binary = record["origin"].get("binary") or {}
        self.assertEqual(binary.get("missing", {}).get("version"),
                         record["runtime"]["missing"]["ollama_version"])
        self.assertIn("**Unverified: 1 of 1 rows**", rendered)
        self.assertIn("`ollama_version`", rendered)


# ------------------------------------------------------------------ run count --


def report_table(rendered: str, heading: str) -> "dict[str, dict[str, str]]":
    """One table of a rendered report: each row's cells by column, keyed by configuration."""
    section = rendered.split(heading, 1)[1].split("\n## ", 1)[0]
    lines = section.splitlines()
    header = next(line for line in lines if line.startswith("| configuration"))
    columns = [cell.strip() for cell in header.strip("|").split("|")]
    rows = {}
    for line in lines:
        if line.startswith("| `"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            rows[cells[0].strip("`")] = dict(zip(columns, cells))
    return rows


class RunCountTest(unittest.TestCase):
    """A row's counts are sums over its runs, and the row says how many (#1949).

    The 2026-08-16 campaign screened every Ollama model twice and every MLX one
    once, then printed both sums side by side: `qwen3:14b`'s 8 majors read as
    worse than `default:base`'s 7, when it made 4 per run.
    """

    FAILING_CASE = {"version": 1, "cases": [{
        "id": "one", "family": "nominal-en", "lang": "en",
        "passages": [{"text": "Alice works at Acme.",
                      "checks": [{"type": "relation_present", "subject": "alice",
                                  "predicate_any": ["works at"], "object": "acme",
                                  "severity": "major", "label": "employer edge missing"}]}],
    }]}

    def screened(self, runs: int) -> dict:
        """The real `screen`, against a server whose every reply misses the edge."""
        fake = FakeOllama(self)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(json.dumps(self.FAILING_CASE), encoding="utf-8")
            cases = bench.load_cases(path)
        with contextlib.redirect_stdout(io.StringIO()):
            return bench.screen(bench.OllamaBackend(fake.url, "m", 512),
                                bench.read_graph_prompt_template(), cases, runs=runs)

    def test_each_table_prints_the_runs_its_counts_sum(self):
        once, twice = self.screened(1), self.screened(2)
        # The same reply graded twice is counted twice: the misreading itself.
        self.assertGreater(once["totals"]["major"], 0)
        self.assertEqual(twice["totals"]["major"], 2 * once["totals"]["major"])
        rendered = bench.render_report({"configurations": {"once": once, "twice": twice}})
        for heading in ("## Configurations", "## Language symmetry"):
            with self.subTest(table=heading):
                rows = report_table(rendered, heading)
                self.assertEqual([rows[name].get("runs") for name in ("once", "twice")],
                                 ["1", "2"])

    def test_rows_summing_different_runs_are_flagged_above_the_table(self):
        once, twice = self.screened(1), self.screened(2)
        mixed = bench.render_report({"configurations": {"once": once, "twice": twice}})
        flag = mixed.find("**Rows sum different numbers of runs**")
        self.assertGreater(flag, 0)
        self.assertLess(flag, mixed.index("| configuration |"))
        uniform = bench.render_report({"configurations": {"a": twice, "b": twice}})
        self.assertNotIn("different numbers of runs", uniform)

    def test_the_published_rows_print_the_runs_their_files_hold(self):
        configurations = bench.merge_configurations(PUBLISHED_CAMPAIGN)
        self.assertGreater(len({entry["runs"] for entry in configurations.values()}), 1)
        rendered = bench.render_report({"configurations": configurations})
        rows = report_table(rendered, "## Configurations")
        self.assertEqual({name: row.get("runs") for name, row in rows.items()},
                         {name: str(entry["runs"]) for name, entry in configurations.items()})
        self.assertIn("**Rows sum different numbers of runs**", rendered)


# ---------------------------------------------------------------- end-to-end --


class EndToEndRuntimeTest(unittest.TestCase):
    """Phase B cannot name the Ollama that served it, and its record says so (#1949).

    The daemon under test talks to Ollama itself, at endpoints it resolves on its
    own. Asking the URL the bench believes it uses would write a guess down as an
    observation, so the record is null with its reason, and the report reads the
    file `unverified` for that reason.
    """

    def endtoend_then_report(self, *extra: str,
                             script: str = FAKE_BINARY_SCRIPT) -> "tuple[dict, str]":
        """`endtoend`, then `report --from-dir`, through `main`: see `endtoend_via_main`."""
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "campaign"
            campaign.mkdir()
            record = endtoend_via_main(campaign / "B-qwen3-14b.json", *extra, script=script)
            report = io.StringIO()
            with contextlib.redirect_stdout(report):
                bench.main(["report", "--from-dir", str(campaign)])
            return record, report.getvalue()

    def test_an_endtoend_file_says_why_it_names_no_runtime_and_the_report_repeats_it(self):
        record, rendered = self.endtoend_then_report()
        self.assertEqual(record.get("runtime"), {
            "ollama_version": None,
            "model_digests": {"qwen3:14b": None, "bge-m3": None},
            "missing": {"runtime": "the daemon resolves its own endpoint"}})
        rows = report_table(rendered, "## End-to-end")
        self.assertEqual(rows["qwen3:14b"].get("provenance"),
                         "unverified (the daemon resolves its own endpoint)")

    def test_an_endtoend_file_records_the_suite_it_stored_and_the_report_prints_it(self):
        """The stubbed phase stores no case row: the cell can only come from the record."""
        with tempfile.TemporaryDirectory() as tmp:
            cases = Path(tmp) / "cases.json"
            cases.write_text(json.dumps(ProvenanceWiringTest.ONE_CASE), encoding="utf-8")
            record, rendered = self.endtoend_then_report("--cases", str(cases))
        digest = scored_digest(ProvenanceWiringTest.ONE_CASE["cases"], "endtoend")
        self.assertEqual(record.get("suite"), {"cases": 1, "definitions_sha256": digest})
        rows = report_table(rendered, "## End-to-end")
        self.assertEqual(rows["qwen3:14b"].get("suite"), f"1 case · {digest[:12]}")

    def test_an_endtoend_file_records_where_it_ran_and_its_cases_file_in_the_repository(self):
        """The shipped cases file is recorded by its path in the repository, never a local one."""
        record, rendered = self.endtoend_then_report()
        self.assertIsInstance(record.get("origin"), dict, "the result file records no origin")
        self.assertEqual(record["origin"], {**checkout_origin(),
                                            "cases_file": "scripts/memory-extraction-cases.json",
                                            "binary": launched_fake_binary(),
                                            "missing": {}})
        rows = report_table(rendered, "## End-to-end")
        self.assertEqual(rows["qwen3:14b"].get("origin"), origin_cell(record["origin"]))

    def test_an_endtoend_file_records_the_binary_it_launched_and_the_report_prints_it(self):
        """A commit does not identify a binary: it may have been built from another tree (#1949).

        The run records what the executable says of itself, its `--version`, and
        the sha256 of its bytes. This one lies outside the repository: its local
        path is not recorded, and the gap says why.
        """
        record, rendered = self.endtoend_then_report()
        self.assertIsInstance(record["origin"].get("binary"), dict,
                              "the result file records no binary")
        expected = launched_fake_binary()
        self.assertEqual(record["origin"]["binary"], expected)
        cell = report_table(rendered, "## End-to-end")["qwen3:14b"]["origin"]
        self.assertTrue(cell.endswith(f" · {FAKE_BINARY_VERSION} ({expected['sha256'][:12]})"),
                        cell)

    def test_a_binary_that_will_not_say_its_version_is_recorded_with_the_reason(self):
        """Never filled in: a version the binary did not print stays null, and says why."""
        script = "#!/bin/sh\nexit 3\n"
        record, _rendered = self.endtoend_then_report(script=script)
        self.assertEqual(record["origin"].get("binary"), {
            "path": None, "version": None,
            "sha256": hashlib.sha256(script.encode("utf-8")).hexdigest(),
            "missing": {"path": bench.OUTSIDE_REPOSITORY,
                        "version": "velesdb-memory --version exited 3"}})

    def test_an_endtoend_file_records_when_it_started(self):
        before = utc_millisecond()
        record, _rendered = self.endtoend_then_report()
        after = datetime.datetime.now(datetime.timezone.utc)
        started = recorded_start(self, record)
        self.assertLessEqual(before, started)
        self.assertLessEqual(started, after)

    def test_the_published_endtoend_runs_read_unverified(self):
        """They predate any runtime record: no reason to repeat, only the gap."""
        runs = bench.merge_configurations(PUBLISHED_CAMPAIGN, "endtoend")
        rows = report_table(bench.render_report({"end_to_end": runs}), "## End-to-end")
        self.assertEqual(set(rows), set(runs))
        self.assertEqual({row.get("provenance") for row in rows.values()}, {"unverified"})


# -------------------------------------------------------------- row per file --


def write_result(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry), encoding="utf-8")


def timed_entry(config: str, p50: float, rounds: int, started_at: "str | None" = None) -> dict:
    """A screening result that differs from another only by its timings, and its start."""
    entry = {"phase": "screen", "config": config, "runs": 1,
             "totals": {"fatal": 0, "major": 0, "minor": 0, "parse_rate": 1.0, "truncated": 0,
                        "p50_seconds": p50, "p95_seconds": p50},
             "warmup": {"rounds": rounds, "stabilised": True}}
    if started_at is not None:
        entry["started_at"] = started_at
    return entry


def rendered_rows(campaign: Path) -> "list[tuple[str, str, str]]":
    """Each screening row of a campaign's report, in table order: label, p50, warm-up."""
    rows = report_table(bench.render_report(
        {"configurations": bench.merge_configurations(campaign)}), "## Configurations")
    return [(name, row["p50"], row["warm-up"]) for name, row in rows.items()]


def timing_cells(entry: dict) -> "tuple[str, str]":
    """The p50 and warm-up cells a row renders for this result file."""
    return f"{entry['totals']['p50_seconds']:.1f}s", str(entry["warmup"]["rounds"])


class RowPerFileTest(unittest.TestCase):
    """Every result file is a row of its own, and none replaces another (#1949).

    The 2026-08-16 order control screened `qwen3:4b-instruct` first and again
    last, as `01-…` and `99-…` in one directory. Both took one label, the replay
    replaced the reference, and the published row printed the replay's timings
    under the reference's name.
    """

    def test_a_configuration_screened_twice_keeps_both_rows(self):
        first = timed_entry("m", 1.2, 3, "2026-08-16T10:00:00.000+00:00")
        replay = timed_entry("m", 1.8, 6, "2026-08-16T10:40:00.000+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "campaign"
            write_result(campaign / "ref" / "01-m.json", first)
            write_result(campaign / "ref" / "99-m.json", replay)
            configurations = bench.merge_configurations(campaign)
        rows = report_table(bench.render_report({"configurations": configurations}),
                            "## Configurations")
        self.assertEqual({name: (row["p50"], row["warm-up"]) for name, row in rows.items()},
                         {"m [ref]": timing_cells(first),
                          "m [ref] (replay: 99-m.json)": timing_cells(replay)})

    def test_the_run_that_started_first_is_the_reference_whatever_its_file_name(self):
        """Names numbered against the run order: the recorded start decides, not the name."""
        later = timed_entry("m", 1.8, 6, "2026-08-16T10:40:00.000+00:00")
        earlier = timed_entry("m", 1.2, 3, "2026-08-16T10:00:00.000+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "campaign"
            write_result(campaign / "ref" / "01-m.json", later)
            write_result(campaign / "ref" / "99-m.json", earlier)
            rows = rendered_rows(campaign)
        self.assertEqual(rows, [("m [ref]", *timing_cells(earlier)),
                                ("m [ref] (replay: 01-m.json)", *timing_cells(later))])

    def test_runs_a_start_cannot_order_are_ordered_by_name_and_say_so(self):
        """No start, a start without its offset, or one start of two: the names decide."""
        for starts in ((None, None), ("2026-08-16T10:40:00", "2026-08-16T10:00:00"),
                       ("2026-08-16T10:40:00.000+00:00", None)):
            with self.subTest(starts=starts), tempfile.TemporaryDirectory() as tmp:
                first = timed_entry("m", 1.2, 3, starts[0])
                second = timed_entry("m", 1.8, 6, starts[1])
                campaign = Path(tmp) / "campaign"
                write_result(campaign / "ref" / "01-m.json", first)
                write_result(campaign / "ref" / "99-m.json", second)
                self.assertEqual(rendered_rows(campaign), [
                    ("m [ref]", *timing_cells(first)),
                    ("m [ref] (replay by file-name order: 99-m.json)", *timing_cells(second))])

    def test_a_label_two_files_would_share_is_refused(self):
        """Not a replay: two directories' labels meet, and neither file may vanish."""
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "campaign"
            write_result(campaign / "flat.json", timed_entry("m [ref]", 1.0, 3))
            write_result(campaign / "ref" / "m.json", timed_entry("m", 2.0, 3))
            with self.assertRaises(SystemExit) as refusal:
                bench.merge_configurations(campaign)
        self.assertIn("m.json", str(refusal.exception))

    def test_the_published_order_control_prints_both_passes(self):
        files = {path.name: json.loads(path.read_text(encoding="utf-8"))
                 for path in sorted(PUBLISHED_CAMPAIGN.rglob("*.json"))}
        screened = [entry for entry in files.values() if entry.get("phase") == "screen"]
        configurations = bench.merge_configurations(PUBLISHED_CAMPAIGN)
        # Every screening file is a row, as the fold cuts it (see `LocalPathTest`).
        self.assertEqual(
            sorted(json.dumps(entry, sort_keys=True) for entry in configurations.values()),
            sorted(json.dumps(bench.without_local_paths(entry), sort_keys=True)
                   for entry in screened))
        first, replay = files["01-qwen3-4b-instruct.json"], files["99-qwen3-4b-instruct.json"]
        self.assertNotEqual(timing_cells(first), timing_cells(replay))
        rows = report_table(bench.render_report({"configurations": configurations}),
                            "## Configurations")
        label = "qwen3:4b-instruct [reference-ollama]"
        self.assertEqual((rows[label]["p50"], rows[label]["warm-up"]), timing_cells(first))
        # No 2026-08-16 file records its start: which pass is the replay rests on
        # the file names, and the label says so.
        replayed = rows[f"{label} (replay by file-name order: 99-qwen3-4b-instruct.json)"]
        self.assertEqual((replayed["p50"], replayed["warm-up"]), timing_cells(replay))


# --------------------------------------------------------------------- suite --


def ids_digest(*ids: str) -> str:
    """How the report digests a file naming its cases' ids and nothing else, restated:
    sha256 of the sorted ids, one per line."""
    return hashlib.sha256("\n".join(sorted(set(ids))).encode("utf-8")).hexdigest()


def scored_digest(cases: "list[dict]", phase: str) -> str:
    """The suite identity's definition, restated: sha256 of what the phase's scorer reads
    of each case, cases sorted by id, serialised with sorted keys and no whitespace.

    Screening reads a case's id, family, language, cross-checks (none when absent)
    and each passage's text and checks; phase B its id, and each passage's text
    with the checks a stored entity can answer.
    """
    if phase == "screen":
        scored = [{"id": case["id"], "family": case["family"], "lang": case["lang"],
                   "passages": [{"text": passage["text"], "checks": passage["checks"]}
                                for passage in case["passages"]],
                   "cross_checks": case.get("cross_checks") or []} for case in cases]
    else:
        scored = [{"id": case["id"],
                   "passages": [{"text": passage["text"],
                                 "checks": bench.graph_checks_for(passage)}
                                for passage in case["passages"]]} for case in cases]
    canonical = json.dumps(sorted(scored, key=lambda case: case["id"]), sort_keys=True,
                           ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def suite_record(*ids: str) -> dict:
    """A recorded suite, over cases that tell apart by their ids alone."""
    cases = [{"id": case_id, "family": "edge", "lang": "en", "passages": []} for case_id in ids]
    return {"suite": {"cases": len(cases), "definitions_sha256": scored_digest(cases, "screen")}}


class SuiteTest(unittest.TestCase):
    """A row names the scenarios it ran, and rows on different ones are marked (#1949).

    #1955 grew the suite from 19 scenarios to 29, so two campaigns can differ in
    what they ran; and phase B can run a subset: the 2026-08-16 end-to-end rows
    covered 1 case and 4 under one heading, and nothing said so.
    """

    def test_the_identity_covers_what_each_case_checks(self):
        """An edited check is another suite under the same ids; a reordered file is not."""
        cases = bench.load_cases(CASES_PATH)
        edited = copy.deepcopy(cases)
        check = next(check for case in edited for passage in case["passages"]
                     for check in passage["checks"] if check["type"] == "attribute_number")
        check["value"] += 1
        reordered = [dict(reversed(list(case.items()))) for case in reversed(cases)]
        for phase in ("screen", "endtoend"):
            with self.subTest(phase=phase):
                identity = bench.suite_identity(cases, phase)
                self.assertNotEqual(bench.suite_identity(edited, phase), identity)
                self.assertEqual(identity, {"cases": len(cases),
                                            "definitions_sha256": scored_digest(cases, phase)})
                self.assertEqual(bench.suite_identity(reordered, phase), identity)

    def test_rows_off_the_common_suite_are_marked_and_flagged(self):
        old = suite_record(*(f"c{index}" for index in range(19)))
        grown = suite_record(*(f"c{index}" for index in range(29)))
        rendered = bench.render_report({"configurations": {"a": old, "b": old, "c": grown}})
        rows = report_table(rendered, "## Configurations")
        self.assertEqual({name: row.get("suite") for name, row in rows.items()}, {
            "a": f"19 cases · {old['suite']['definitions_sha256'][:12]}",
            "b": f"19 cases · {old['suite']['definitions_sha256'][:12]}",
            "c": f"29 cases · {grown['suite']['definitions_sha256'][:12]} (differs)"})
        flag = rendered.find("**Rows ran different suites**")
        self.assertGreater(flag, 0)
        self.assertLess(flag, rendered.index("| configuration |"))
        uniform = bench.render_report({"configurations": {"a": old, "b": old}})
        self.assertNotIn("different suites", uniform)
        self.assertNotIn("(differs)", uniform)

    def test_a_suite_named_by_its_ids_alone_says_its_checks_are_unrecorded(self):
        """A file with no suite names the cases it scored, not what they checked.

        Its digest is of ids, so it never equals a recorded one, even over the
        same cases: two such files may have been graded against different checks.
        """
        scored_only = {"cases": [{"id": "a"}, {"id": "b"}, {"id": "a"}]}
        recorded = suite_record("a", "b")
        rendered = bench.render_report({"configurations": {"old": scored_only, "new": recorded}})
        rows = report_table(rendered, "## Configurations")
        self.assertEqual({name: row.get("suite") for name, row in rows.items()}, {
            "old": f"2 case ids · {ids_digest('a', 'b')[:12]} (differs)",
            "new": f"2 cases · {recorded['suite']['definitions_sha256'][:12]} (differs)"})
        self.assertIn("**Checks unrecorded: 1 of 2 rows**", rendered)
        self.assertNotIn("Checks unrecorded",
                         bench.render_report({"configurations": {"new": recorded}}))

    def test_the_published_endtoend_rows_show_the_cases_each_covered(self):
        """Older files record no suite; their scored rows still name every case."""
        runs = bench.merge_configurations(PUBLISHED_CAMPAIGN, "endtoend")
        rendered = bench.render_report({"end_to_end": runs})
        rows = report_table(rendered, "## End-to-end")
        covered = {name: {case["id"] for case in entry["cases"]} for name, entry in runs.items()}
        self.assertGreater(len({len(ids) for ids in covered.values()}), 1)
        # Two rows on two suites: neither leads, so both are marked.
        self.assertEqual(
            {name: row.get("suite") for name, row in rows.items()},
            {name: f"{len(ids)} {'case id' if len(ids) == 1 else 'case ids'} · "
                   f"{ids_digest(*ids)[:12]} (differs)" for name, ids in covered.items()})
        section = rendered.split("## End-to-end", 1)[1]
        self.assertIn("**Rows ran different suites**", section)
        self.assertIn(f"**Checks unrecorded: {len(runs)} of {len(runs)} rows**", section)

    def test_the_published_screening_rows_ran_one_suite(self):
        configurations = bench.merge_configurations(PUBLISHED_CAMPAIGN)
        section = bench.render_report({"configurations": configurations}).split(
            "## Language symmetry", 1)[0]
        rows = report_table(section, "## Configurations")
        expected = {name: {case["id"] for case in entry["cases"]}
                    for name, entry in configurations.items()}
        self.assertEqual({name: row.get("suite") for name, row in rows.items()},
                         {name: f"{len(ids)} case ids · {ids_digest(*ids)[:12]}"
                          for name, ids in expected.items()})
        self.assertNotIn("different suites", section)
        count = len(configurations)
        self.assertIn(f"**Checks unrecorded: {count} of {count} rows**", section)


class ReadLog(dict):
    """A case or a passage that notes every key read from it, however it is read."""

    def __init__(self, data: dict, reads: "set[str]") -> None:
        super().__init__(data)
        self.reads = reads

    def __getitem__(self, key: str) -> object:
        self.reads.add(key)
        return super().__getitem__(key)

    def get(self, key: str, default: object = None) -> object:
        self.reads.add(key)
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        self.reads.add(key)
        return super().__contains__(key)

    def __iter__(self):
        self.reads.update(dict.keys(self))
        return super().__iter__()

    def keys(self):
        self.reads.update(dict.keys(self))
        return super().keys()

    def values(self):
        self.reads.update(dict.keys(self))
        return super().values()

    def items(self):
        self.reads.update(dict.keys(self))
        return super().items()


class SilentModel:
    """A model that answers every prompt at once, with an empty graph."""

    def generate(self, _prompt: str) -> dict:
        return {"content": '{"relations": [], "attributes": []}', "seconds": 0.0,
                "completion_tokens": 1, "truncated": False}


def screen_silently(cases: "list[dict]") -> None:
    """Phase A over `cases`, as `screen` runs it, against a model that answers nothing."""
    with contextlib.redirect_stdout(io.StringIO()):
        bench.screen(SilentModel(), bench.read_graph_prompt_template(), cases)


def store_silently(cases: "list[dict]") -> None:
    """Phase B over `cases`, as `endtoend` runs it, against a daemon that stores nothing."""
    with mock.patch.object(bench, "McpClient"), \
            mock.patch.object(bench, "remember_passage", return_value={"seconds": 0.0}), \
            mock.patch.object(bench, "await_edges", return_value={"seconds": 0.0, "profile": {}}), \
            mock.patch.object(bench, "burst", return_value={}):
        bench.endtoend(mock.Mock(), cases)


def fields_read(cases: "list[dict]", score) -> "dict[str, set[str]]":
    """The keys `score` reads of a case, and of a passage, while it scores every case."""
    reads: "dict[str, set[str]]" = {"case": set(), "passage": set()}
    score([ReadLog({**case, "passages": [ReadLog(passage, reads["passage"])
                                         for passage in case["passages"]]}, reads["case"])
           for case in cases])
    return reads


def edited(value: object) -> object:
    """`value` saying something else: a string reworded, a list grown by its own first item."""
    return value + " (edited)" if isinstance(value, str) else value + value[:1]


def with_key_edited(cases: "list[dict]", of, key: str) -> "list[dict]":
    """A copy of `cases` with `key` edited in every holder `of` finds that has it."""
    changed = copy.deepcopy(cases)
    for holder in of(changed):
        if key in holder:
            holder[key] = edited(holder[key])
    return changed


def fields_covered(cases: "list[dict]", phase: str) -> "dict[str, set[str]]":
    """The keys of a case, and of a passage, whose edit moves the phase's suite digest."""
    holders = {"case": lambda suite: suite,
               "passage": lambda suite: [passage for case in suite for passage in case["passages"]]}
    reference = bench.suite_identity(cases, phase)
    covered: "dict[str, set[str]]" = {"case": set(), "passage": set()}
    for level, of in holders.items():
        for key in {key for holder in of(cases) for key in holder}:
            if bench.suite_identity(with_key_edited(cases, of, key), phase) != reference:
                covered[level].add(key)
    return covered


class ScoredSuiteTest(unittest.TestCase):
    """A suite digest covers what its phase's scorer reads of a case, and nothing else (#1949).

    It digested each case as the cases file holds it, so a reworded `note`, or a
    case moved across the train/holdout line, neither of which any scorer reads,
    printed two suites that score alike as two different ones.
    """

    CASE = {"id": "one", "family": "nominal-en", "lang": "en",
            "note": "first wording", "split": "train",
            "passages": [{"text": "Nothing in this sentence relates to anything.",
                          "checks": [{"type": "relations_empty", "severity": "minor",
                                      "label": "a relation was invented"}]}]}

    def recorded_suites(self, case: dict) -> "dict[str, dict]":
        """The suite `screen` and `endtoend` each record over this one case, through `main`."""
        fake = FakeOllama(self, models=[{"name": "qwen3:14b", "digest": DIGEST}])
        with tempfile.TemporaryDirectory() as tmp:
            cases = Path(tmp) / "cases.json"
            cases.write_text(json.dumps({"version": 1, "cases": [case]}), encoding="utf-8")
            return {"screen": screen_via_main(fake, cases, Path(tmp) / "A.json")["suite"],
                    "endtoend": endtoend_via_main(Path(tmp) / "B.json",
                                                  "--cases", str(cases))["suite"]}

    def test_a_reworded_note_or_a_moved_split_leaves_the_recorded_suite_put(self):
        reference = self.recorded_suites(self.CASE)
        reworded = self.recorded_suites({**self.CASE, "note": "second wording",
                                         "split": "holdout"})
        self.assertEqual(reworded, reference)
        # A passage the model reads, and phase B writes, is another suite.
        passage = {**self.CASE["passages"][0], "text": "Nothing here relates to anything else."}
        retexted = self.recorded_suites({**self.CASE, "passages": [passage]})
        for phase, suite in reference.items():
            with self.subTest(phase=phase):
                self.assertNotEqual(retexted[phase], suite)

    def test_the_digest_covers_exactly_what_each_phase_reads_of_a_case(self):
        """What a scorer reads is recorded while it scores the shipped suite; what the
        digest covers, by editing each field of it in turn. Both must be the same keys."""
        cases = bench.load_cases(CASES_PATH)
        for phase, score in (("screen", screen_silently), ("endtoend", store_silently)):
            with self.subTest(phase=phase):
                read = fields_read(cases, score)
                self.assertEqual(fields_covered(cases, phase), read)
                self.assertFalse({"note", "split"} & read["case"])


# ---------------------------------------------------------- published report --


PUBLISHED_RESULTS = PUBLISHED_CAMPAIGN.parent / "2026-08-16-memory-extraction.json"
PUBLISHED_REPORT = PUBLISHED_CAMPAIGN.parent / "2026-08-16-memory-extraction-report.md"


class PublishedReportTest(unittest.TestCase):
    """The committed 2026-08-16 report is what its generator renders, byte for byte.

    It says `Do not edit: re-run the generator instead`. #1955 changed the
    generator the day after it was committed and did not re-run it, so the report
    went on printing `major 0` for two models none of whose replies parsed: a
    count its result files do not support, since nothing was there to grade.
    """

    maxDiff = None

    def test_the_committed_report_is_what_the_generator_renders(self):
        results = json.loads(PUBLISHED_RESULTS.read_text(encoding="utf-8"))
        self.assertEqual(PUBLISHED_REPORT.read_text(encoding="utf-8"),
                         bench.render_report(results))

    def test_the_consolidated_file_holds_what_the_result_files_hold(self):
        """The report renders the consolidated file, so that file must be the campaign's."""
        results = json.loads(PUBLISHED_RESULTS.read_text(encoding="utf-8"))
        self.assertEqual(results["configurations"],
                         bench.merge_configurations(PUBLISHED_CAMPAIGN))
        self.assertEqual(results["end_to_end"],
                         bench.merge_configurations(PUBLISHED_CAMPAIGN, "endtoend"))

    def test_the_published_report_prints_no_environment_and_no_origin_its_runs_lack(self):
        """Host, commit and cases file are each run's, in its `origin` cell (#1949).

        Read when the report was rendered, the 2026-08-16 report printed a
        machine, a commit and a cases file under a local worktree path as the
        campaign's, while none of its result files records any of them.
        """
        results = json.loads(PUBLISHED_RESULTS.read_text(encoding="utf-8"))
        self.assertNotIn("environment", results)
        self.assertNotIn("## Environment", PUBLISHED_REPORT.read_text(encoding="utf-8"))
        rendered = bench.render_report(results)
        rows = {**report_table(rendered, "## Configurations"),
                **report_table(rendered, "## End-to-end")}
        self.assertEqual(len(rows), len(results["configurations"]) + len(results["end_to_end"]))
        self.assertEqual({row.get("origin") for row in rows.values()}, {"unrecorded"})


# ------------------------------------------------------------- local paths --


# A local absolute path as text carries one: `/` or `~/` opening a word, then a
# name. Neither a URL's `//` nor the `/` of `n/a` opens a word.
ABSOLUTE_PATH = re.compile(r"(?<![\w.:/~-])~?/[\w.-]")


def lines_with_a_path(text: str) -> "list[str]":
    """Every line of `text` that holds a local absolute path, stripped."""
    return [line.strip() for line in text.splitlines() if ABSOLUTE_PATH.search(line)]


class LocalPathTest(unittest.TestCase):
    """What a report derives names a model by what identifies it, never by where it lay (#1949).

    The 2026-08-16 MLX result files record their weights by absolute path under
    a home directory, and the campaign file folded from them copied every one.
    The result files keep theirs: they are the evidence.
    """

    def test_a_model_is_named_by_its_directory_and_keeps_its_digest(self):
        weights = "/home/someone/models/mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
        entry = timed_entry("qwen3-30b", 1.0, 3)
        entry["storage"] = {"path": weights, "device": "/dev/disk3s5", "reachable": True}
        entry["residency_before"] = {"models": [{"id": "qwen3-30b", "model_path": weights,
                                                 "digest": DIGEST}]}
        with tempfile.TemporaryDirectory() as tmp:
            write_result(Path(tmp) / "qwen3-30b.json", entry)
            folded = bench.merge_configurations(Path(tmp))["qwen3-30b"]
        name = "Qwen3-30B-A3B-Instruct-2507-4bit"
        self.assertEqual(folded["storage"], {"path": name, "device": "disk3s5", "reachable": True})
        self.assertEqual(folded["residency_before"]["models"],
                         [{"id": "qwen3-30b", "model_path": name, "digest": DIGEST}])

    def test_no_rendered_output_carries_a_local_path(self):
        evidence = [path.name for path in PUBLISHED_CAMPAIGN.rglob("*.json")
                    if lines_with_a_path(path.read_text(encoding="utf-8"))]
        self.assertTrue(evidence, "no result file holds a local path: nothing would be tested")
        with tempfile.TemporaryDirectory() as tmp:
            merged, report = Path(tmp) / "campaign.json", Path(tmp) / "report.md"
            bench.main(["report", "--from-dir", str(PUBLISHED_CAMPAIGN),
                        "--merged-out", str(merged), "--out", str(report)])
            outputs = {"--merged-out": merged.read_text(encoding="utf-8"),
                       "--out": report.read_text(encoding="utf-8")}
        outputs[PUBLISHED_RESULTS.name] = PUBLISHED_RESULTS.read_text(encoding="utf-8")
        outputs[PUBLISHED_REPORT.name] = PUBLISHED_REPORT.read_text(encoding="utf-8")
        for output, text in outputs.items():
            with self.subTest(output=output):
                self.assertEqual(lines_with_a_path(text), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
