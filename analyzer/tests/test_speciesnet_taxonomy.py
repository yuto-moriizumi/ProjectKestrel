"""Unit tests for SpeciesNet taxonomy routing helpers."""

import unittest

from kestrel_analyzer.ml.speciesnet_taxonomy import (
    is_bird_taxon,
    is_ignored_prediction,
    route_prediction,
    wildlife_display_name,
)


class TestSpeciesNetTaxonomy(unittest.TestCase):
    def test_examples_ignore(self):
        self.assertTrue(is_ignored_prediction("f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"))
        self.assertTrue(is_ignored_prediction("e2895ed5-780b-48f6-8a11-9e27cb594511;;;;;;vehicle"))

    def test_examples_wildlife_display(self):
        raw = (
            "04eda76f-c0e7-4e9e-85c3-5b1542db2915;"
            "amphibia;anura;bufonidae;rhinella;marina;cane toad"
        )
        self.assertEqual(wildlife_display_name(raw), "cane toad")

    def test_bird_aves(self):
        raw = "b1352069-a39c-4a84-a949-60044271c5c1;aves;;;;;bird"
        self.assertTrue(is_bird_taxon(raw))
        route, label = route_prediction(raw, wildlife_enabled=True)
        self.assertEqual(route, "bird")
        self.assertEqual(label, "bird")

    def test_wildlife_route_when_enabled(self):
        raw = (
            "04eda76f-c0e7-4e9e-85c3-5b1542db2915;"
            "amphibia;anura;bufonidae;rhinella;marina;cane toad"
        )
        route, label = route_prediction(raw, wildlife_enabled=True)
        self.assertEqual(route, "wildlife")
        self.assertEqual(label, "cane toad")

    def test_wildlife_disabled_skips_non_bird(self):
        raw = (
            "04eda76f-c0e7-4e9e-85c3-5b1542db2915;"
            "amphibia;anura;bufonidae;rhinella;marina;cane toad"
        )
        route, label = route_prediction(raw, wildlife_enabled=False)
        self.assertEqual(route, "ignore")
        self.assertIsNone(label)


if __name__ == "__main__":
    unittest.main()
