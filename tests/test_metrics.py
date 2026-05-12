from unittest import TestCase

from ase import Atoms
from ase.build import molecule
from rdkit import Chem

from src.performance.metrics import MoleculeAnalyzer


class TestMoleculeAnalyzerStatic(TestCase):
    def test_get_compact_smiles_ethanol(self):
        smiles = "CCO"
        compact = MoleculeAnalyzer.get_compact_smiles(smiles)
        # Round-trip through RDKit should give a valid canonical SMILES
        mol = Chem.MolFromSmiles(compact)
        self.assertIsNotNone(mol)

    def test_get_compact_smiles_canonical(self):
        # Both representations of ethanol should yield the same canonical form
        s1 = MoleculeAnalyzer.get_compact_smiles("OCC")
        s2 = MoleculeAnalyzer.get_compact_smiles("CCO")
        self.assertEqual(s1, s2)

    def test_check_charge_neutrality_neutral(self):
        mol = Chem.MolFromSmiles("CCO")
        self.assertTrue(MoleculeAnalyzer.check_charge_neutrality(mol))

    def test_check_charge_neutrality_charged(self):
        mol = Chem.MolFromSmiles("[NH4+]")
        self.assertFalse(MoleculeAnalyzer.check_charge_neutrality(mol))

    def test_check_charge_neutrality_none(self):
        self.assertFalse(MoleculeAnalyzer.check_charge_neutrality(None))


class TestMoleculeAnalyzerGetMol(TestCase):
    def setUp(self):
        self.analyzer = MoleculeAnalyzer(use_huckel=True)

    def test_methane_is_valid(self):
        atoms = molecule("CH4")
        result = self.analyzer.get_mol(atoms)
        self.assertEqual(result["info"], "valid")
        self.assertIsNotNone(result["mol"])

    def test_ethanol_is_valid(self):
        atoms = molecule("CH3CH2OH")
        result = self.analyzer.get_mol(atoms)
        self.assertEqual(result["info"], "valid")

    def test_empty_atoms_fails(self):
        result = self.analyzer.get_mol(Atoms())
        self.assertIn(result["info"], ("failed", "crashed"))
        self.assertIsNone(result["mol"])
