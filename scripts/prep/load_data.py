from src.data.reference_dataloader import ReferenceDataLoader

if __name__ == "__main__":
    loader = ReferenceDataLoader(data_dir="data")
    ref_data = loader._load_data(mol_dataset="qm7")
    energies = loader.get_benchmark_energies(ref_data)
    print(energies)
