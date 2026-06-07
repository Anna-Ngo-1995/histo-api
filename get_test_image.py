from datasets import load_dataset
dataset = load_dataset("1aurent/PatchCamelyon", split="test")

# Chercher une image avec tumeur
for i, sample in enumerate(dataset):
    if sample["label"] == 1:
        sample["image"].save("test_histo_tumeur.png")
        print(f"Image {i} - Vraie etiquette : Tumeur")
        break