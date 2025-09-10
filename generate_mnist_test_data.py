import os
import sys

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Augmented Data Generation ---
from imagegeneration.BasicImageTransformations import transform_images

aug_output_dir = 'datamodels/mnist/data/images/generated_inputs_aug'
os.makedirs(aug_output_dir, exist_ok=True)
# Generate all augmentations for MNIST
for aug_type in ['Rotate', 'Translate', 'Brightness', 'Blur']:
    print(f"Generating {aug_type} augmented images for MNIST...")
    transform_images('mnist', aug_type, 10000, aug_output_dir)

# --- Adversarial Data Generation ---
from imagegeneration.AdvAttacks_Art import art_attack_cifar

# Generate adversarial images for Lenet1
adv_lenet1_dir = 'datamodels/mnist/data/images/generated_inputs_adv_lenet1'
os.makedirs(adv_lenet1_dir, exist_ok=True)
for attack_type in ['CW', 'FGSM', 'PGD']:
    print(f"Generating {attack_type} adversarial images for MNIST (Lenet1)...")
    art_attack_cifar(
        data_type='mnist',
        modelname='Lenet1',
        attack_type=attack_type.lower(),  # 'cw', 'fgsm', 'pgd'
        output_dir=adv_lenet1_dir
    )

# Generate adversarial images for Lenet5
adv_lenet5_dir = 'datamodels/mnist/data/images/generated_inputs_adv_lenet5'
os.makedirs(adv_lenet5_dir, exist_ok=True)
for attack_type in ['CW', 'FGSM', 'PGD']:
    print(f"Generating {attack_type} adversarial images for MNIST (Lenet5)...")
    art_attack_cifar(
        data_type='mnist',
        modelname='Lenet5',
        attack_type=attack_type.lower(),  # 'cw', 'fgsm', 'pgd'
        output_dir=adv_lenet5_dir
    )

print("All MNIST augmented and adversarial data generated.")
