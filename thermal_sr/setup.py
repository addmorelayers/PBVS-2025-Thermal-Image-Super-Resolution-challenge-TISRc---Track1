from setuptools import setup, find_packages

setup(
    name="thermal_sr",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pytorch-lightning",
        "numpy",
        "matplotlib",
        "torchvision",
        "tqdm",
        "torchmetrics",
        "PyYAML",
        "pillow",  # for image processing
    ],
    python_requires='>=3.7',
    author="X2",
    description="Thermal Image Super Resolution"
)
