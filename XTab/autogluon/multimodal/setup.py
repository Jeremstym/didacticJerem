#!/usr/bin/env python
###########################
# This code block is a HACK (!), but is necessary to avoid code duplication. Do NOT alter these lines.
import importlib.util
import os

from setuptools import setup

filepath = os.path.abspath(os.path.dirname(__file__))
filepath_import = os.path.join(filepath, "..", "core", "src", "autogluon", "core", "_setup_utils.py")
spec = importlib.util.spec_from_file_location("ag_min_dependencies", filepath_import)
ag = importlib.util.module_from_spec(spec)
# Identical to `from autogluon.core import _setup_utils as ag`, but works without `autogluon.core` being installed.
spec.loader.exec_module(ag)
###########################

version = ag.load_version_file()
version = ag.update_version(version)

submodule = "multimodal"
install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "Pillow",
    "tqdm",
    "boto3",
    # "requests",
    "jsonschema",
    # "seqeval",
    "evaluate",
    # "accelerate",
    "timm",
    "torch",
    "torchvision",
    # "torchtext<0.14.0",
    # "fairscale>=0.4.5,<=0.4.6",
    "scikit-image",
    # "smart_open>=5.2.1,<5.3.0",
    "pytorch_lightning",
    "text-unidecode",
    # "torchmetrics>=0.8.0,<0.9.0",
    "transformers",
    "nptyping<1.5",
    "omegaconf",
    # "sentencepiece>=0.1.95,<0.2.0",
    f"autogluon.core[raytune]=={version}",
    f"autogluon.features=={version}",
    f"autogluon.common=={version}",
    "pytorch-metric-learning",
    "nlpaug",
    "nltk",
    # "openmim>0.1.5,<=0.2.1",
    # "pycocotools>=2.0.4,<=2.0.4",
    # "defusedxml>=0.7.1,<=0.7.1",
    # "albumentations>=1.1.0,<=1.2.0",
]

install_requires = ag.get_dependency_version_ranges(install_requires)

extras_require = {
    "tests": [
        "black~=22.0,>=22.3",
        "isort>=5.10",
        "datasets>=2.3.2,<=2.3.2",
        "onnxruntime-gpu>=1.12.1,<=1.12.1;platform_system!='Darwin'",
    ]
}


if __name__ == "__main__":
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup_args["package_data"]["autogluon.multimodal"] = [
        "configs/data/*.yaml",
        "configs/model/*.yaml",
        "configs/optimization/*.yaml",
        "configs/environment/*.yaml",
        "configs/distiller/*.yaml",
        "configs/matcher/*.yaml",
    ]
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
