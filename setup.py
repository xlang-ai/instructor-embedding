from setuptools import setup, find_packages

setup(
    name="instructor",
    version="1.0.0",
    author="Hongjin SU",
    author_email="hjsu@cs.hku.hk",
    description="text embeddings",
    long_description="instructor class",
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    packages=find_packages(),
    python_requires=">=3.6.0",
    install_requires=[
        'transformers>=4.6.0,<5.0.0',
        'tqdm',
        'torch>=1.6.0',
        'beir==1.0.0',
        'torchvision',
        'numpy',
        'scikit-learn',
        'scipy',
        'nltk',
        'sentencepiece',
        'huggingface-hub>=0.4.0',
        'sentence_transformers'
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="INSTRUCTOR code base"
)
