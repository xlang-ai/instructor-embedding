from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='InstructorEmbedding',
    packages=['InstructorEmbedding'],
    version='1.0.0',
    license='MIT',
    description='Text embedding tool',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Hongjin SU',
    author_email='hjsu@cs.hku.hk',
    url='https://github.com/HKUNLP/instructor-embedding',
    keywords=['sentence', 'embedding', 'text', 'nlp', 'instructor']
)
