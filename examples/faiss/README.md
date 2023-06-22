# Use in conjunction with FAISS

[Faiss](https://github.com/facebookresearch/faiss) is a library for efficient similarity search and clustering of dense vectors, made by Meta AI research.

The following extended example parses arXiv metadata out of a `json` file `arxiv-metadata-10000.json`. It has 10000 lines, each of which is a `json` object containing metadata for a paper on arXiv. For example, the first line is (after formatting for easy reading):

```json
{
  "id": "0704.0001",
  "submitter": "Pavel Nadolsky",
  "authors": "C. Bal\\'azs, E. L. Berger, P. M. Nadolsky, C.-P. Yuan",
  "title": "Calculation of prompt diphoton production cross sections at Tevatron and\n  LHC energies",
  "comments": "37 pages, 15 figures; published version",
  "journal-ref": "Phys.Rev.D76:013009,2007",
  "doi": "10.1103/PhysRevD.76.013009",
  "report-no": "ANL-HEP-PR-07-12",
  "categories": "hep-ph",
  "license": null,
  "abstract": "  A fully differential calculation in perturbative quantum chromodynamics is\npresented for the production of massive photon pairs at hadron colliders. All\nnext-to-leading order perturbative contributions from quark-antiquark,\ngluon-(anti)quark, and gluon-gluon subprocesses are included, as well as\nall-orders resummation of initial-state gluon radiation valid at\nnext-to-next-to-leading logarithmic accuracy. The region of phase space is\nspecified ...",
  "versions": [ { "version": "v1", "created": "Mon, 2 Apr 2007 19:18:42 GMT" }, { "version": "v2", "created": "Tue, 24 Jul 2007 20:10:27 GMT" } ],
  "update_date": "2008-11-26",
  "authors_parsed": [ [ "Balázs", "C.", "" ], [ "Berger", "E. L.", "" ], [ "Nadolsky", "P. M.", "" ], [ "Yuan", "C. -P.", "" ] ]}
```

The entire Arxiv has 1.7 million papers, found in [arXiv Dataset | Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv).



```python
from InstructorEmbedding import INSTRUCTOR
model_ins = INSTRUCTOR('hkunlp/instructor-base')

sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
instruction = "Represent the Science title:"
embeddings = model_ins.encode([[instruction,sentence]])
print(embeddings.shape)

# --------------------------------------------------------------------------------

import json
import numpy as np
import faiss
import torch

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)

# FAISS index setup
dimension = 768  # Instructor-XL output dimension
index_ins = faiss.IndexFlatL2(dimension)

# Extract and vectorize data
db_filename = 'arxiv-metadata-10000.json'
num_lines = 10000
batch_size = 4

# Load all papers from JSON
with open(db_filename, 'r') as f:
    papers = [json.loads(line) for line in f]

# Extract the papers' titles and abstracts
texts = [f"{paper['title']}: {paper['abstract']}" for paper in papers]

# Preparation for encoding
instructions = ["Represent the science titles and abstracts: "] * len(texts)

# Prepare the inputs
inputs = [[instr, txt] for instr, txt in zip(instructions, texts)]

# Create vectors using Instructor
vectors = model_ins.encode(
    sentences=inputs[:num_lines],
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_numpy=True,
    device=str(device)
)

# Add the vectors to the FAISS index
index_ins.add(np.array(vectors).astype('float32'))

print(f"Added {num_lines} papers to the FAISS index.")

# --------------------------------------------------------------------------------

def search_ins(query, k=5):
    vector = model_ins.encode(["Represent the query to a science database: ", query])
    _, indices = index_ins.search(np.array(vector[1]).reshape(1, -1).astype('float32'), k)
    return indices[0]

for query in queries:
    print(f"Question: {query}\n")
    line_numbers = search_ins(query, k=2)
    print_paper_details(line_numbers)
    print('-'*80)
```
