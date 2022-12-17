import sys
import os
import csv
import json

def convert2beir(data_path, output_path):

    splits = ['test', 'dev', 'train']
    queries_path = os.path.join(output_path, "queries.jsonl")
    corpus_path = os.path.join(output_path, "corpus.jsonl")
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    queries = []
    with open(queries_path, "w", encoding="utf-8") as fout:
        with open(os.path.join(data_path, f"topic.tsv"), "r", encoding="utf-8") as fin:
            reader = csv.reader(fin, delimiter="\t")
            for x in reader:
                qdict = {
                    "_id": x[0],
                    "text": x[1]
                }
                json.dump(qdict, fout, ensure_ascii=False)
                fout.write('\n')

    with open(os.path.join(data_path, "collection", "docs.jsonl"), "r") as fin:
        with open(corpus_path, "w", encoding="utf-8") as fout:
            for line in fin:
                x = json.loads(line)
                x["_id"] = x["id"]
                x["text"] = x["contents"]
                x["title"] = ""
                del x["id"]
                del x["contents"]
                json.dump(x, fout, ensure_ascii=False)
                fout.write('\n')


    for split in splits:

        qrels_path = os.path.join(output_path, "qrels", f"{split}.tsv")
        os.makedirs(os.path.dirname(qrels_path), exist_ok=True)

        with open(os.path.join(data_path, f"qrels.{split}.txt"), "r", encoding="utf-8") as fin:
            with open(qrels_path, "w", encoding="utf-8") as fout:
                writer = csv.writer(fout, delimiter='\t')
                writer.writerow(["query-id", "corpus-id", "score"])
                for line in fin:
                    line = line.strip()
                    el = line.split()
                    qid = el[0]
                    i = el[2]
                    s = el[3]
                    writer.writerow([qid, i, s])


if __name__ == '__main__':
    convert2beir(sys.argv[1], sys.argv[2])