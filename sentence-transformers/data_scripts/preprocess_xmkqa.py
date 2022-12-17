import sys
import os
import json
from collections import defaultdict

def preprocess_xmkqa(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mkqa = []
    with open(input_path, 'r') as fin:
        for line in fin:
            ex = json.loads(line)
            mkqa.append(ex)
    mkqadict = {ex['example_id']:ex for ex in mkqa}

    langs = ['en', 'ar', 'fi', 'ja', 'ko', 'ru', 'es', 'sv', 'he', 'th', \
            'da', 'de', 'fr', 'it', 'nl', 'pl', 'pt', 'hu', 'vi', 'ms', \
            'km', 'no', 'tr', 'zh_cn', 'zh_hk', 'zh_tw']
    langdata = defaultdict(list)

    for ex in mkqa:
        answers = [] 
        for a in ex['answers']['en']:
            flag = False
            if not (a['type'] == 'unanswerable' or a['type'] == 'binary' or a['type'] == 'long_answer'):
                flag = True
                answers.extend(a.get("aliases", []))
                answers.append(a.get("text"))
        if flag:
            for lang in langs:
                langex = {
                    'id': ex['example_id'],
                    'lang': lang,
                    'question': ex['queries'][lang], #question in specific languages
                    'answers': answers #english answers
                }
                langdata[lang].append(langex)


    for lang, data in langdata.items():
        with open(os.path.join(output_dir, f'{lang}.jsonl'), 'w') as fout:
            for ex in data:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write('\n')

if __name__ == '__main__':
    preprocess_xmkqa(sys.argv[1], sys.argv[2])
