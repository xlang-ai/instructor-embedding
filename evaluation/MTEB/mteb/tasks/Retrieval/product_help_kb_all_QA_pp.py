from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask


class CustomRetrieval(AbsTaskRetrieval, BeIRTask):
    @property
    def description(self):
        return {
            "name": "product_help_kb_all_QA_pp",
            "beir_name": "product_help_kb_all_QA_pp",
            "description": (
                "custom data for benchmarking sentence transformer model"
            ),
            "reference": "",
            "type": "Retrieval",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "is_custom_dataset" : True
        }
