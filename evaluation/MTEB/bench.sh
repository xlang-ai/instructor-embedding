CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/evaluate_model.py --model_name hkunlp/instructor-base --output_dir outputs_for_corpus_prepend_fix --task_name product_help_kb_without_INPUT --batch_size 768 --result_file results --save_suffix base --overlap_percentage 5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/evaluate_model.py --model_name hkunlp/instructor-large --output_dir outputs_for_corpus_prepend_fix --task_name product_help_kb_without_INPUT product_help_kb_with_INPUT product_help_kb_all_QA_pp product_help_kb_zoho_all_QA_pp product_help_kb_all_QA_pp_related product_help_kb_all_QA_pp_relevant product_help_kb_all_DTDR_pp QuoraRetrieval --batch_size 512 --result_file results --save_suffix large


CUDA_VISIBLE_DEVICES=0 python examples/evaluate_model.py --model_name hkunlp/instructor-base --output_dir outputs_testing --task_name product_help_kb_without_INPUT product_help_kb_with_INPUT --batch_size 768 --result_file results --save_suffix base