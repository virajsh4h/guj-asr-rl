# Gujarati ASR benchmark report (optimized runner)

Generated: 2026-11-06T18:12:38.519780+00:00
Database: `/data/gujarati-asr-sota/results/predictions_optimized.sqlite`

## Evaluation protocol

- FLEURS Gujarati **test** split; IndicVoices Gujarati **valid** split; Kathbath Gujarati **valid** split.
- Corpus WER/CER are computed on normalized reference/hypothesis text.
- Macro scores should only be compared when a model has complete coverage of all selected datasets.
- Whisper uses direct SDPA batched generation with `max_new_tokens=440` to avoid Gujarati BPE truncation.

## Per-dataset leaderboard

| Dataset | Model | N | Coverage | Corpus WER | Corpus CER | Mean capped CER | Median RTF |
|---|---|---|---|---|---|---|---|
| fleurs | gemma-4-E4B-it | 1000 | 100.0% | 24.68% | 7.77% | 7.94% | 0.0437 |
| fleurs | gemma-4-E4B-sft | 1000 | 100.0% | 19.22% | 5.63% | 5.72% | 0.0147 |
| fleurs | gemma-4-E4B-step5 | 1000 | 100.0% | 24.14% | 7.35% | 7.58% | 0.0254 |
| fleurs | indic-transcribe-core | 1000 | 100.0% | 15.87% | 5.01% | 5.27% | 0.0774 |
| fleurs | sravaani-1.0 | 1000 | 100.0% | 17.36% | 5.37% | 5.64% | 0.0062 |
| fleurs | whisper-gujarati-small | 1000 | 100.0% | 27.75% | 10.92% | 9.11% | 0.0110 |
| fleurs | whisper-small | 1000 | 100.0% | 258.29% | 102.43% | 86.91% | 0.0126 |
| indicvoices | gemma-4-E4B-it | 2968 | 100.0% | 35.91% | 16.59% | 24.76% | 0.0512 |
| indicvoices | gemma-4-E4B-sft | 2968 | 100.0% | 23.93% | 9.27% | 15.28% | 0.0191 |
| indicvoices | gemma-4-E4B-step5 | 2968 | 100.0% | 35.25% | 15.90% | 24.16% | 0.0310 |
| indicvoices | indic-transcribe-core | 2968 | 100.0% | 15.88% | 5.36% | 8.96% | 0.0822 |
| indicvoices | sravaani-1.0 | 2968 | 100.0% | 20.95% | 7.31% | 21.86% | 0.0068 |
| indicvoices | whisper-gujarati-small | 2968 | 100.0% | 50.09% | 36.23% | 39.41% | 0.0120 |
| indicvoices | whisper-small | 2968 | 100.0% | 313.38% | 149.52% | 89.34% | 0.0193 |
| kathbath | gemma-4-E4B-it | 2905 | 100.0% | 25.81% | 7.77% | 7.46% | 0.0455 |
| kathbath | gemma-4-E4B-sft | 2905 | 100.0% | 17.52% | 4.41% | 4.37% | 0.0146 |
| kathbath | gemma-4-E4B-step5 | 2905 | 100.0% | 24.35% | 6.79% | 6.79% | 0.0254 |
| kathbath | indic-transcribe-core | 2905 | 100.0% | 12.00% | 2.41% | 2.39% | 0.0822 |
| kathbath | sravaani-1.0 | 2905 | 100.0% | 12.73% | 2.67% | 2.66% | 0.0064 |
| kathbath | whisper-gujarati-small | 2905 | 100.0% | 26.33% | 9.48% | 9.42% | 0.0124 |
| kathbath | whisper-small | 2905 | 100.0% | 268.38% | 106.62% | 86.53% | 0.0205 |

## Complete-coverage macro / pooled scores

| Model | Total N | Macro WER | Macro CER | Pooled WER | Pooled CER |
|---|---|---|---|---|---|
| gemma-4-E4B-it | 6873 | 28.80% | 10.71% | 30.21% | 11.49% |
| gemma-4-E4B-sft | 6873 | 20.22% | 6.44% | 20.82% | 6.74% |
| gemma-4-E4B-step5 | 6873 | 27.92% | 10.01% | 29.31% | 10.76% |
| indic-transcribe-core | 6873 | 14.58% | 4.26% | 14.60% | 4.24% |
| sravaani-1.0 | 6873 | 17.01% | 5.12% | 17.48% | 5.24% |
| whisper-gujarati-small | 6873 | 34.72% | 18.88% | 37.54% | 21.10% |
| whisper-small | 6873 | 280.02% | 119.52% | 286.93% | 123.78% |

## Throughput observations

| Model | Dataset | Samples | Samples/s | Audio s/s | Batch | Batch mode | CPU workers | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|
| sravaani-1.0 | fleurs | 100 | 6.497 | 69.162 | 8 | multi_file_batch_with_fallback | 8 | 2.49 |
| sravaani-1.0 | indicvoices | 100 | 18.654 | 127.326 | 8 | multi_file_batch_with_fallback | 8 | 2.49 |
| sravaani-1.0 | kathbath | 100 | 26.364 | 171.003 | 8 | multi_file_batch_with_fallback | 8 | 2.49 |
| gemma-4-E4B-it | fleurs | 100 | 2.052 | 21.843 | 8 | multimodal_batch | 8 | 15.82 |
| gemma-4-E4B-it | indicvoices | 100 | 2.848 | 19.441 | 8 | multimodal_batch | 8 | 15.82 |
| gemma-4-E4B-it | kathbath | 100 | 3.828 | 24.831 | 8 | multimodal_batch | 8 | 15.82 |
| whisper-gujarati-small | fleurs | 100 | 1.456 | 15.500 | 8 | transformers_pipeline_batch | 8 | 3.05 |
| whisper-gujarati-small | indicvoices | 100 | 1.809 | 12.345 | 8 | transformers_pipeline_batch | 8 | 3.05 |
| whisper-gujarati-small | kathbath | 100 | 1.556 | 10.091 | 8 | transformers_pipeline_batch | 8 | 3.05 |
| whisper-small | fleurs | 100 | 1.707 | 18.175 | 8 | transformers_pipeline_batch | 8 | 3.05 |
| whisper-small | indicvoices | 100 | 1.466 | 10.008 | 8 | transformers_pipeline_batch | 8 | 3.05 |
| whisper-small | kathbath | 100 | 2.420 | 15.699 | 8 | transformers_pipeline_batch | 8 | 3.05 |
| sravaani-1.0 | fleurs | 900 | 13.489 | 141.318 | 8 | multi_file_batch_with_fallback | 8 | 2.53 |
| sravaani-1.0 | indicvoices | 2868 | 23.711 | 155.115 | 8 | multi_file_batch_with_fallback | 8 | 2.56 |
| sravaani-1.0 | kathbath | 2805 | 27.712 | 157.485 | 8 | multi_file_batch_with_fallback | 8 | 2.56 |
| gemma-4-E4B-it | fleurs | 900 | 2.101 | 22.012 | 8 | multimodal_batch | 8 | 15.86 |
| gemma-4-E4B-it | indicvoices | 2868 | 3.060 | 20.020 | 8 | multimodal_batch | 8 | 15.90 |
| gemma-4-E4B-it | kathbath | 1429 | 3.558 | 20.064 | 64 | multimodal_batch | 16 | 15.33 |
| whisper-gujarati-small | fleurs | 900 | 1.932 | 20.246 | 64 | transformers_pipeline_batch | 16 | 21.04 |
| whisper-gujarati-small | fleurs | 1000 | 8.766 | 91.989 | 32 | seq2seq_sdpa_batch | 8 | 2.74 |
| whisper-gujarati-small | indicvoices | 2968 | 13.723 | 89.906 | 32 | seq2seq_sdpa_batch | 8 | 2.74 |
| whisper-gujarati-small | kathbath | 2905 | 14.304 | 81.682 | 32 | seq2seq_sdpa_batch | 8 | 2.74 |
| whisper-small | fleurs | 1000 | 7.975 | 83.683 | 32 | seq2seq_sdpa_batch | 8 | 2.73 |
| whisper-small | indicvoices | 2968 | 8.310 | 54.447 | 32 | seq2seq_sdpa_batch | 8 | 2.73 |
| whisper-small | kathbath | 2905 | 8.956 | 51.146 | 32 | seq2seq_sdpa_batch | 8 | 2.73 |
| gemma-4-E4B-step5 | fleurs | 20 | 1.409 | 16.029 | 8 | multimodal_step5_batch | 8 | 15.38 |
| gemma-4-E4B-step5 | fleurs | 980 | 3.568 | 37.379 | 16 | multimodal_step5_batch | 8 | 16.92 |
| gemma-4-E4B-step5 | indicvoices | 2968 | 5.071 | 33.224 | 16 | multimodal_step5_batch | 8 | 17.00 |
| gemma-4-E4B-step5 | kathbath | 2905 | 6.944 | 39.656 | 16 | multimodal_step5_batch | 8 | 17.00 |
| gemma-4-E4B-sft | fleurs | 1000 | 3.314 | 34.779 | 16 | multimodal_sft_batch | 8 | 17.81 |
| gemma-4-E4B-sft | indicvoices | 2968 | 4.126 | 27.030 | 16 | multimodal_sft_batch | 8 | 17.92 |
| gemma-4-E4B-sft | kathbath | 2905 | 6.268 | 35.794 | 16 | multimodal_sft_batch | 8 | 17.92 |
| gemma-4-E4B-sft | fleurs | 1000 | 2.804 | 29.426 | 16 | multimodal_sft_batch | 8 | 17.82 |
| gemma-4-E4B-sft | indicvoices | 2968 | 4.167 | 27.298 | 16 | multimodal_sft_batch | 8 | 17.93 |
| gemma-4-E4B-sft | kathbath | 2905 | 6.253 | 35.706 | 16 | multimodal_sft_batch | 8 | 17.93 |
| gemma-4-E4B-sft | fleurs | 20 | 2.248 | 25.575 | 24 | multimodal_sft_merged_batch | 8 | 17.33 |
| gemma-4-E4B-sft | fleurs | 1000 | 5.981 | 62.766 | 32 | multimodal_sft_merged_batch | 8 | 19.04 |
| gemma-4-E4B-sft | indicvoices | 2968 | 8.466 | 55.466 | 32 | multimodal_sft_merged_batch | 8 | 19.19 |
| gemma-4-E4B-sft | kathbath | 2905 | 11.966 | 68.335 | 32 | multimodal_sft_merged_batch | 8 | 19.19 |