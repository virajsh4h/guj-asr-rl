# Gujarati ASR leaderboard

Corpus CER over fleurs, indicvoices, kathbath. Intervals are 95% bootstrap over utterances (2000 resamples, seed 0).

## Three-dataset macro

| # | Model | Macro CER | 95% CI | Macro WER | N |
|---|---|---|---|---|---|

## Per dataset

| Model | Dataset | N | CER | 95% CI | WER | Drift % | Loop % | RTF |
|---|---|---|---|---|---|---|---|---|
| gemma-4-E4B-it | fleurs | 669/1000 | 7.75 | 7.03–8.63 | 24.45 | 3.9 | 0.1 | 0.2454 |
| sravaani-1.0 | fleurs | 321/1000 | 5.50 | 4.79–6.26 | 17.83 | 0.0 | 0.0 | 0.0072 |
| sravaani-1.0 | indicvoices | 2958/2968 | 7.27 | 6.95–7.61 | 20.85 | 17.1 | 1.0 | 0.0081 |
| sravaani-1.0 | kathbath | 2895/2905 | 2.67 | 2.55–2.80 | 12.73 | 0.1 | 0.0 | 0.0082 |
| whisper-small | fleurs | 10/1000 | 86.57 | 82.94–91.55 | 195.65 | 90.0 | 10.0 | 0.1358 |
| whisper-small | indicvoices | 10/2968 | 118.16 | 91.19–208.00 | 200.00 | 80.0 | 40.0 | 0.2016 |
| whisper-small | kathbath | 10/2905 | 84.33 | 82.88–85.74 | 212.38 | 100.0 | 0.0 | 0.1321 |

## Partial coverage (not ranked)

| Model | N | Coverage |
|---|---|---|
| sravaani-1.0 | 6174 | 77.1% |
| gemma-4-E4B-it | 669 | 66.9% |
| whisper-small | 30 | 0.6% |
