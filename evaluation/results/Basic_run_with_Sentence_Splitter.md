# Basic_run_with_Sentence_Splitter

## Description
Baseline run with Sentence Splitter with 400 Chunks

## Configuration
- **Embedding Model**: intfloat/e5-small-v2
- **Generation Model**: gemini-2.5-flash

## Results Summary

| TOP_K | Hit Rate | MRR    |
|-------|----------|--------|
| 2     | 0.7254   | 0.631746 |
| 4     | 0.8429   | 0.668122 |
| 6     | 0.8889   | 0.676799 |
| 8     | 0.9270   | 0.681844 |
| 10    | 0.9460   | 0.683890 |
| 12    | 0.9508   | 0.684311 |
| 20    | 0.9778   | 0.685986 |
