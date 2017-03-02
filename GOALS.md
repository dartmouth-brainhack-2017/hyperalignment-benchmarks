# Goals for BrainHack

Make a script that can take data as input and output a benchmark measures.

# Input
- Input data should be a list of datasets
- Each individual dataset should have same number of samples (or observations)

# Output
- Results of tests

# Benchmark Tests
1. Time segment classification
2. Inter-subject(dataset) correlation of each feature.

# Steps
1. Load the datasets.
2. Normalization step such as zscoring or percent signal change.
3. Split the data into training and testing parts.
4. Run Hyperalignment on training half and get transformation mappers.
5. Apply mappers to test half.
6. Run benchmark tests.