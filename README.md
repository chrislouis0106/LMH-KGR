# LMH-KGR : Learning Multi-hop Path for Multi-hop Knowledge Graph Reasoning
- The paper is under WWW review.
# Encoder 
- Pata sampling and path encoding.
- The datasets consist of umls, kinship, WN18RR, NELL-995, and FB15K-237.
- The number of paths sampled and the ratio include 2, 5, 10 and 1,2,3, respectively.
- For umls dataset and valid path 2 and ratio 3,
```
python encoder_conv_2.py --dataset umls  --number_multihop_valid_paths 2 --ratio_valid_invalid 3  --cuda 0
```
# Decoder 
- Train the multi-hop reasoning decoder.
```
./experiment-rs.sh configs/<dataset>-rs.sh --train <gpu-ID>
```
# Acknowledgement
We refer to the code of MultihopKg [here](https://github.com/salesforce/MultiHopKG). Thanks for their contributions.
