model-index:
- name: final_xl_results
  results:
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_counterfactual
      name: MTEB AmazonCounterfactualClassification (en)
      config: en
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 85.08955223880596
    - type: ap
      value: 52.66066378722476
    - type: f1
      value: 79.63340218960269
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_polarity
      name: MTEB AmazonPolarityClassification
      config: default
      split: test
      revision: e2d317d38cd51312af73b3d32a06d1a08b442046
    metrics:
    - type: accuracy
      value: 86.542
    - type: ap
      value: 81.92695193008987
    - type: f1
      value: 86.51466132573681
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (en)
      config: en
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 42.964
    - type: f1
      value: 41.43146249774862
  - task:
      type: Retrieval
    dataset:
      type: arguana
      name: MTEB ArguAna
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 29.872
    - type: map_at_10
      value: 46.342
    - type: map_at_100
      value: 47.152
    - type: map_at_1000
      value: 47.154
    - type: map_at_3
      value: 41.216
    - type: map_at_5
      value: 44.035999999999994
    - type: mrr_at_1
      value: 30.939
    - type: mrr_at_10
      value: 46.756
    - type: mrr_at_100
      value: 47.573
    - type: mrr_at_1000
      value: 47.575
    - type: mrr_at_3
      value: 41.548
    - type: mrr_at_5
      value: 44.425
    - type: ndcg_at_1
      value: 29.872
    - type: ndcg_at_10
      value: 55.65
    - type: ndcg_at_100
      value: 58.88099999999999
    - type: ndcg_at_1000
      value: 58.951
    - type: ndcg_at_3
      value: 45.0
    - type: ndcg_at_5
      value: 50.09
    - type: precision_at_1
      value: 29.872
    - type: precision_at_10
      value: 8.549
    - type: precision_at_100
      value: 0.991
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 18.658
    - type: precision_at_5
      value: 13.669999999999998
    - type: recall_at_1
      value: 29.872
    - type: recall_at_10
      value: 85.491
    - type: recall_at_100
      value: 99.075
    - type: recall_at_1000
      value: 99.644
    - type: recall_at_3
      value: 55.974000000000004
    - type: recall_at_5
      value: 68.35
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-p2p
      name: MTEB ArxivClusteringP2P
      config: default
      split: test
      revision: a122ad7f3f0291bf49cc6f4d32aa80929df69d5d
    metrics:
    - type: v_measure
      value: 42.452729850641276
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-s2s
      name: MTEB ArxivClusteringS2S
      config: default
      split: test
      revision: f910caf1a6075f7329cdf8c1a6135696f37dbd53
    metrics:
    - type: v_measure
      value: 32.21141846480423
  - task:
      type: Reranking
    dataset:
      type: mteb/askubuntudupquestions-reranking
      name: MTEB AskUbuntuDupQuestions
      config: default
      split: test
      revision: 2000358ca161889fa9c082cb41daa8dcfb161a54
    metrics:
    - type: map
      value: 65.34710928952622
    - type: mrr
      value: 77.61124301983028
  - task:
      type: STS
    dataset:
      type: mteb/biosses-sts
      name: MTEB BIOSSES
      config: default
      split: test
      revision: d3fb88f8f02e40887cd149695127462bbcf29b4a
    metrics:
    - type: cos_sim_spearman
      value: 84.15312230525639
  - task:
      type: Classification
    dataset:
      type: mteb/banking77
      name: MTEB Banking77Classification
      config: default
      split: test
      revision: 0fd18e25b25c072e09e0d92ab615fda904d66300
    metrics:
    - type: accuracy
      value: 82.66233766233766
    - type: f1
      value: 82.04175284777669
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-p2p
      name: MTEB BiorxivClusteringP2P
      config: default
      split: test
      revision: 65b79d1d13f80053f67aca9498d9402c2d9f1f40
    metrics:
    - type: v_measure
      value: 37.36697339826455
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-s2s
      name: MTEB BiorxivClusteringS2S
      config: default
      split: test
      revision: 258694dd0231531bc1fd9de6ceb52a0853c6d908
    metrics:
    - type: v_measure
      value: 30.551241447593092
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackAndroidRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 36.797000000000004
    - type: map_at_10
      value: 48.46
    - type: map_at_100
      value: 49.968
    - type: map_at_1000
      value: 50.080000000000005
    - type: map_at_3
      value: 44.71
    - type: map_at_5
      value: 46.592
    - type: mrr_at_1
      value: 45.494
    - type: mrr_at_10
      value: 54.747
    - type: mrr_at_100
      value: 55.43599999999999
    - type: mrr_at_1000
      value: 55.464999999999996
    - type: mrr_at_3
      value: 52.361000000000004
    - type: mrr_at_5
      value: 53.727000000000004
    - type: ndcg_at_1
      value: 45.494
    - type: ndcg_at_10
      value: 54.989
    - type: ndcg_at_100
      value: 60.096000000000004
    - type: ndcg_at_1000
      value: 61.58
    - type: ndcg_at_3
      value: 49.977
    - type: ndcg_at_5
      value: 51.964999999999996
    - type: precision_at_1
      value: 45.494
    - type: precision_at_10
      value: 10.558
    - type: precision_at_100
      value: 1.6049999999999998
    - type: precision_at_1000
      value: 0.203
    - type: precision_at_3
      value: 23.796
    - type: precision_at_5
      value: 16.881
    - type: recall_at_1
      value: 36.797000000000004
    - type: recall_at_10
      value: 66.83
    - type: recall_at_100
      value: 88.34100000000001
    - type: recall_at_1000
      value: 97.202
    - type: recall_at_3
      value: 51.961999999999996
    - type: recall_at_5
      value: 57.940000000000005
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackEnglishRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 32.597
    - type: map_at_10
      value: 43.424
    - type: map_at_100
      value: 44.78
    - type: map_at_1000
      value: 44.913
    - type: map_at_3
      value: 40.315
    - type: map_at_5
      value: 41.987
    - type: mrr_at_1
      value: 40.382
    - type: mrr_at_10
      value: 49.219
    - type: mrr_at_100
      value: 49.895
    - type: mrr_at_1000
      value: 49.936
    - type: mrr_at_3
      value: 46.996
    - type: mrr_at_5
      value: 48.231
    - type: ndcg_at_1
      value: 40.382
    - type: ndcg_at_10
      value: 49.318
    - type: ndcg_at_100
      value: 53.839999999999996
    - type: ndcg_at_1000
      value: 55.82899999999999
    - type: ndcg_at_3
      value: 44.914
    - type: ndcg_at_5
      value: 46.798
    - type: precision_at_1
      value: 40.382
    - type: precision_at_10
      value: 9.274000000000001
    - type: precision_at_100
      value: 1.497
    - type: precision_at_1000
      value: 0.198
    - type: precision_at_3
      value: 21.592
    - type: precision_at_5
      value: 15.159
    - type: recall_at_1
      value: 32.597
    - type: recall_at_10
      value: 59.882000000000005
    - type: recall_at_100
      value: 78.446
    - type: recall_at_1000
      value: 90.88000000000001
    - type: recall_at_3
      value: 46.9
    - type: recall_at_5
      value: 52.222
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackGamingRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 43.8
    - type: map_at_10
      value: 57.293000000000006
    - type: map_at_100
      value: 58.321
    - type: map_at_1000
      value: 58.361
    - type: map_at_3
      value: 53.839999999999996
    - type: map_at_5
      value: 55.838
    - type: mrr_at_1
      value: 49.592000000000006
    - type: mrr_at_10
      value: 60.643
    - type: mrr_at_100
      value: 61.23499999999999
    - type: mrr_at_1000
      value: 61.251999999999995
    - type: mrr_at_3
      value: 58.265
    - type: mrr_at_5
      value: 59.717
    - type: ndcg_at_1
      value: 49.592000000000006
    - type: ndcg_at_10
      value: 63.364
    - type: ndcg_at_100
      value: 67.167
    - type: ndcg_at_1000
      value: 67.867
    - type: ndcg_at_3
      value: 57.912
    - type: ndcg_at_5
      value: 60.697
    - type: precision_at_1
      value: 49.592000000000006
    - type: precision_at_10
      value: 10.088
    - type: precision_at_100
      value: 1.2930000000000001
    - type: precision_at_1000
      value: 0.13899999999999998
    - type: precision_at_3
      value: 25.789
    - type: precision_at_5
      value: 17.541999999999998
    - type: recall_at_1
      value: 43.8
    - type: recall_at_10
      value: 77.635
    - type: recall_at_100
      value: 93.748
    - type: recall_at_1000
      value: 98.468
    - type: recall_at_3
      value: 63.223
    - type: recall_at_5
      value: 70.122
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackGisRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 27.721
    - type: map_at_10
      value: 35.626999999999995
    - type: map_at_100
      value: 36.719
    - type: map_at_1000
      value: 36.8
    - type: map_at_3
      value: 32.781
    - type: map_at_5
      value: 34.333999999999996
    - type: mrr_at_1
      value: 29.604999999999997
    - type: mrr_at_10
      value: 37.564
    - type: mrr_at_100
      value: 38.505
    - type: mrr_at_1000
      value: 38.565
    - type: mrr_at_3
      value: 34.727000000000004
    - type: mrr_at_5
      value: 36.207
    - type: ndcg_at_1
      value: 29.604999999999997
    - type: ndcg_at_10
      value: 40.575
    - type: ndcg_at_100
      value: 45.613
    - type: ndcg_at_1000
      value: 47.676
    - type: ndcg_at_3
      value: 34.811
    - type: ndcg_at_5
      value: 37.491
    - type: precision_at_1
      value: 29.604999999999997
    - type: precision_at_10
      value: 6.1690000000000005
    - type: precision_at_100
      value: 0.906
    - type: precision_at_1000
      value: 0.11199999999999999
    - type: precision_at_3
      value: 14.237
    - type: precision_at_5
      value: 10.056
    - type: recall_at_1
      value: 27.721
    - type: recall_at_10
      value: 54.041
    - type: recall_at_100
      value: 76.62299999999999
    - type: recall_at_1000
      value: 92.134
    - type: recall_at_3
      value: 38.582
    - type: recall_at_5
      value: 44.989000000000004
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackMathematicaRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 16.553
    - type: map_at_10
      value: 25.384
    - type: map_at_100
      value: 26.655
    - type: map_at_1000
      value: 26.778000000000002
    - type: map_at_3
      value: 22.733
    - type: map_at_5
      value: 24.119
    - type: mrr_at_1
      value: 20.149
    - type: mrr_at_10
      value: 29.705
    - type: mrr_at_100
      value: 30.672
    - type: mrr_at_1000
      value: 30.737
    - type: mrr_at_3
      value: 27.032
    - type: mrr_at_5
      value: 28.369
    - type: ndcg_at_1
      value: 20.149
    - type: ndcg_at_10
      value: 30.843999999999998
    - type: ndcg_at_100
      value: 36.716
    - type: ndcg_at_1000
      value: 39.495000000000005
    - type: ndcg_at_3
      value: 25.918999999999997
    - type: ndcg_at_5
      value: 27.992
    - type: precision_at_1
      value: 20.149
    - type: precision_at_10
      value: 5.858
    - type: precision_at_100
      value: 1.009
    - type: precision_at_1000
      value: 0.13799999999999998
    - type: precision_at_3
      value: 12.645000000000001
    - type: precision_at_5
      value: 9.179
    - type: recall_at_1
      value: 16.553
    - type: recall_at_10
      value: 43.136
    - type: recall_at_100
      value: 68.562
    - type: recall_at_1000
      value: 88.208
    - type: recall_at_3
      value: 29.493000000000002
    - type: recall_at_5
      value: 34.751
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackPhysicsRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 28.000999999999998
    - type: map_at_10
      value: 39.004
    - type: map_at_100
      value: 40.461999999999996
    - type: map_at_1000
      value: 40.566
    - type: map_at_3
      value: 35.805
    - type: map_at_5
      value: 37.672
    - type: mrr_at_1
      value: 33.782000000000004
    - type: mrr_at_10
      value: 44.702
    - type: mrr_at_100
      value: 45.528
    - type: mrr_at_1000
      value: 45.576
    - type: mrr_at_3
      value: 42.14
    - type: mrr_at_5
      value: 43.651
    - type: ndcg_at_1
      value: 33.782000000000004
    - type: ndcg_at_10
      value: 45.275999999999996
    - type: ndcg_at_100
      value: 50.888
    - type: ndcg_at_1000
      value: 52.879
    - type: ndcg_at_3
      value: 40.191
    - type: ndcg_at_5
      value: 42.731
    - type: precision_at_1
      value: 33.782000000000004
    - type: precision_at_10
      value: 8.200000000000001
    - type: precision_at_100
      value: 1.287
    - type: precision_at_1000
      value: 0.16199999999999998
    - type: precision_at_3
      value: 19.185
    - type: precision_at_5
      value: 13.667000000000002
    - type: recall_at_1
      value: 28.000999999999998
    - type: recall_at_10
      value: 58.131
    - type: recall_at_100
      value: 80.869
    - type: recall_at_1000
      value: 93.931
    - type: recall_at_3
      value: 44.161
    - type: recall_at_5
      value: 50.592000000000006
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackProgrammersRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 28.047
    - type: map_at_10
      value: 38.596000000000004
    - type: map_at_100
      value: 40.116
    - type: map_at_1000
      value: 40.232
    - type: map_at_3
      value: 35.205
    - type: map_at_5
      value: 37.076
    - type: mrr_at_1
      value: 34.932
    - type: mrr_at_10
      value: 44.496
    - type: mrr_at_100
      value: 45.47
    - type: mrr_at_1000
      value: 45.519999999999996
    - type: mrr_at_3
      value: 41.743
    - type: mrr_at_5
      value: 43.352000000000004
    - type: ndcg_at_1
      value: 34.932
    - type: ndcg_at_10
      value: 44.901
    - type: ndcg_at_100
      value: 50.788999999999994
    - type: ndcg_at_1000
      value: 52.867
    - type: ndcg_at_3
      value: 39.449
    - type: ndcg_at_5
      value: 41.929
    - type: precision_at_1
      value: 34.932
    - type: precision_at_10
      value: 8.311
    - type: precision_at_100
      value: 1.3050000000000002
    - type: precision_at_1000
      value: 0.166
    - type: precision_at_3
      value: 18.836
    - type: precision_at_5
      value: 13.447000000000001
    - type: recall_at_1
      value: 28.047
    - type: recall_at_10
      value: 57.717
    - type: recall_at_100
      value: 82.182
    - type: recall_at_1000
      value: 95.82000000000001
    - type: recall_at_3
      value: 42.448
    - type: recall_at_5
      value: 49.071
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 27.861250000000005
    - type: map_at_10
      value: 37.529583333333335
    - type: map_at_100
      value: 38.7915
    - type: map_at_1000
      value: 38.90558333333335
    - type: map_at_3
      value: 34.57333333333333
    - type: map_at_5
      value: 36.187166666666656
    - type: mrr_at_1
      value: 32.88291666666666
    - type: mrr_at_10
      value: 41.79750000000001
    - type: mrr_at_100
      value: 42.63183333333333
    - type: mrr_at_1000
      value: 42.68483333333333
    - type: mrr_at_3
      value: 39.313750000000006
    - type: mrr_at_5
      value: 40.70483333333333
    - type: ndcg_at_1
      value: 32.88291666666666
    - type: ndcg_at_10
      value: 43.09408333333333
    - type: ndcg_at_100
      value: 48.22158333333333
    - type: ndcg_at_1000
      value: 50.358000000000004
    - type: ndcg_at_3
      value: 38.129583333333336
    - type: ndcg_at_5
      value: 40.39266666666666
    - type: precision_at_1
      value: 32.88291666666666
    - type: precision_at_10
      value: 7.5584999999999996
    - type: precision_at_100
      value: 1.1903333333333332
    - type: precision_at_1000
      value: 0.15658333333333332
    - type: precision_at_3
      value: 17.495916666666666
    - type: precision_at_5
      value: 12.373833333333332
    - type: recall_at_1
      value: 27.861250000000005
    - type: recall_at_10
      value: 55.215916666666665
    - type: recall_at_100
      value: 77.392
    - type: recall_at_1000
      value: 92.04908333333334
    - type: recall_at_3
      value: 41.37475
    - type: recall_at_5
      value: 47.22908333333333
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackStatsRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 25.064999999999998
    - type: map_at_10
      value: 31.635999999999996
    - type: map_at_100
      value: 32.596000000000004
    - type: map_at_1000
      value: 32.695
    - type: map_at_3
      value: 29.612
    - type: map_at_5
      value: 30.768
    - type: mrr_at_1
      value: 28.528
    - type: mrr_at_10
      value: 34.717
    - type: mrr_at_100
      value: 35.558
    - type: mrr_at_1000
      value: 35.626000000000005
    - type: mrr_at_3
      value: 32.745000000000005
    - type: mrr_at_5
      value: 33.819
    - type: ndcg_at_1
      value: 28.528
    - type: ndcg_at_10
      value: 35.647
    - type: ndcg_at_100
      value: 40.207
    - type: ndcg_at_1000
      value: 42.695
    - type: ndcg_at_3
      value: 31.878
    - type: ndcg_at_5
      value: 33.634
    - type: precision_at_1
      value: 28.528
    - type: precision_at_10
      value: 5.46
    - type: precision_at_100
      value: 0.84
    - type: precision_at_1000
      value: 0.11399999999999999
    - type: precision_at_3
      value: 13.547999999999998
    - type: precision_at_5
      value: 9.325
    - type: recall_at_1
      value: 25.064999999999998
    - type: recall_at_10
      value: 45.096000000000004
    - type: recall_at_100
      value: 65.658
    - type: recall_at_1000
      value: 84.128
    - type: recall_at_3
      value: 34.337
    - type: recall_at_5
      value: 38.849000000000004
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackTexRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 17.276
    - type: map_at_10
      value: 24.535
    - type: map_at_100
      value: 25.655
    - type: map_at_1000
      value: 25.782
    - type: map_at_3
      value: 22.228
    - type: map_at_5
      value: 23.612
    - type: mrr_at_1
      value: 21.266
    - type: mrr_at_10
      value: 28.474
    - type: mrr_at_100
      value: 29.398000000000003
    - type: mrr_at_1000
      value: 29.482000000000003
    - type: mrr_at_3
      value: 26.245
    - type: mrr_at_5
      value: 27.624
    - type: ndcg_at_1
      value: 21.266
    - type: ndcg_at_10
      value: 29.087000000000003
    - type: ndcg_at_100
      value: 34.374
    - type: ndcg_at_1000
      value: 37.433
    - type: ndcg_at_3
      value: 25.040000000000003
    - type: ndcg_at_5
      value: 27.116
    - type: precision_at_1
      value: 21.266
    - type: precision_at_10
      value: 5.258
    - type: precision_at_100
      value: 0.9299999999999999
    - type: precision_at_1000
      value: 0.13699999999999998
    - type: precision_at_3
      value: 11.849
    - type: precision_at_5
      value: 8.699
    - type: recall_at_1
      value: 17.276
    - type: recall_at_10
      value: 38.928000000000004
    - type: recall_at_100
      value: 62.529
    - type: recall_at_1000
      value: 84.44800000000001
    - type: recall_at_3
      value: 27.554000000000002
    - type: recall_at_5
      value: 32.915
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackUnixRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 27.297
    - type: map_at_10
      value: 36.957
    - type: map_at_100
      value: 38.252
    - type: map_at_1000
      value: 38.356
    - type: map_at_3
      value: 34.121
    - type: map_at_5
      value: 35.782000000000004
    - type: mrr_at_1
      value: 32.275999999999996
    - type: mrr_at_10
      value: 41.198
    - type: mrr_at_100
      value: 42.131
    - type: mrr_at_1000
      value: 42.186
    - type: mrr_at_3
      value: 38.557
    - type: mrr_at_5
      value: 40.12
    - type: ndcg_at_1
      value: 32.275999999999996
    - type: ndcg_at_10
      value: 42.516
    - type: ndcg_at_100
      value: 48.15
    - type: ndcg_at_1000
      value: 50.344
    - type: ndcg_at_3
      value: 37.423
    - type: ndcg_at_5
      value: 39.919
    - type: precision_at_1
      value: 32.275999999999996
    - type: precision_at_10
      value: 7.155
    - type: precision_at_100
      value: 1.123
    - type: precision_at_1000
      value: 0.14200000000000002
    - type: precision_at_3
      value: 17.163999999999998
    - type: precision_at_5
      value: 12.127
    - type: recall_at_1
      value: 27.297
    - type: recall_at_10
      value: 55.238
    - type: recall_at_100
      value: 79.2
    - type: recall_at_1000
      value: 94.258
    - type: recall_at_3
      value: 41.327000000000005
    - type: recall_at_5
      value: 47.588
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackWebmastersRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 29.142000000000003
    - type: map_at_10
      value: 38.769
    - type: map_at_100
      value: 40.292
    - type: map_at_1000
      value: 40.510000000000005
    - type: map_at_3
      value: 35.39
    - type: map_at_5
      value: 37.009
    - type: mrr_at_1
      value: 34.19
    - type: mrr_at_10
      value: 43.418
    - type: mrr_at_100
      value: 44.132
    - type: mrr_at_1000
      value: 44.175
    - type: mrr_at_3
      value: 40.547
    - type: mrr_at_5
      value: 42.088
    - type: ndcg_at_1
      value: 34.19
    - type: ndcg_at_10
      value: 45.14
    - type: ndcg_at_100
      value: 50.364
    - type: ndcg_at_1000
      value: 52.481
    - type: ndcg_at_3
      value: 39.466
    - type: ndcg_at_5
      value: 41.772
    - type: precision_at_1
      value: 34.19
    - type: precision_at_10
      value: 8.715
    - type: precision_at_100
      value: 1.6150000000000002
    - type: precision_at_1000
      value: 0.247
    - type: precision_at_3
      value: 18.248
    - type: precision_at_5
      value: 13.161999999999999
    - type: recall_at_1
      value: 29.142000000000003
    - type: recall_at_10
      value: 57.577999999999996
    - type: recall_at_100
      value: 81.428
    - type: recall_at_1000
      value: 94.017
    - type: recall_at_3
      value: 41.402
    - type: recall_at_5
      value: 47.695
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackWordpressRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 22.039
    - type: map_at_10
      value: 30.669999999999998
    - type: map_at_100
      value: 31.682
    - type: map_at_1000
      value: 31.794
    - type: map_at_3
      value: 28.139999999999997
    - type: map_at_5
      value: 29.457
    - type: mrr_at_1
      value: 24.399
    - type: mrr_at_10
      value: 32.687
    - type: mrr_at_100
      value: 33.622
    - type: mrr_at_1000
      value: 33.698
    - type: mrr_at_3
      value: 30.407
    - type: mrr_at_5
      value: 31.552999999999997
    - type: ndcg_at_1
      value: 24.399
    - type: ndcg_at_10
      value: 35.472
    - type: ndcg_at_100
      value: 40.455000000000005
    - type: ndcg_at_1000
      value: 43.15
    - type: ndcg_at_3
      value: 30.575000000000003
    - type: ndcg_at_5
      value: 32.668
    - type: precision_at_1
      value: 24.399
    - type: precision_at_10
      value: 5.656
    - type: precision_at_100
      value: 0.874
    - type: precision_at_1000
      value: 0.121
    - type: precision_at_3
      value: 13.062000000000001
    - type: precision_at_5
      value: 9.242
    - type: recall_at_1
      value: 22.039
    - type: recall_at_10
      value: 48.379
    - type: recall_at_100
      value: 71.11800000000001
    - type: recall_at_1000
      value: 91.095
    - type: recall_at_3
      value: 35.108
    - type: recall_at_5
      value: 40.015
  - task:
      type: Retrieval
    dataset:
      type: climate-fever
      name: MTEB ClimateFEVER
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 10.144
    - type: map_at_10
      value: 18.238
    - type: map_at_100
      value: 20.143
    - type: map_at_1000
      value: 20.346
    - type: map_at_3
      value: 14.809
    - type: map_at_5
      value: 16.567999999999998
    - type: mrr_at_1
      value: 22.671
    - type: mrr_at_10
      value: 34.906
    - type: mrr_at_100
      value: 35.858000000000004
    - type: mrr_at_1000
      value: 35.898
    - type: mrr_at_3
      value: 31.238
    - type: mrr_at_5
      value: 33.342
    - type: ndcg_at_1
      value: 22.671
    - type: ndcg_at_10
      value: 26.540000000000003
    - type: ndcg_at_100
      value: 34.138000000000005
    - type: ndcg_at_1000
      value: 37.72
    - type: ndcg_at_3
      value: 20.766000000000002
    - type: ndcg_at_5
      value: 22.927
    - type: precision_at_1
      value: 22.671
    - type: precision_at_10
      value: 8.619
    - type: precision_at_100
      value: 1.678
    - type: precision_at_1000
      value: 0.23500000000000001
    - type: precision_at_3
      value: 15.592
    - type: precision_at_5
      value: 12.43
    - type: recall_at_1
      value: 10.144
    - type: recall_at_10
      value: 33.46
    - type: recall_at_100
      value: 59.758
    - type: recall_at_1000
      value: 79.704
    - type: recall_at_3
      value: 19.604
    - type: recall_at_5
      value: 25.367
  - task:
      type: Retrieval
    dataset:
      type: dbpedia-entity
      name: MTEB DBPedia
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 8.654
    - type: map_at_10
      value: 18.506
    - type: map_at_100
      value: 26.412999999999997
    - type: map_at_1000
      value: 28.13
    - type: map_at_3
      value: 13.379
    - type: map_at_5
      value: 15.529000000000002
    - type: mrr_at_1
      value: 66.0
    - type: mrr_at_10
      value: 74.13
    - type: mrr_at_100
      value: 74.48700000000001
    - type: mrr_at_1000
      value: 74.49799999999999
    - type: mrr_at_3
      value: 72.75
    - type: mrr_at_5
      value: 73.762
    - type: ndcg_at_1
      value: 54.50000000000001
    - type: ndcg_at_10
      value: 40.236
    - type: ndcg_at_100
      value: 44.690999999999995
    - type: ndcg_at_1000
      value: 52.195
    - type: ndcg_at_3
      value: 45.632
    - type: ndcg_at_5
      value: 42.952
    - type: precision_at_1
      value: 66.0
    - type: precision_at_10
      value: 31.724999999999998
    - type: precision_at_100
      value: 10.299999999999999
    - type: precision_at_1000
      value: 2.194
    - type: precision_at_3
      value: 48.75
    - type: precision_at_5
      value: 41.6
    - type: recall_at_1
      value: 8.654
    - type: recall_at_10
      value: 23.74
    - type: recall_at_100
      value: 50.346999999999994
    - type: recall_at_1000
      value: 74.376
    - type: recall_at_3
      value: 14.636
    - type: recall_at_5
      value: 18.009
  - task:
      type: Classification
    dataset:
      type: mteb/emotion
      name: MTEB EmotionClassification
      config: default
      split: test
      revision: 4f58c6b202a23cf9a4da393831edf4f9183cad37
    metrics:
    - type: accuracy
      value: 53.245
    - type: f1
      value: 48.74520523753552
  - task:
      type: Retrieval
    dataset:
      type: fever
      name: MTEB FEVER
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 51.729
    - type: map_at_10
      value: 63.904
    - type: map_at_100
      value: 64.363
    - type: map_at_1000
      value: 64.38199999999999
    - type: map_at_3
      value: 61.393
    - type: map_at_5
      value: 63.02100000000001
    - type: mrr_at_1
      value: 55.686
    - type: mrr_at_10
      value: 67.804
    - type: mrr_at_100
      value: 68.15299999999999
    - type: mrr_at_1000
      value: 68.161
    - type: mrr_at_3
      value: 65.494
    - type: mrr_at_5
      value: 67.01599999999999
    - type: ndcg_at_1
      value: 55.686
    - type: ndcg_at_10
      value: 70.025
    - type: ndcg_at_100
      value: 72.011
    - type: ndcg_at_1000
      value: 72.443
    - type: ndcg_at_3
      value: 65.32900000000001
    - type: ndcg_at_5
      value: 68.05600000000001
    - type: precision_at_1
      value: 55.686
    - type: precision_at_10
      value: 9.358
    - type: precision_at_100
      value: 1.05
    - type: precision_at_1000
      value: 0.11
    - type: precision_at_3
      value: 26.318
    - type: precision_at_5
      value: 17.321
    - type: recall_at_1
      value: 51.729
    - type: recall_at_10
      value: 85.04
    - type: recall_at_100
      value: 93.777
    - type: recall_at_1000
      value: 96.824
    - type: recall_at_3
      value: 72.521
    - type: recall_at_5
      value: 79.148
  - task:
      type: Retrieval
    dataset:
      type: fiqa
      name: MTEB FiQA2018
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 23.765
    - type: map_at_10
      value: 39.114
    - type: map_at_100
      value: 40.987
    - type: map_at_1000
      value: 41.155
    - type: map_at_3
      value: 34.028000000000006
    - type: map_at_5
      value: 36.925000000000004
    - type: mrr_at_1
      value: 46.451
    - type: mrr_at_10
      value: 54.711
    - type: mrr_at_100
      value: 55.509
    - type: mrr_at_1000
      value: 55.535000000000004
    - type: mrr_at_3
      value: 52.649
    - type: mrr_at_5
      value: 53.729000000000006
    - type: ndcg_at_1
      value: 46.451
    - type: ndcg_at_10
      value: 46.955999999999996
    - type: ndcg_at_100
      value: 53.686
    - type: ndcg_at_1000
      value: 56.230000000000004
    - type: ndcg_at_3
      value: 43.374
    - type: ndcg_at_5
      value: 44.372
    - type: precision_at_1
      value: 46.451
    - type: precision_at_10
      value: 13.256
    - type: precision_at_100
      value: 2.019
    - type: precision_at_1000
      value: 0.247
    - type: precision_at_3
      value: 29.115000000000002
    - type: precision_at_5
      value: 21.389
    - type: recall_at_1
      value: 23.765
    - type: recall_at_10
      value: 53.452999999999996
    - type: recall_at_100
      value: 78.828
    - type: recall_at_1000
      value: 93.938
    - type: recall_at_3
      value: 39.023
    - type: recall_at_5
      value: 45.18
  - task:
      type: Retrieval
    dataset:
      type: hotpotqa
      name: MTEB HotpotQA
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 31.918000000000003
    - type: map_at_10
      value: 46.741
    - type: map_at_100
      value: 47.762
    - type: map_at_1000
      value: 47.849000000000004
    - type: map_at_3
      value: 43.578
    - type: map_at_5
      value: 45.395
    - type: mrr_at_1
      value: 63.834999999999994
    - type: mrr_at_10
      value: 71.312
    - type: mrr_at_100
      value: 71.695
    - type: mrr_at_1000
      value: 71.714
    - type: mrr_at_3
      value: 69.82000000000001
    - type: mrr_at_5
      value: 70.726
    - type: ndcg_at_1
      value: 63.834999999999994
    - type: ndcg_at_10
      value: 55.879999999999995
    - type: ndcg_at_100
      value: 59.723000000000006
    - type: ndcg_at_1000
      value: 61.49400000000001
    - type: ndcg_at_3
      value: 50.964
    - type: ndcg_at_5
      value: 53.47
    - type: precision_at_1
      value: 63.834999999999994
    - type: precision_at_10
      value: 11.845
    - type: precision_at_100
      value: 1.4869999999999999
    - type: precision_at_1000
      value: 0.172
    - type: precision_at_3
      value: 32.158
    - type: precision_at_5
      value: 21.278
    - type: recall_at_1
      value: 31.918000000000003
    - type: recall_at_10
      value: 59.223000000000006
    - type: recall_at_100
      value: 74.328
    - type: recall_at_1000
      value: 86.05000000000001
    - type: recall_at_3
      value: 48.238
    - type: recall_at_5
      value: 53.193999999999996
  - task:
      type: Classification
    dataset:
      type: mteb/imdb
      name: MTEB ImdbClassification
      config: default
      split: test
      revision: 3d86128a09e091d6018b6d26cad27f2739fc2db7
    metrics:
    - type: accuracy
      value: 79.7896
    - type: ap
      value: 73.65166029460288
    - type: f1
      value: 79.71794693711813
  - task:
      type: Retrieval
    dataset:
      type: msmarco
      name: MTEB MSMARCO
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 22.239
    - type: map_at_10
      value: 34.542
    - type: map_at_100
      value: 35.717999999999996
    - type: map_at_1000
      value: 35.764
    - type: map_at_3
      value: 30.432
    - type: map_at_5
      value: 32.81
    - type: mrr_at_1
      value: 22.908
    - type: mrr_at_10
      value: 35.127
    - type: mrr_at_100
      value: 36.238
    - type: mrr_at_1000
      value: 36.278
    - type: mrr_at_3
      value: 31.076999999999998
    - type: mrr_at_5
      value: 33.419
    - type: ndcg_at_1
      value: 22.908
    - type: ndcg_at_10
      value: 41.607
    - type: ndcg_at_100
      value: 47.28
    - type: ndcg_at_1000
      value: 48.414
    - type: ndcg_at_3
      value: 33.253
    - type: ndcg_at_5
      value: 37.486000000000004
    - type: precision_at_1
      value: 22.908
    - type: precision_at_10
      value: 6.645
    - type: precision_at_100
      value: 0.9490000000000001
    - type: precision_at_1000
      value: 0.105
    - type: precision_at_3
      value: 14.130999999999998
    - type: precision_at_5
      value: 10.616
    - type: recall_at_1
      value: 22.239
    - type: recall_at_10
      value: 63.42
    - type: recall_at_100
      value: 89.696
    - type: recall_at_1000
      value: 98.351
    - type: recall_at_3
      value: 40.77
    - type: recall_at_5
      value: 50.93
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_domain
      name: MTEB MTOPDomainClassification (en)
      config: en
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 95.06839945280439
    - type: f1
      value: 94.74276398224072
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_intent
      name: MTEB MTOPIntentClassification (en)
      config: en
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 72.25718194254446
    - type: f1
      value: 53.91164489161391
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (en)
      config: en
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 71.47948890383323
    - type: f1
      value: 69.98520247230257
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (en)
      config: en
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 76.46603900470748
    - type: f1
      value: 76.44111526065399
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-p2p
      name: MTEB MedrxivClusteringP2P
      config: default
      split: test
      revision: e7a26af6f3ae46b30dde8737f02c07b1505bcc73
    metrics:
    - type: v_measure
      value: 33.19106070798198
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-s2s
      name: MTEB MedrxivClusteringS2S
      config: default
      split: test
      revision: 35191c8c0dca72d8ff3efcd72aa802307d469663
    metrics:
    - type: v_measure
      value: 30.78772205248094
  - task:
      type: Reranking
    dataset:
      type: mteb/mind_small
      name: MTEB MindSmallReranking
      config: default
      split: test
      revision: 3bdac13927fdc888b903db93b2ffdbd90b295a69
    metrics:
    - type: map
      value: 31.811231631488507
    - type: mrr
      value: 32.98200485378021
  - task:
      type: Retrieval
    dataset:
      type: nfcorpus
      name: MTEB NFCorpus
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 6.9
    - type: map_at_10
      value: 13.703000000000001
    - type: map_at_100
      value: 17.251
    - type: map_at_1000
      value: 18.795
    - type: map_at_3
      value: 10.366999999999999
    - type: map_at_5
      value: 11.675
    - type: mrr_at_1
      value: 47.059
    - type: mrr_at_10
      value: 55.816
    - type: mrr_at_100
      value: 56.434
    - type: mrr_at_1000
      value: 56.467
    - type: mrr_at_3
      value: 53.973000000000006
    - type: mrr_at_5
      value: 55.257999999999996
    - type: ndcg_at_1
      value: 44.737
    - type: ndcg_at_10
      value: 35.997
    - type: ndcg_at_100
      value: 33.487
    - type: ndcg_at_1000
      value: 41.897
    - type: ndcg_at_3
      value: 41.18
    - type: ndcg_at_5
      value: 38.721
    - type: precision_at_1
      value: 46.129999999999995
    - type: precision_at_10
      value: 26.533
    - type: precision_at_100
      value: 8.706
    - type: precision_at_1000
      value: 2.16
    - type: precision_at_3
      value: 38.493
    - type: precision_at_5
      value: 33.189
    - type: recall_at_1
      value: 6.9
    - type: recall_at_10
      value: 17.488999999999997
    - type: recall_at_100
      value: 34.583000000000006
    - type: recall_at_1000
      value: 64.942
    - type: recall_at_3
      value: 11.494
    - type: recall_at_5
      value: 13.496
  - task:
      type: Retrieval
    dataset:
      type: nq
      name: MTEB NQ
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 33.028999999999996
    - type: map_at_10
      value: 49.307
    - type: map_at_100
      value: 50.205
    - type: map_at_1000
      value: 50.23
    - type: map_at_3
      value: 44.782
    - type: map_at_5
      value: 47.599999999999994
    - type: mrr_at_1
      value: 37.108999999999995
    - type: mrr_at_10
      value: 51.742999999999995
    - type: mrr_at_100
      value: 52.405
    - type: mrr_at_1000
      value: 52.422000000000004
    - type: mrr_at_3
      value: 48.087999999999994
    - type: mrr_at_5
      value: 50.414
    - type: ndcg_at_1
      value: 37.08
    - type: ndcg_at_10
      value: 57.236
    - type: ndcg_at_100
      value: 60.931999999999995
    - type: ndcg_at_1000
      value: 61.522
    - type: ndcg_at_3
      value: 48.93
    - type: ndcg_at_5
      value: 53.561
    - type: precision_at_1
      value: 37.08
    - type: precision_at_10
      value: 9.386
    - type: precision_at_100
      value: 1.1480000000000001
    - type: precision_at_1000
      value: 0.12
    - type: precision_at_3
      value: 22.258
    - type: precision_at_5
      value: 16.025
    - type: recall_at_1
      value: 33.028999999999996
    - type: recall_at_10
      value: 78.805
    - type: recall_at_100
      value: 94.643
    - type: recall_at_1000
      value: 99.039
    - type: recall_at_3
      value: 57.602
    - type: recall_at_5
      value: 68.253
  - task:
      type: Retrieval
    dataset:
      type: quora
      name: MTEB QuoraRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 71.122
    - type: map_at_10
      value: 85.237
    - type: map_at_100
      value: 85.872
    - type: map_at_1000
      value: 85.885
    - type: map_at_3
      value: 82.27499999999999
    - type: map_at_5
      value: 84.13199999999999
    - type: mrr_at_1
      value: 81.73
    - type: mrr_at_10
      value: 87.834
    - type: mrr_at_100
      value: 87.92
    - type: mrr_at_1000
      value: 87.921
    - type: mrr_at_3
      value: 86.878
    - type: mrr_at_5
      value: 87.512
    - type: ndcg_at_1
      value: 81.73
    - type: ndcg_at_10
      value: 88.85499999999999
    - type: ndcg_at_100
      value: 89.992
    - type: ndcg_at_1000
      value: 90.07
    - type: ndcg_at_3
      value: 85.997
    - type: ndcg_at_5
      value: 87.55199999999999
    - type: precision_at_1
      value: 81.73
    - type: precision_at_10
      value: 13.491
    - type: precision_at_100
      value: 1.536
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.623
    - type: precision_at_5
      value: 24.742
    - type: recall_at_1
      value: 71.122
    - type: recall_at_10
      value: 95.935
    - type: recall_at_100
      value: 99.657
    - type: recall_at_1000
      value: 99.996
    - type: recall_at_3
      value: 87.80799999999999
    - type: recall_at_5
      value: 92.161
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering
      name: MTEB RedditClustering
      config: default
      split: test
      revision: 24640382cdbf8abc73003fb0fa6d111a705499eb
    metrics:
    - type: v_measure
      value: 63.490029238193756
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering-p2p
      name: MTEB RedditClusteringP2P
      config: default
      split: test
      revision: 282350215ef01743dc01b456c7f5241fa8937f16
    metrics:
    - type: v_measure
      value: 65.13153408508836
  - task:
      type: Retrieval
    dataset:
      type: scidocs
      name: MTEB SCIDOCS
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 4.202999999999999
    - type: map_at_10
      value: 10.174
    - type: map_at_100
      value: 12.138
    - type: map_at_1000
      value: 12.418
    - type: map_at_3
      value: 7.379
    - type: map_at_5
      value: 8.727
    - type: mrr_at_1
      value: 20.7
    - type: mrr_at_10
      value: 30.389
    - type: mrr_at_100
      value: 31.566
    - type: mrr_at_1000
      value: 31.637999999999998
    - type: mrr_at_3
      value: 27.133000000000003
    - type: mrr_at_5
      value: 29.078
    - type: ndcg_at_1
      value: 20.7
    - type: ndcg_at_10
      value: 17.355999999999998
    - type: ndcg_at_100
      value: 25.151
    - type: ndcg_at_1000
      value: 30.37
    - type: ndcg_at_3
      value: 16.528000000000002
    - type: ndcg_at_5
      value: 14.396999999999998
    - type: precision_at_1
      value: 20.7
    - type: precision_at_10
      value: 8.98
    - type: precision_at_100
      value: 2.015
    - type: precision_at_1000
      value: 0.327
    - type: precision_at_3
      value: 15.367
    - type: precision_at_5
      value: 12.559999999999999
    - type: recall_at_1
      value: 4.202999999999999
    - type: recall_at_10
      value: 18.197
    - type: recall_at_100
      value: 40.903
    - type: recall_at_1000
      value: 66.427
    - type: recall_at_3
      value: 9.362
    - type: recall_at_5
      value: 12.747
  - task:
      type: STS
    dataset:
      type: mteb/sickr-sts
      name: MTEB SICK-R
      config: default
      split: test
      revision: a6ea5a8cab320b040a23452cc28066d9beae2cee
    metrics:
    - type: cos_sim_spearman
      value: 81.69890989765257
  - task:
      type: STS
    dataset:
      type: mteb/sts12-sts
      name: MTEB STS12
      config: default
      split: test
      revision: a0d554a64d88156834ff5ae9920b964011b16384
    metrics:
    - type: cos_sim_spearman
      value: 75.31953790551489
  - task:
      type: STS
    dataset:
      type: mteb/sts13-sts
      name: MTEB STS13
      config: default
      split: test
      revision: 7e90230a92c190f1bf69ae9002b8cea547a64cca
    metrics:
    - type: cos_sim_spearman
      value: 87.44050861280759
  - task:
      type: STS
    dataset:
      type: mteb/sts14-sts
      name: MTEB STS14
      config: default
      split: test
      revision: 6031580fec1f6af667f0bd2da0a551cf4f0b2375
    metrics:
    - type: cos_sim_spearman
      value: 81.86922869270393
  - task:
      type: STS
    dataset:
      type: mteb/sts15-sts
      name: MTEB STS15
      config: default
      split: test
      revision: ae752c7c21bf194d8b67fd573edf7ae58183cbe3
    metrics:
    - type: cos_sim_spearman
      value: 88.9399170304284
  - task:
      type: STS
    dataset:
      type: mteb/sts16-sts
      name: MTEB STS16
      config: default
      split: test
      revision: 4d8694f8f0e0100860b497b999b3dbed754a0513
    metrics:
    - type: cos_sim_spearman
      value: 85.38015314088582
  - task:
      type: STS
    dataset:
      type: mteb/sts17-crosslingual-sts
      name: MTEB STS17 (en-en)
      config: en-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_spearman
      value: 90.53653527788835
  - task:
      type: STS
    dataset:
      type: mteb/sts22-crosslingual-sts
      name: MTEB STS22 (en)
      config: en
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_spearman
      value: 68.64526474250209
  - task:
      type: STS
    dataset:
      type: mteb/stsbenchmark-sts
      name: MTEB STSBenchmark
      config: default
      split: test
      revision: b0fddb56ed78048fa8b90373c8a3cfc37b684831
    metrics:
    - type: cos_sim_spearman
      value: 86.56156983963042
  - task:
      type: Reranking
    dataset:
      type: mteb/scidocs-reranking
      name: MTEB SciDocsRR
      config: default
      split: test
      revision: d3c5e1fc0b855ab6097bf1cda04dd73947d7caab
    metrics:
    - type: map
      value: 79.48610254648003
    - type: mrr
      value: 94.02481505422682
  - task:
      type: Retrieval
    dataset:
      type: scifact
      name: MTEB SciFact
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 48.983
    - type: map_at_10
      value: 59.077999999999996
    - type: map_at_100
      value: 59.536
    - type: map_at_1000
      value: 59.575
    - type: map_at_3
      value: 55.691
    - type: map_at_5
      value: 57.410000000000004
    - type: mrr_at_1
      value: 51.666999999999994
    - type: mrr_at_10
      value: 60.427
    - type: mrr_at_100
      value: 60.763
    - type: mrr_at_1000
      value: 60.79900000000001
    - type: mrr_at_3
      value: 57.556
    - type: mrr_at_5
      value: 59.089000000000006
    - type: ndcg_at_1
      value: 51.666999999999994
    - type: ndcg_at_10
      value: 64.559
    - type: ndcg_at_100
      value: 66.58
    - type: ndcg_at_1000
      value: 67.64
    - type: ndcg_at_3
      value: 58.287
    - type: ndcg_at_5
      value: 61.001000000000005
    - type: precision_at_1
      value: 51.666999999999994
    - type: precision_at_10
      value: 9.067
    - type: precision_at_100
      value: 1.0170000000000001
    - type: precision_at_1000
      value: 0.11100000000000002
    - type: precision_at_3
      value: 23.0
    - type: precision_at_5
      value: 15.6
    - type: recall_at_1
      value: 48.983
    - type: recall_at_10
      value: 80.289
    - type: recall_at_100
      value: 89.43299999999999
    - type: recall_at_1000
      value: 97.667
    - type: recall_at_3
      value: 62.978
    - type: recall_at_5
      value: 69.872
  - task:
      type: PairClassification
    dataset:
      type: mteb/sprintduplicatequestions-pairclassification
      name: MTEB SprintDuplicateQuestions
      config: default
      split: test
      revision: d66bd1f72af766a5cc4b0ca5e00c162f89e8cc46
    metrics:
    - type: cos_sim_accuracy
      value: 99.79009900990098
    - type: cos_sim_ap
      value: 94.94115052608419
    - type: cos_sim_f1
      value: 89.1260162601626
    - type: cos_sim_precision
      value: 90.599173553719
    - type: cos_sim_recall
      value: 87.7
    - type: dot_accuracy
      value: 99.79009900990098
    - type: dot_ap
      value: 94.94115052608419
    - type: dot_f1
      value: 89.1260162601626
    - type: dot_precision
      value: 90.599173553719
    - type: dot_recall
      value: 87.7
    - type: euclidean_accuracy
      value: 99.79009900990098
    - type: euclidean_ap
      value: 94.94115052608419
    - type: euclidean_f1
      value: 89.1260162601626
    - type: euclidean_precision
      value: 90.599173553719
    - type: euclidean_recall
      value: 87.7
    - type: manhattan_accuracy
      value: 99.7940594059406
    - type: manhattan_ap
      value: 94.95271414642431
    - type: manhattan_f1
      value: 89.24508790072387
    - type: manhattan_precision
      value: 92.3982869379015
    - type: manhattan_recall
      value: 86.3
    - type: max_accuracy
      value: 99.7940594059406
    - type: max_ap
      value: 94.95271414642431
    - type: max_f1
      value: 89.24508790072387
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering
      name: MTEB StackExchangeClustering
      config: default
      split: test
      revision: 6cbc1f7b2bc0622f2e39d2c77fa502909748c259
    metrics:
    - type: v_measure
      value: 68.43866571935851
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering-p2p
      name: MTEB StackExchangeClusteringP2P
      config: default
      split: test
      revision: 815ca46b2622cec33ccafc3735d572c266efdb44
    metrics:
    - type: v_measure
      value: 35.16579026551532
  - task:
      type: Reranking
    dataset:
      type: mteb/stackoverflowdupquestions-reranking
      name: MTEB StackOverflowDupQuestions
      config: default
      split: test
      revision: e185fbe320c72810689fc5848eb6114e1ef5ec69
    metrics:
    - type: map
      value: 52.518952473513934
    - type: mrr
      value: 53.292457134368895
  - task:
      type: Summarization
    dataset:
      type: mteb/summeval
      name: MTEB SummEval
      config: default
      split: test
      revision: cda12ad7615edc362dbf25a00fdd61d3b1eaf93c
    metrics:
    - type: cos_sim_pearson
      value: 31.12529588316604
    - type: cos_sim_spearman
      value: 32.31662126895294
    - type: dot_pearson
      value: 31.125303796647056
    - type: dot_spearman
      value: 32.31662126895294
  - task:
      type: Retrieval
    dataset:
      type: trec-covid
      name: MTEB TRECCOVID
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 0.219
    - type: map_at_10
      value: 1.7469999999999999
    - type: map_at_100
      value: 10.177999999999999
    - type: map_at_1000
      value: 26.108999999999998
    - type: map_at_3
      value: 0.64
    - type: map_at_5
      value: 0.968
    - type: mrr_at_1
      value: 82.0
    - type: mrr_at_10
      value: 89.067
    - type: mrr_at_100
      value: 89.067
    - type: mrr_at_1000
      value: 89.067
    - type: mrr_at_3
      value: 88.333
    - type: mrr_at_5
      value: 88.73299999999999
    - type: ndcg_at_1
      value: 78.0
    - type: ndcg_at_10
      value: 71.398
    - type: ndcg_at_100
      value: 55.574999999999996
    - type: ndcg_at_1000
      value: 51.771
    - type: ndcg_at_3
      value: 77.765
    - type: ndcg_at_5
      value: 73.614
    - type: precision_at_1
      value: 82.0
    - type: precision_at_10
      value: 75.4
    - type: precision_at_100
      value: 58.040000000000006
    - type: precision_at_1000
      value: 23.516000000000002
    - type: precision_at_3
      value: 84.0
    - type: precision_at_5
      value: 78.4
    - type: recall_at_1
      value: 0.219
    - type: recall_at_10
      value: 1.958
    - type: recall_at_100
      value: 13.797999999999998
    - type: recall_at_1000
      value: 49.881
    - type: recall_at_3
      value: 0.672
    - type: recall_at_5
      value: 1.0370000000000001
  - task:
      type: Retrieval
    dataset:
      type: webis-touche2020
      name: MTEB Touche2020
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 1.8610000000000002
    - type: map_at_10
      value: 8.705
    - type: map_at_100
      value: 15.164
    - type: map_at_1000
      value: 16.78
    - type: map_at_3
      value: 4.346
    - type: map_at_5
      value: 6.151
    - type: mrr_at_1
      value: 22.448999999999998
    - type: mrr_at_10
      value: 41.556
    - type: mrr_at_100
      value: 42.484
    - type: mrr_at_1000
      value: 42.494
    - type: mrr_at_3
      value: 37.755
    - type: mrr_at_5
      value: 40.102
    - type: ndcg_at_1
      value: 21.429000000000002
    - type: ndcg_at_10
      value: 23.439
    - type: ndcg_at_100
      value: 36.948
    - type: ndcg_at_1000
      value: 48.408
    - type: ndcg_at_3
      value: 22.261
    - type: ndcg_at_5
      value: 23.085
    - type: precision_at_1
      value: 22.448999999999998
    - type: precision_at_10
      value: 21.633
    - type: precision_at_100
      value: 8.02
    - type: precision_at_1000
      value: 1.5939999999999999
    - type: precision_at_3
      value: 23.810000000000002
    - type: precision_at_5
      value: 24.490000000000002
    - type: recall_at_1
      value: 1.8610000000000002
    - type: recall_at_10
      value: 15.876000000000001
    - type: recall_at_100
      value: 50.300999999999995
    - type: recall_at_1000
      value: 86.098
    - type: recall_at_3
      value: 5.892
    - type: recall_at_5
      value: 9.443
  - task:
      type: Classification
    dataset:
      type: mteb/toxic_conversations_50k
      name: MTEB ToxicConversationsClassification
      config: default
      split: test
      revision: d7c0de2777da35d6aae2200a62c6e0e5af397c4c
    metrics:
    - type: accuracy
      value: 70.3264
    - type: ap
      value: 13.249577616243794
    - type: f1
      value: 53.621518367695685
  - task:
      type: Classification
    dataset:
      type: mteb/tweet_sentiment_extraction
      name: MTEB TweetSentimentExtractionClassification
      config: default
      split: test
      revision: d604517c81ca91fe16a244d1248fc021f9ecee7a
    metrics:
    - type: accuracy
      value: 61.57611771363894
    - type: f1
      value: 61.79797478568639
  - task:
      type: Clustering
    dataset:
      type: mteb/twentynewsgroups-clustering
      name: MTEB TwentyNewsgroupsClustering
      config: default
      split: test
      revision: 6125ec4e24fa026cec8a478383ee943acfbd5449
    metrics:
    - type: v_measure
      value: 53.38315344479284
  - task:
      type: PairClassification
    dataset:
      type: mteb/twittersemeval2015-pairclassification
      name: MTEB TwitterSemEval2015
      config: default
      split: test
      revision: 70970daeab8776df92f5ea462b6173c0b46fd2d1
    metrics:
    - type: cos_sim_accuracy
      value: 87.55438993860642
    - type: cos_sim_ap
      value: 77.98702600017738
    - type: cos_sim_f1
      value: 71.94971653931476
    - type: cos_sim_precision
      value: 67.50693802035153
    - type: cos_sim_recall
      value: 77.01846965699208
    - type: dot_accuracy
      value: 87.55438993860642
    - type: dot_ap
      value: 77.98702925907986
    - type: dot_f1
      value: 71.94971653931476
    - type: dot_precision
      value: 67.50693802035153
    - type: dot_recall
      value: 77.01846965699208
    - type: euclidean_accuracy
      value: 87.55438993860642
    - type: euclidean_ap
      value: 77.98702951957925
    - type: euclidean_f1
      value: 71.94971653931476
    - type: euclidean_precision
      value: 67.50693802035153
    - type: euclidean_recall
      value: 77.01846965699208
    - type: manhattan_accuracy
      value: 87.54246885617214
    - type: manhattan_ap
      value: 77.95531413902947
    - type: manhattan_f1
      value: 71.93605683836589
    - type: manhattan_precision
      value: 69.28152492668622
    - type: manhattan_recall
      value: 74.80211081794195
    - type: max_accuracy
      value: 87.55438993860642
    - type: max_ap
      value: 77.98702951957925
    - type: max_f1
      value: 71.94971653931476
  - task:
      type: PairClassification
    dataset:
      type: mteb/twitterurlcorpus-pairclassification
      name: MTEB TwitterURLCorpus
      config: default
      split: test
      revision: 8b6510b0b1fa4e4c4f879467980e9be563ec1cdf
    metrics:
    - type: cos_sim_accuracy
      value: 89.47296930182016
    - type: cos_sim_ap
      value: 86.92853616302108
    - type: cos_sim_f1
      value: 79.35138351681047
    - type: cos_sim_precision
      value: 76.74820143884892
    - type: cos_sim_recall
      value: 82.13735756082538
    - type: dot_accuracy
      value: 89.47296930182016
    - type: dot_ap
      value: 86.92854339601595
    - type: dot_f1
      value: 79.35138351681047
    - type: dot_precision
      value: 76.74820143884892
    - type: dot_recall
      value: 82.13735756082538
    - type: euclidean_accuracy
      value: 89.47296930182016
    - type: euclidean_ap
      value: 86.92854191061649
    - type: euclidean_f1
      value: 79.35138351681047
    - type: euclidean_precision
      value: 76.74820143884892
    - type: euclidean_recall
      value: 82.13735756082538
    - type: manhattan_accuracy
      value: 89.47685023479644
    - type: manhattan_ap
      value: 86.90063722679578
    - type: manhattan_f1
      value: 79.30753865502702
    - type: manhattan_precision
      value: 76.32066068631639
    - type: manhattan_recall
      value: 82.53772713273791
    - type: max_accuracy
      value: 89.47685023479644
    - type: max_ap
      value: 86.92854339601595
    - type: max_f1
      value: 79.35138351681047
---