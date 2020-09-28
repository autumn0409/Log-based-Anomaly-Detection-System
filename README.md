# Log-based Anomaly Detection System
The final project of deep learning and practice (summer 2020) in NCTU.

First, we adopt Drain to parse log messages to extract log events (templates).
Then, we extracts semantic information of log events and represents them as semantic vectors.
Finally, we detects anomalies by utilizing an attention-based Bi-LSTM model, which has the ability to capture the contextual
information in the log sequences and automatically learn the importance of different log events.

We have evaluated our system using the public HDFS dataset, and the
recall, precision and F1-score achieved by it are 0.9991, 0.9318 and 0.9643, respectively

## Data Discription
* reference: [https://github.com/logpai/loghub](https://github.com/logpai/loghub)
* total: 575,061 blocks
    - 16,838 anomaly blocks
    - 558,223 normal blocks
* training: randomly select
    - 6,000 anomaly blocks
    - 6,000 normal blocks
* testing: rest data

## Preprocessing Order
1. unzip HDFS_1.tar.gz into log_data/
2. parse_log.py
3. embbedding_table.py
4. IDF.py
5. template2vec.py
6. preprocessing.py

## Reference
* [https://dl.acm.org/doi/10.1145/3338906.3338931](https://dl.acm.org/doi/10.1145/3338906.3338931)
* [http://jiemingzhu.github.io/pub/pjhe_icws2017.pdf](http://jiemingzhu.github.io/pub/pjhe_icws2017.pdf)
* [https://arxiv.org/pdf/1612.03651.pdf](https://arxiv.org/pdf/1612.03651.pdf)
