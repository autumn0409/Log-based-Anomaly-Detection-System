# Log-based Anomaly Detection System
The final project of deep learning and practice (summer 2020) in NCTU.

First, we adopt Drain to parse log messages to extract log events (templates).
Then, we extracts semantic information of log events and represents them as semantic vectors using sentense-bert.
Finally, we detects anomalies by utilizing an attention-based Bi-LSTM model, which has the ability to capture the contextual
information in the log sequences and automatically learn the importance of different log events.

We have evaluated our system using the public HDFS dataset, and the
recall, precision and F1-score achieved by it are 0.9977, 0.9691 and 0.9832, respectively

## Preprocessing Order
* unzip HDFS_1.tar.gz into log_data/
* parse_log.py
* preprocessing.py

## Data Discription
* total: 575,061 blocks
    - 16,838 anomaly blocks
    - 558,223 normal blocks
* training: randomly select
    - 6,000 anomaly blocks
    - 6,000 normal blocks
* testing: rest data

## Reference
* [https://dl.acm.org/doi/10.1145/3338906.3338931](https://dl.acm.org/doi/10.1145/3338906.3338931)
* [https://ieeexplore.ieee.org/document/8029742](https://ieeexplore.ieee.org/document/8029742)
* [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
