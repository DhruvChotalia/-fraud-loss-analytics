# Data profile

Generated 2026-09-27 from the latest successful load of each source file. Only records with valid amount, fraud flag and timestamp are profiled.

## Overview by source file

| source_file    |   transactions |   cards |   merchants | first_day   | last_day   |   total_amount |   fraud_txns |   fraud_rate_pct |   fraud_amount |
|:---------------|---------------:|--------:|------------:|:------------|:-----------|---------------:|-------------:|-----------------:|---------------:|
| fraudTest.csv  |         555719 |     924 |         693 | 2020-06-21  | 2020-12-31 |    3.85629e+07 |         2145 |            0.386 |    1.13332e+06 |
| fraudTrain.csv |        1296675 |     983 |         693 | 2019-01-01  | 2020-06-21 |    9.12224e+07 |         7506 |            0.579 |    3.98809e+06 |

## Fraud by category (highest fraud rate first)

| category       |   transactions |   fraud_txns |   fraud_rate_pct |     fraud_amount |   avg_fraud_ticket |   avg_legit_ticket |
|:---------------|---------------:|-------------:|-----------------:|-----------------:|-------------------:|-------------------:|
| shopping_net   |         139322 |         2219 |            1.593 |      2.21485e+06 |             998.13 |              72.19 |
| misc_net       |          90654 |         1182 |            1.304 | 944010           |             798.65 |              70.69 |
| grocery_pos    |         176191 |         2228 |            1.265 | 695665           |             312.24 |             114.14 |
| shopping_pos   |         166463 |         1056 |            0.634 | 928132           |             878.91 |              73.8  |
| gas_transport  |         188029 |          772 |            0.411 |   9442.53        |              12.23 |              63.69 |
| misc_pos       |         114229 |          322 |            0.282 |  68494.8         |             212.72 |              62.25 |
| grocery_net    |          64878 |          175 |            0.27  |   2108.21        |              12.05 |              53.8  |
| travel         |          57956 |          156 |            0.269 |   1399.47        |               8.97 |             112.05 |
| personal_care  |         130085 |          290 |            0.223 |   7571.96        |              26.11 |              48.1  |
| entertainment  |         134118 |          292 |            0.218 | 147400           |             504.79 |              63.18 |
| kids_pets      |         161727 |          304 |            0.188 |   5619.53        |              18.49 |              57.6  |
| food_dining    |         130729 |          205 |            0.157 |  24739.2         |             120.68 |              50.88 |
| home           |         175460 |          265 |            0.151 |  68232           |             257.48 |              57.89 |
| health_fitness |         122553 |          185 |            0.151 |   3751.36        |              20.28 |              54.14 |

## Top 10 merchants by fraud loss

| merchant                             |   transactions |   fraud_txns |   fraud_amount |
|:-------------------------------------|---------------:|-------------:|---------------:|
| fraud_Kozey-Boehm                    |           2758 |           60 |        60410.6 |
| fraud_Terry-Huel                     |           2864 |           56 |        55203.8 |
| fraud_Kuhic LLC                      |           2842 |           53 |        54282.9 |
| fraud_Mosciski, Ziemann and Farrell  |           2821 |           53 |        52948.7 |
| fraud_Heathcote, Yost and Kertzmann  |           2786 |           51 |        51708.5 |
| fraud_Langworth, Boehm and Gulgowski |           2817 |           52 |        51617.9 |
| fraud_Boyer-Reichert                 |           2779 |           51 |        51608.7 |
| fraud_Boyer PLC                      |           4999 |           55 |        51363.5 |
| fraud_Schmeler, Bashirian and Price  |           2788 |           52 |        51237.9 |
| fraud_Jast Ltd                       |           2757 |           51 |        50856.6 |

## Fraud by hour of day

|   hour |   transactions |   fraud_txns |   fraud_rate_pct |
|-------:|---------------:|-------------:|-----------------:|
|      0 |          60655 |          823 |            1.357 |
|      1 |          61330 |          827 |            1.348 |
|      2 |          60796 |          793 |            1.304 |
|      3 |          60968 |          803 |            1.317 |
|      4 |          59938 |           61 |            0.102 |
|      5 |          60088 |           80 |            0.133 |
|      6 |          60406 |           54 |            0.089 |
|      7 |          60301 |           72 |            0.119 |
|      8 |          60498 |           59 |            0.098 |
|      9 |          60231 |           61 |            0.101 |
|     10 |          60320 |           52 |            0.086 |
|     11 |          60170 |           59 |            0.098 |
|     12 |          93294 |           84 |            0.09  |
|     13 |          93492 |           94 |            0.101 |
|     14 |          93089 |          100 |            0.107 |
|     15 |          93439 |          100 |            0.107 |
|     16 |          94289 |           97 |            0.103 |
|     17 |          93514 |           94 |            0.101 |
|     18 |          94052 |          111 |            0.118 |
|     19 |          93433 |          105 |            0.112 |
|     20 |          93081 |           98 |            0.105 |
|     21 |          93738 |          101 |            0.108 |
|     22 |          95370 |         2481 |            2.601 |
|     23 |          95902 |         2442 |            2.546 |

## Monthly fraud loss

| month   |   transactions |   fraud_txns |   fraud_amount |
|:--------|---------------:|-------------:|---------------:|
| 2019-01 |          52525 |          506 |         261780 |
| 2019-02 |          49866 |          517 |         274051 |
| 2019-03 |          70939 |          494 |         237638 |
| 2019-04 |          68078 |          376 |         202067 |
| 2019-05 |          72532 |          408 |         210549 |
| 2019-06 |          86064 |          354 |         178205 |
| 2019-07 |          86596 |          331 |         188702 |
| 2019-08 |          87359 |          382 |         203951 |
| 2019-09 |          70652 |          418 |         217675 |
| 2019-10 |          68758 |          454 |         257740 |
| 2019-11 |          70421 |          388 |         200307 |
| 2019-12 |         141060 |          592 |         335158 |
| 2020-01 |          52202 |          343 |         182595 |
| 2020-02 |          47791 |          336 |         183950 |
| 2020-03 |          72850 |          444 |         234090 |
| 2020-04 |          66892 |          302 |         152174 |
| 2020-05 |          74343 |          527 |         287226 |
| 2020-06 |          87805 |          467 |         253505 |
| 2020-07 |          85848 |          321 |         158669 |
| 2020-08 |          88759 |          415 |         208785 |
| 2020-09 |          69533 |          340 |         202701 |
| 2020-10 |          69348 |          384 |         195573 |
| 2020-11 |          72635 |          294 |         153182 |
| 2020-12 |         139538 |          258 |         141139 |

## Control results for these loads

| source_file    | control_type   | control_name                 | severity   | status   |     actual_value | details                                               |
|:---------------|:---------------|:-----------------------------|:-----------|:---------|-----------------:|:------------------------------------------------------|
| fraudTest.csv  | RECONCILIATION | distinct_trans_num           | CRITICAL   | PASS     | 555719           | fraudTest.csv: source=555719 bronze=555719            |
| fraudTest.csv  | RECONCILIATION | fraud_count                  | CRITICAL   | PASS     |   2145           | fraudTest.csv: source=2145 bronze=2145                |
| fraudTest.csv  | RECONCILIATION | row_count                    | CRITICAL   | PASS     | 555719           | fraudTest.csv: source=555719 bronze=555719            |
| fraudTest.csv  | RECONCILIATION | total_amount                 | CRITICAL   | PASS     |      3.85629e+07 | fraudTest.csv: source=38562903.11 bronze=38562903.11  |
| fraudTest.csv  | DATA_QUALITY   | amt_not_numeric              | CRITICAL   | PASS     |      0           | 0 of 555,719 records broke this rule                  |
| fraudTest.csv  | DATA_QUALITY   | amt_not_positive             | CRITICAL   | PASS     |      0           | 0 of 555,719 records broke this rule                  |
| fraudTest.csv  | DATA_QUALITY   | category_missing             | WARNING    | PASS     |      0           | 0 of 555,719 records broke this rule                  |
| fraudTest.csv  | DATA_QUALITY   | cc_num_missing               | CRITICAL   | PASS     |      0           | 0 of 555,719 records broke this rule                  |
| fraudTest.csv  | DATA_QUALITY   | customer_coords_out_of_range | WARNING    | PASS     |      0           | 0 of 555,719 records broke this rule                  |
| fraudTest.csv  | DATA_QUALITY   | is_fraud_invalid             | CRITICAL   | PASS     |      0           | 0 of 555,719 records broke this rule                  |
| fraudTest.csv  | DATA_QUALITY   | merchant_coords_out_of_range | WARNING    | PASS     |      0           | 0 of 555,719 records broke this rule                  |
| fraudTest.csv  | DATA_QUALITY   | timestamp_invalid            | CRITICAL   | PASS     |      0           | 0 of 555,719 records broke this rule                  |
| fraudTest.csv  | DATA_QUALITY   | trans_num_duplicate          | CRITICAL   | PASS     |      0           | 0 of 555,719 records broke this rule                  |
| fraudTest.csv  | DATA_QUALITY   | trans_num_missing            | CRITICAL   | PASS     |      0           | 0 of 555,719 records broke this rule                  |
| fraudTrain.csv | RECONCILIATION | distinct_trans_num           | CRITICAL   | PASS     |      1.29668e+06 | fraudTrain.csv: source=1296675 bronze=1296675         |
| fraudTrain.csv | RECONCILIATION | fraud_count                  | CRITICAL   | PASS     |   7506           | fraudTrain.csv: source=7506 bronze=7506               |
| fraudTrain.csv | RECONCILIATION | row_count                    | CRITICAL   | PASS     |      1.29668e+06 | fraudTrain.csv: source=1296675 bronze=1296675         |
| fraudTrain.csv | RECONCILIATION | total_amount                 | CRITICAL   | PASS     |      9.12224e+07 | fraudTrain.csv: source=91222428.90 bronze=91222428.90 |
| fraudTrain.csv | DATA_QUALITY   | amt_not_numeric              | CRITICAL   | PASS     |      0           | 0 of 1,296,675 records broke this rule                |
| fraudTrain.csv | DATA_QUALITY   | amt_not_positive             | CRITICAL   | PASS     |      0           | 0 of 1,296,675 records broke this rule                |
| fraudTrain.csv | DATA_QUALITY   | category_missing             | WARNING    | PASS     |      0           | 0 of 1,296,675 records broke this rule                |
| fraudTrain.csv | DATA_QUALITY   | cc_num_missing               | CRITICAL   | PASS     |      0           | 0 of 1,296,675 records broke this rule                |
| fraudTrain.csv | DATA_QUALITY   | customer_coords_out_of_range | WARNING    | PASS     |      0           | 0 of 1,296,675 records broke this rule                |
| fraudTrain.csv | DATA_QUALITY   | is_fraud_invalid             | CRITICAL   | PASS     |      0           | 0 of 1,296,675 records broke this rule                |
| fraudTrain.csv | DATA_QUALITY   | merchant_coords_out_of_range | WARNING    | PASS     |      0           | 0 of 1,296,675 records broke this rule                |
| fraudTrain.csv | DATA_QUALITY   | timestamp_invalid            | CRITICAL   | PASS     |      0           | 0 of 1,296,675 records broke this rule                |
| fraudTrain.csv | DATA_QUALITY   | trans_num_duplicate          | CRITICAL   | PASS     |      0           | 0 of 1,296,675 records broke this rule                |
| fraudTrain.csv | DATA_QUALITY   | trans_num_missing            | CRITICAL   | PASS     |      0           | 0 of 1,296,675 records broke this rule                |
