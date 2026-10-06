/*
  Mart: fraud KPIs aggregated by merchant category.

  This is the primary table powering the "fraud by category" Power BI visual
  and the LLM weekly brief. Replaces the ad-hoc Spark SQL in notebook 04.
*/

with base as (
    select * from {{ ref('stg_transactions') }}
),

aggregated as (
    select
        category,
        count(*)                                            as total_transactions,
        sum(is_fraud)                                       as fraud_count,
        count(*) - sum(is_fraud)                            as legit_count,
        round(avg(is_fraud::numeric) * 100, 3)              as fraud_rate_pct,
        round(sum(case when is_fraud = 1 then amount else 0 end), 2)
                                                            as total_fraud_losses,
        round(avg(case when is_fraud = 1 then amount end), 2)
                                                            as avg_fraud_amount,
        round(avg(amount), 2)                               as avg_txn_amount,
        max(amount)                                         as max_txn_amount
    from base
    group by category
)

select
    category,
    total_transactions,
    fraud_count,
    legit_count,
    fraud_rate_pct,
    total_fraud_losses,
    avg_fraud_amount,
    avg_txn_amount,
    max_txn_amount,
    -- rank by loss so dashboards can easily show top N
    rank() over (order by total_fraud_losses desc)          as loss_rank,
    rank() over (order by fraud_rate_pct desc)              as rate_rank
from aggregated
order by total_fraud_losses desc
