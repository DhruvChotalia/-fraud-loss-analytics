/*
  Mart: monthly fraud loss trend.

  This is the time series that feeds the Holt-Winters forecast in notebook 06.
  Having it as a dbt model means the forecast notebook can read a clean,
  pre-aggregated table instead of re-doing the aggregation every run.
*/

with base as (
    select * from {{ ref('stg_transactions') }}
),

monthly as (
    select
        txn_year                                            as year,
        txn_month                                          as month,
        -- ISO year-month for easy sorting and Power BI date axis
        (txn_year::text || '-' || lpad(txn_month::text, 2, '0'))::text
                                                            as year_month,
        count(*)                                            as total_transactions,
        sum(is_fraud)                                       as fraud_count,
        round(avg(is_fraud::numeric) * 100, 3)              as fraud_rate_pct,
        round(sum(case when is_fraud = 1 then amount else 0 end), 2)
                                                            as total_fraud_losses,
        round(avg(case when is_fraud = 1 then amount end), 2)
                                                            as avg_fraud_amount
    from base
    group by txn_year, txn_month
)

select
    year,
    month,
    year_month,
    total_transactions,
    fraud_count,
    fraud_rate_pct,
    total_fraud_losses,
    avg_fraud_amount,
    -- month-over-month loss change
    round(
        total_fraud_losses - lag(total_fraud_losses) over (order by year, month),
        2
    )                                                       as mom_loss_change,
    -- rolling 3-month average (smooths seasonal noise)
    round(
        avg(total_fraud_losses) over (
            order by year, month
            rows between 2 preceding and current row
        ),
        2
    )                                                       as rolling_3m_avg_losses
from monthly
order by year, month
