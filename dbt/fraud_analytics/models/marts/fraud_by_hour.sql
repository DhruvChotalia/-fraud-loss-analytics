/*
  Mart: fraud KPIs aggregated by hour of day.

  Powers the "fraud by hour" Power BI visual and the anomaly alert in the
  LLM analyst. The 22:00-23:00 spike is the key finding from this table.
*/

with base as (
    select * from {{ ref('stg_transactions') }}
),

aggregated as (
    select
        txn_hour                                            as hour_of_day,
        count(*)                                            as total_transactions,
        sum(is_fraud)                                       as fraud_count,
        round(avg(is_fraud::numeric) * 100, 3)              as fraud_rate_pct,
        round(sum(case when is_fraud = 1 then amount else 0 end), 2)
                                                            as total_fraud_losses,
        round(avg(case when is_fraud = 1 then amount end), 2)
                                                            as avg_fraud_amount
    from base
    group by txn_hour
)

select
    hour_of_day,
    total_transactions,
    fraud_count,
    fraud_rate_pct,
    total_fraud_losses,
    avg_fraud_amount,
    -- label for Power BI
    lpad(hour_of_day::text, 2, '0') || ':00'               as hour_label,
    -- flag the high-risk window (22:00-23:00 from data profile)
    case when hour_of_day in (22, 23) then true else false end
                                                            as is_high_risk_hour
from aggregated
order by hour_of_day
