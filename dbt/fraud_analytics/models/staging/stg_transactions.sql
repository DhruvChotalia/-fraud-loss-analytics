/*
  Staging model: clean, typed view of bronze.transactions_raw.

  Why staging exists: bronze stores everything as TEXT (by design — no silent
  type coercion on load). Staging is the first place we cast to proper types,
  apply masks, and rename columns to a consistent standard. Everything
  downstream reads from staging, never from bronze directly.
*/

with source as (
    select * from bronze.transactions_raw
),

typed as (
    select
        trans_num                                           as transaction_id,
        trans_date_trans_time::timestamp                    as txn_time,
        cc_num                                              as card_number_masked,
        merchant,
        category,
        amt::numeric(12, 2)                                 as amount,
        gender,
        city,
        state,
        zip,
        lat::numeric(9, 6)                                  as customer_lat,
        long::numeric(9, 6)                                 as customer_long,
        city_pop::integer                                   as city_population,
        dob::date                                           as date_of_birth,
        merch_lat::numeric(9, 6)                            as merchant_lat,
        merch_long::numeric(9, 6)                           as merchant_long,
        is_fraud::integer                                   as is_fraud,

        -- derived columns
        extract(hour from trans_date_trans_time::timestamp) as txn_hour,
        extract(dow  from trans_date_trans_time::timestamp) as txn_dow,
        extract(month from trans_date_trans_time::timestamp) as txn_month,
        extract(year  from trans_date_trans_time::timestamp) as txn_year,
        case when extract(dow from trans_date_trans_time::timestamp) in (0, 6)
             then 1 else 0 end                              as is_weekend,

        -- geo distance (degrees, same formula as the Spark model)
        sqrt(
            power(merch_lat::numeric - lat::numeric, 2) +
            power(merch_long::numeric - long::numeric, 2)
        )                                                   as geo_distance,

        -- lineage passthrough
        _run_id,
        _source_file,
        _loaded_at

    from source
    where
        amt ~ '^[0-9]+(\.[0-9]+)?$'          -- only numeric amounts
        and amt::numeric > 0                  -- only positive amounts
        and is_fraud in ('0', '1')            -- only valid fraud flags
        and trans_num is not null             -- only identified transactions
)

select * from typed
