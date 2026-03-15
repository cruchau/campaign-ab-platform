# About the dataset

## Marketing A/B testing dataset

Marketing companies want to run successful campaigns, but the market is complex and several options can work. So normally they tun A/B tests, that is a randomized experimentation process wherein two or more versions of a variable (web page, page element, banner, etc.) are shown to different segments of people at the same time to determine which version leaves the maximum impact and drive business metrics.

The companies are interested in answering two questions:
- Would the campaign be successful?
- If the campaign was successful, how much of that success could be attributed to the ads?

With the second question in mind, we normally do an A/B test. The majority of the people will be exposed to ads (the experimental group). And a small portion of people (the control group) would instead see a Public Service Announcement (PSA) (or nothing) in the exact size and place the ad would normally be.

The idea of the dataset is to analyze the groups, find if the ads were successful, how much the company can make from the ads, and if the difference between the groups is statistically significant.

Data dictionary:
- Index: Row index
- user id: User ID (unique)
- test group: If "ad" the person saw the advertisement, if "psa" they only saw the public service announcement
- converted: If a person bought the product then True, else is False
- total ads: Amount of ads seen by person
- most ads day: Day that the person saw the biggest amount of ads
- most ads hour: Hour of day that the person saw the biggest amount of ads

---

**Ready for Layer 1 — dbt.**

Here's what dbt will do with these Parquet files. dbt is a transformation tool — it takes raw data and builds a clean, structured data warehouse on top of it using SQL. It doesn't move data or run infrastructure — it just runs SQL queries and materialises the results as tables.

Our dbt project will have three layers:
```
data/raw/           ← Parquet files (Layer 0 output)
    ↓
staging/            ← exact copy of raw, just typed and named correctly
    ↓
intermediate/       ← business logic: user-level metrics, exposure calculations
    ↓
marts/              ← final tables: fact_exposures, dim_campaigns, fct_ab_results
                       these are what the A/B engine and dashboard query