# dpo_pipeline — LLM-Autobidding DPO Data Generation Pipeline
#
# Generates DPO (Direct Preference Optimization) training data for an
# LLM-based auto-bidding agent by running multi-probe simulations in
# the AuctionNet ad-auction environment under different macro scenarios.
#
# Pipeline: Scenario Config → Multi-Probe Simulation → Scoring → Preference Extraction → DPO JSONL
