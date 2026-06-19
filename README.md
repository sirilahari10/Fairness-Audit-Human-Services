# Auditing the Void: Mitigating Algorithmic Bias in Human Services

In my day-to-day work building AI pipelines in healthcare, a model failure usually just means another log file error or a slightly skewed performance metric. But in the public sector, a false positive isn't just a lost data point—it’s an invasive caseworker visit. A false negative is a family denied housing assistance.

Standard machine learning metrics (like global accuracy) actively lie to us when structural bias exists in the training data.

I built this quick PoW to demonstrate how I approach algorithmic fairness and model auditing. It’s easy to train `LogisticRegression.fit()`. The actual engineering challenge is proving the model won't hurt people.

### The Experiment
1. **The Data:** I generated a synthetic dataset simulating 5,000 households applying for emergency rental assistance, predicting 90-day eviction risk.
2. **The Structural Bias:** I injected a proxy variable—`prior_system_involvement`—which is historically inflated for vulnerable neighborhoods due to over-policing or systemic over-documentation. 
3. **The Audit:** I trained a standard model. It achieved great overall accuracy. But when I audited the Disparate Impact, the False Positive Rate for vulnerable neighborhoods was exponentially higher than for well-resourced neighborhoods. The model was silently penalizing the vulnerable group.
4. **The Mitigation:** I wrote a custom intervention layer to enforce **Equalized Odds**. By adjusting the decision thresholds based on the underlying distribution skews, I equalized the False Positive Rate across demographics without destroying the model's predictive power.

### Why This Matters
In human services, we cannot just drop noisy columns like zip codes, because ML models will find latent proxies anyway. We have to mathematically audit the outputs and force fairness into the pipeline architecture. 
