# Thesis-Models

For more information you could read my as of yet totally unfinished draft. I'm basically computing an equilibrium in an unorthodox physician-patient market, where patients get utility based on their "taste" for sick leaves and actual medical need, γ_i and κ_i respectively, they then compute their chance s_ij of visiting doctor j, out of which the doctor may compute her own expected patient utility and certificates granted, Q_j and X_j respectively.

The key input in all this is the vector of physicians' κ_j, the threshold "over" which they're willing to dole out certificates to patients (based on their medical need κi). I'm looking for an algorithm that will run succesive loops over κ_j until equilibrium is reached.

Once such an algorithm is secured, first for a Logit-like s_ij then for a different version based on Schnell (2024), the next step would be to look at the data we have on physicians in Chile using the PANDAS module, and run calibration exercises (probably a GMM-method with our model's moments). Once parameters are attained, we would run counterfactual exercises with those same algorithms. And that's the thesis, hopefully.
