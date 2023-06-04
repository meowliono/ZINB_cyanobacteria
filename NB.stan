functions {
    int neg_binomial_2_log_safe_rng(real eta, real phi) {
      real gamma_rate = gamma_rng(phi, phi / exp(eta));
      if (gamma_rate >= exp(20.79))
        return -9;      
      return poisson_rng(gamma_rate);
    }
}
data{
  int<lower = 0> N;
  int<lower = 0> K;
  matrix[N,K] x;
  int<lower =0> y[N];
}
parameters{
  real intercept;
  real<lower=0> phi;
  vector[K] beta_mu;
}
transformed parameters{
  vector[N] mu;
  mu = exp(intercept + x*beta_mu);
}
model{
  //intercept ~ normal(0,5);
  //phi ~ cauchy(0,3);
  //beta_mu ~ normal(0,5);
  for (n in 1:N){
    target += neg_binomial_2_lpmf(y[n]|mu[n], phi);
  }
}
generated quantities {
  vector[N] log_lik;
  for (i in 1:N){
    log_lik[i] = neg_binomial_2_lpmf(y[i]|mu[i], phi);
    }
  }
