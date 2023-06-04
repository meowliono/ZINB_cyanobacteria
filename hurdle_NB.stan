functions {
    int neg_binomial_2_safe_rng(real eta, real phi) {
      real gamma_rate = gamma_rng(phi, phi / eta);
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
  int<lower =0> N_test;
  matrix[N_test,K] x_test;
}

parameters{
  real intercept1;
  real intercept2;
  real<lower=0.000001> phi; 
  vector[K] beta_theta;
  vector[K] beta_mu;
}

transformed parameters{
  vector[N] theta;
  vector[N] mu;
  theta = intercept1 + x * beta_theta;
  mu = exp(intercept2 + x*beta_mu);
}

model{
  intercept1 ~ normal(0,1);
  intercept2 ~ normal(0,1);
  beta_theta ~ normal(0,1);
  phi ~ cauchy(0,1);
  beta_mu ~ normal(0,1);
  for (n in 1:N){
    if(y[n]==0)
      target += bernoulli_logit_lpmf(1|theta[n]);
    else
      target += bernoulli_logit_lpmf(0|theta[n])+neg_binomial_2_lpmf(y[n]|mu[n], phi)-neg_binomial_2_lccdf(0|mu[n],phi);
  }
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N){
    if(y[n]==0)
      log_lik[n] = bernoulli_logit_lpmf(1|theta[n]);
    else
      log_lik[n]= bernoulli_logit_lpmf(0|theta[n])+neg_binomial_2_lpmf(y[n]|mu[n], phi)-neg_binomial_2_lccdf(0|mu[n],phi);
  }
}

