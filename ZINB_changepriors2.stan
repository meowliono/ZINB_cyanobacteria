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
  int<lower =0> N_test;
  matrix[N_test,K] x_test;
}

parameters{
  real intercept1;
  real intercept2;
  real<lower=0> phi; 
  vector[K] beta_theta;
  vector[K] beta_mu;
}

transformed parameters{
  vector[N] theta;
  vector[N] mu;
  theta = inv_logit(intercept1 + x * beta_theta);
  mu = intercept2 + x*beta_mu;
}

model{
  intercept1 ~ uniform(-10,10);
  intercept2 ~ uniform(-10,10);
  beta_theta ~ uniform(-10,10);
  phi ~ uniform(0,1);
  beta_mu ~ uniform(-10,10);
  for (n in 1:N){
    if(y[n]==0)
      target += log_sum_exp(bernoulli_lpmf(1|theta[n]), bernoulli_lpmf(0|theta[n])+neg_binomial_2_log_lpmf(0|mu[n],phi));
    else
      target += bernoulli_lpmf(0|theta[n])+neg_binomial_2_log_lpmf(y[n]|mu[n], phi);
  }
}

generated quantities {
  vector[N_test] y_pre;
  for (i in 1:N_test)
    if(bernoulli_rng(inv_logit(intercept1 + x_test[i,]*beta_theta))==1)
      y_pre[i]=0;
    else
      y_pre[i]=neg_binomial_2_log_safe_rng(intercept2 + x_test[i,]*beta_mu, phi);
  }