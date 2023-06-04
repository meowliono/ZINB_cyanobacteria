functions {
    int neg_binomial_2_log_safe_rng(real eta, real phi) {
      real gamma_rate = gamma_rng(phi, phi / exp(eta));
      if (gamma_rate >= exp(20.79))
        return -9;      
      return poisson_rng(gamma_rate);
    }
    real zero_prop(int N, vector y){
      vector[N] zero;
      for (i in 1:N)
        if (y[i]==0)
          zero[i]=1;
        else
          zero[i]=0;
      return sum(zero)/N;
    }
}
data{
  int<lower = 0> N;
  int<lower = 0> K;
  matrix[N,K] x;
  int<lower =0> y[N];
  int<lower =0> N_test;
  vector[N_test] y1;
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
      target += log_sum_exp(bernoulli_lpmf(1|theta[n]), bernoulli_lpmf(0|theta[n])+neg_binomial_2_lpmf(0|mu[n],phi));
    else
      target += bernoulli_lpmf(0|theta[n])+neg_binomial_2_lpmf(y[n]|mu[n], phi);
  }
}

generated quantities {
  vector[N_test] y_pre;
  real indicator1;
  real indicator2;
  for (i in 1:N_test)
    if(bernoulli_rng(inv_logit(intercept1 + x_test[i,]*beta_theta))==1)
      y_pre[i]=0;
    else
      y_pre[i]=neg_binomial_2_log_safe_rng(intercept2 + x_test[i,]*beta_mu, phi);
  indicator1= zero_prop(N_test,y_pre)>zero_prop(N_test,y1);
  indicator2= mean(y_pre)>mean(y1);
  }
