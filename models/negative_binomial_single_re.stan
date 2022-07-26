data {
  int<lower=1> N;
  real A;
  int<lower=1> p;
  array[N] real depth;
  array[N] int y;
  matrix[N, p] x;
  real<lower=0> B_p;
  real<lower=0> disp_scale;

  // Random Effects
  int<lower=1> num_subjs;
  array[N] int<lower=1, upper=num_subjs> subj_map;
  real<lower=0> re_p;
}

parameters {
  real<offset=A, multiplier=3> beta_0;
  vector[p-1] beta_x;
  real<lower=0> reciprocal_phi;
  vector[num_subjs] subj_re;  // Subject effect on intercept
}

transformed parameters {
  vector[p] beta_var = append_row(beta_0, beta_x);
  vector[N] lam;
  real phi = inv(reciprocal_phi);

  for (n in 1:N) {
    lam[n] = depth[n];
    for (i in 1:p) {
      // when i = 1 -> Intercept (1)
      // Add subj effect
      lam[n] = lam[n] + (beta_var[i] + subj_re[subj_map[n]]) * x[n, i];
    }
  }
}

model {
  beta_0 ~ normal(A, 3);
  beta_x ~ normal(0, B_p);
  reciprocal_phi ~ lognormal(0, disp_scale);
  subj_re ~ normal(0, re_p);

  y ~ neg_binomial_2_log(lam, phi);
}

generated quantities {
  vector[N] y_predict;
  vector[N] log_lhood;

  for (n in 1:N) {
    y_predict[n] = neg_binomial_2_log_rng(lam[n], phi);
    log_lhood[n] = neg_binomial_2_log_lpmf(y[n] | lam[n], phi);
  }
}
