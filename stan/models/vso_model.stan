data {
  int<lower=1> NS;                      // number of subjects
  int<lower=1> MT;                      // maximum number of valid trials
  array[NS] int<lower=1, upper=MT> ST;  // number of valid trials per subject
  array[NS, MT] real rew;               // observed reward outcomes
  array[NS, MT] real pun;               // observed punishment outcomes
  array[NS, MT] int<lower=-1, upper=2> choice;  // selected choices (Conflict = 2, Safe = 1, NA = -1)
  array[NS, MT, 2] real p_rew;          // Possible rewards for each choice 
  array[NS, MT, 2] real p_pun;          // Possible punishments for each choice
}

transformed data {
  vector[2] initQ;
  initQ = rep_vector(0.5, 2);           // Initial Q-values
}

parameters {
  vector[6] mu_pr;                      // Group-level mean parameters
  vector<lower=0>[6] sigma;             // Group-level standard deviations
  
  // Subject-level raw parameters (for Matt trick)
  vector[NS] alpha_rew_pos_pr;
  vector[NS] alpha_rew_neg_pr;
  vector[NS] alpha_pun_pos_pr;
  vector[NS] alpha_pun_neg_pr;
  vector[NS] beta_rew_pr;
  vector[NS] beta_pun_pr;
}

transformed parameters {
  // Transformed subject-level parameters
  vector<lower=0, upper=1>[NS] alpha_rew_pos;
  vector<lower=0, upper=1>[NS] alpha_rew_neg;
  vector<lower=0, upper=1>[NS] alpha_pun_pos;
  vector<lower=0, upper=1>[NS] alpha_pun_neg;
  vector<lower=0, upper=10>[NS] beta_rew;
  vector<lower=0, upper=10>[NS] beta_pun;
  
  for (i in 1:NS) {
    alpha_rew_pos[i] = Phi_approx(mu_pr[1] + sigma[1] * alpha_rew_pos_pr[i]);
    alpha_rew_neg[i] = Phi_approx(mu_pr[2] + sigma[2] * alpha_rew_neg_pr[i]);
    alpha_pun_pos[i] = Phi_approx(mu_pr[3] + sigma[3] * alpha_pun_pos_pr[i]);
    alpha_pun_neg[i] = Phi_approx(mu_pr[4] + sigma[4] * alpha_pun_neg_pr[i]);
    beta_rew[i] = Phi_approx(mu_pr[5] + sigma[5] * beta_rew_pr[i]) * 10;
    beta_pun[i] = Phi_approx(mu_pr[6] + sigma[6] * beta_pun_pr[i]) * 10;
  }
}

model {
  // Priors for group-level parameters
  mu_pr ~ normal(0, 1);
  sigma ~ normal(0, 0.2);
  
  // Priors for individual differences
  alpha_rew_pos_pr ~ normal(0, 1.0);
  alpha_rew_neg_pr ~ normal(0, 1.0);
  alpha_pun_pos_pr ~ normal(0, 1.0);
  alpha_pun_neg_pr ~ normal(0, 1.0);
  beta_rew_pr ~ normal(0, 1.0);
  beta_pun_pr ~ normal(0, 1.0);
  
  // Likelihood
  for (i in 1:NS) {
    vector[2] Qr;
    vector[2] Qp;
    vector[2] Qsum;
    
    real PEr;
    real PEp;
    
    Qr = initQ;
    Qp = initQ;
    Qsum = initQ;
    
    for (t in 1:ST[i]) {
      if (choice[i, t] > 0) {
        choice[i, t] ~ categorical_logit(Qsum);
      }
      
      PEr = rew[i, t] - Qr[choice[i, t]];
      PEp = pun[i, t] - Qp[choice[i, t]];
      
      Qr[choice[i, t]] += (PEr > 0) ? alpha_rew_pos[i] * PEr : alpha_rew_neg[i] * PEr;
      Qp[choice[i, t]] += (PEp > 0) ? alpha_pun_pos[i] * PEp : alpha_pun_neg[i] * PEp;
      
      Qsum = beta_rew[i] * Qr - beta_pun[i] * Qp; 
    }
  }
}

generated quantities {
  // Group-level parameter means
  real<lower=0, upper=1> mu_alpha_rew_pos;
  real<lower=0, upper=1> mu_alpha_rew_neg;
  real<lower=0, upper=1> mu_alpha_pun_pos;
  real<lower=0, upper=1> mu_alpha_pun_neg;
  real<lower=0, upper=10> mu_beta_rew;
  real<lower=0, upper=10> mu_beta_pun;
  
  // Log likelihood and predictions
  array[NS] real log_lik;
  array[NS, MT] int y_pred;  
  y_pred = rep_array(-1, NS, MT);
  array[NS, MT] real pred_rew;  
  array[NS, MT] real pred_pun;  
  
  // Initialize all to -1 (invalid/missing)
  pred_rew = rep_array(-1.0, NS, MT);
  pred_pun = rep_array(-1.0, NS, MT);
  
  // Transform group-level means
  mu_alpha_rew_pos = Phi_approx(mu_pr[1]);
  mu_alpha_rew_neg = Phi_approx(mu_pr[2]);
  mu_alpha_pun_pos = Phi_approx(mu_pr[3]);
  mu_alpha_pun_neg = Phi_approx(mu_pr[4]);
  mu_beta_rew = Phi_approx(mu_pr[5]) * 10;
  mu_beta_pun = Phi_approx(mu_pr[6]) * 10;
  
  {
    for (i in 1:NS) {
      vector[2] Qr_obs;
      vector[2] Qp_obs;
      vector[2] Qsum_obs;
      
      vector[2] Qr_pred;
      vector[2] Qp_pred;
      vector[2] Qsum_pred;
      
      real PEr_obs;
      real PEp_obs;
      real PEr_pred;
      real PEp_pred;
      int pred_choice;
      
      Qr_obs = initQ;
      Qp_obs = initQ;
      Qsum_obs = initQ;
      
      Qr_pred = initQ;
      Qp_pred = initQ;
      Qsum_pred = initQ;
      
      log_lik[i] = 0.0;
      
      for (t in 1:ST[i]) {
        if (choice[i, t] > 0) {  
        log_lik[i] += categorical_logit_lpmf(choice[i, t] | Qsum_obs);
        }
        
        pred_choice = categorical_rng(softmax(Qsum_pred));
        y_pred[i, t] = pred_choice;
        
        if (choice[i, t] > 0) {  
        PEr_obs = rew[i, t] - Qr_obs[choice[i, t]];
        PEp_obs = pun[i, t] - Qp_obs[choice[i, t]];
        
        Qr_obs[choice[i, t]] += (PEr_obs > 0) ? alpha_rew_pos[i] * PEr_obs : alpha_rew_neg[i] * PEr_obs;
        Qp_obs[choice[i, t]] += (PEp_obs > 0) ? alpha_pun_pos[i] * PEp_obs : alpha_pun_neg[i] * PEp_obs;
        Qsum_obs = beta_rew[i] * Qr_obs - beta_pun[i] * Qp_obs;
        }
        
        if (pred_choice > 0) {
          pred_rew[i, t] = p_rew[i, t, pred_choice];
          pred_pun[i, t] = p_pun[i, t, pred_choice];
        }
        
        PEr_pred = p_rew[i, t, pred_choice] - Qr_pred[pred_choice];
        PEp_pred = p_pun[i, t, pred_choice] - Qp_pred[pred_choice];
        
        Qr_pred[pred_choice] += (PEr_pred > 0) ? alpha_rew_pos[i] * PEr_pred : alpha_rew_neg[i] * PEr_pred;
        Qp_pred[pred_choice] += (PEp_pred > 0) ? alpha_pun_pos[i] * PEp_pred : alpha_pun_neg[i] * PEp_pred;
        Qsum_pred = beta_rew[i] * Qr_pred - beta_pun[i] * Qp_pred;
      }
    }
  }
}
