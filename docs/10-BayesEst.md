# Introduction to Bayesian Estimation

In this chapter, you will learn about the Bayesian approach to estimation by fitting regression models using the <code class='package'>brms</code> package [@burkner_brms_2017]. This is the most flexible approach to modelling as you can select your relevant outcome and predictors rather than relying on out-of-the-box statistical tests. We will be focusing on estimation and exploring the posterior of your model to make inferences. You will build on the skills you learnt in chapter 9, but extending it to more flexible priors and statistical models. We are mainly going to focus on simple and multiple linear regression in this chapter, but the final section outlines further resources to learn about more advanced distribution families and models.

You are always welcome to provide feedback on our resources, but this book is part of a new suite of materials we are developing. If you have any comments, please complete this <a href="https://forms.office.com/e/Wc18LDDSpF" target="_blank">online short anonymous form</a> or contact one of the lecturing team directly.

## Learning objectives

By the end of this chapter, you should be able to: 

1. Understand the steps involved in fitting and exploring Bayesian regression models. 

2. Apply these steps to simple linear regression using continuous and categorical predictors. 

3. Apply these steps to multiple linear regression and interactions. 

4. Create data visualisation to graphically communication the results of your Bayesian regression models. 

To follow along to this chapter and try the code yourself, please download the data files we will be using in [this zip file](data/10_data.zip).

In this chapter, we need a few extra packages. The one most likely to cause trouble is the main <code class='package'>brms</code> package since it uses Stan and you need a C++ compiler. See the [installing R appendix](#installing-r) for guidance. If you are really struggling or its very slow on your computer, <code class='package'>brms</code> is available on the R Studio server. See the course overview page for a link if you have never used it before. 


```r
library(brms) # fitting Bayesian models
library(bayestestR) # helper functions for plotting and understanding the models
library(tidybayes) # helper functions for combining plotting and tidy data from models
library(tidyverse)
library(see) # helper functions for plotting objects from bayestestR
library(emmeans) # Handy function for calculating (marginal) effect sizes
```

## Simple Linear Regression 

### Guided example (Schroeder & Epley, 2015)

For this guided activity, we will use data from the study by @schroeder_sound_2015. We used this in the chapter 9 for the independent activity, so we will explore the data set as the guided example in this chapter to see how we can refit it as a Bayesian regression model. 

As a reminder, the aim of the study was to investigate whether delivering a short speech to a potential employer would be more effective at landing you a job than writing the speech down and the employer reading it themselves. Thirty-nine professional recruiters were randomly assigned to receive a job application speech as either a transcript for them to read or an audio recording of them reading the speech. 

The recruiters then rated the applicants on perceived intellect, their impression of the applicant, and whether they would recommend hiring the candidate. All ratings were originally on a Likert scale ranging from 0 (low intellect, impression etc.) to 10 (high impression, recommendation etc.), with the final value representing the mean across several items. 

For this example, we will focus on the hire rating (variable <code><span class='st'>"Hire_Rating"</span></code>) to see whether the audio condition would lead to higher ratings than the transcript condition (variable <code><span class='st'>"CONDITION"</span></code>). 

Remember the key steps of Bayesian modelling from lecture 10 [@heino_bayesian_2018]:

1. Identify data relevant to the research question 

2. Define a descriptive model, whose parameters capture the research question

3. Specify prior probability distributions on parameters in the model 

4. Update the prior to a posterior distribution using Bayesian inference 

5. Check your model against data, and identify potential problems

#### Identify data

For this example, we have the data from Schroeder and Epley with one outcome and one categorical predictor. The data are coded 0 for those in the transcript group and 1 for those in the audio group. 


```r
Schroeder_data <- read_csv("data/Schroeder_hiring.csv")
```

#### Define a descriptive model

The next step is to define a descriptive model. In chapter 9, we used the <code class='package'>BayesFactor</code> package to use out-of-the-box tests like a t-test, but we saw in the lecture with the <a href="https://lindeloev.github.io/tests-as-linear/" target="_blank">Lindelöv (2019) blog post</a>, common statistical models are just different expressions of linear models. So, we can express the same t-test as a linear model, using <code><span class='st'>"CONDITION"</span></code> as a single categorical predictor of <code><span class='st'>"Hire_Rating"</span></code> as our outcome. You can enter this directly in the <code><span class='fu'>brm</span><span class='op'>(</span><span class='op'>)</span></code> function below, but its normally a good idea to clearly outline each component.  


```r
Schroeder_model1 <- bf(Hire_Rating ~ CONDITION)
```

#### Specify prior probability of parameters

Once you get used to the <code class='package'>brms</code> package, you start to learn which priors you need for simple cases, but now we have stated a model, we can see which parameters can be assigned a prior. 


```r
get_prior(Schroeder_model1, # Model we defined above
          data = Schroeder_data) # Which data frame are we using? 
```

<div class="kable-table">

<table>
 <thead>
  <tr>
   <th style="text-align:left;"> prior </th>
   <th style="text-align:left;"> class </th>
   <th style="text-align:left;"> coef </th>
   <th style="text-align:left;"> group </th>
   <th style="text-align:left;"> resp </th>
   <th style="text-align:left;"> dpar </th>
   <th style="text-align:left;"> nlpar </th>
   <th style="text-align:left;"> bound </th>
   <th style="text-align:left;"> source </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> b </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> default </td>
  </tr>
  <tr>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> b </td>
   <td style="text-align:left;"> CONDITION </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> default </td>
  </tr>
  <tr>
   <td style="text-align:left;"> student_t(3, 4, 3) </td>
   <td style="text-align:left;"> Intercept </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> default </td>
  </tr>
  <tr>
   <td style="text-align:left;"> student_t(3, 0, 3) </td>
   <td style="text-align:left;"> sigma </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> default </td>
  </tr>
</tbody>
</table>

</div>

This tells us which priors we can set and what the default settings are. We have the prior, the class of prior, relevant coefficients, and the source which will all be default for now. The prior tells you what the default is. For example, there are flat uninformative priors on coefficients. When we set priors, we can either set priors for a whole class, or specific to each coefficient. With one predictor, there is only one coefficient prior to set, so it makes no difference. But when you have multiple predictors like later in chapter 10, it becomes more useful. 

Coefficients are assigned flat priors, meaning anything is possible between minus infinity and infinity. We can visualise the priors to see what they expect one-by-one. You will see how you can plot the priors yourself shortly. 

<img src="10-BayesEst_files/figure-html/default flat prior-1.png" width="100%" style="display: block; margin: auto;" />

The intercept and sigma are assigned student t distributions for priors. These are both quite weak priors to have minimal influence on the model, but they do not factor in your knowledge about the parameters. The default prior for the intercept peaks slightly above 0 and most likely between -5 and 15. 

<img src="10-BayesEst_files/figure-html/plot default intercept prior-1.png" width="100%" style="display: block; margin: auto;" />

The default prior for sigma peaks at 0 and most likely between -10 and 10. Just keep in mind sigma as the standard deviation cannot be smaller than 0, so this definitely does not factor in parameter knowledge.   

<img src="10-BayesEst_files/figure-html/plot default sigma prior-1.png" width="100%" style="display: block; margin: auto;" />

For our example, we can define our own informative priors using information from Schroeder and Epley. Their paper contains four studies and our data set focuses on the fourth where they apply their findings to professional recruiters. Study 1 preceded this and used students, so we can pretend we are the researchers and use this as a source of our priors for the "later" study. 

Focusing on hire rating, they found: 

> "Evaluators who heard pitches also reported being significantly more likely to hire the candidates (*M* = 4.34, *SD* = 2.26) than did evaluators who read exactly the same pitches (*M* = 3.06, *SD* = 3.15), *t*(156) = 2.49, *p* = .01, 95% CI of the difference = [0.22, 2.34], *d* = 0.40 (see Fig. 1)". 

So, for our intercept and reference group, we can set a normally distributed prior around a mean of 3 and SD of 3 for the transcript group. Note the rounded values since these are approximations for what we expect about the measures and manipulations. We are factoring in what we know about the parameters from our topic and method knowledge.  

It is normally a good idea to visualise this process to check the numbers you enter match your expectations. For the intercept, a mean and SD of 3 look like this when generating the numbers from a normal distribution:


```r
prior <- c(prior(normal(3, 3), class = Intercept)) # Set prior and class

prior %>% 
  parse_dist() %>% 
  ggplot(aes(y = 0, dist = .dist, args = .args, fill = prior)) +
  stat_slab(normalize = "panels") +
  scale_fill_viridis_d(option = "plasma", end = 0.9) +
  guides(fill = "none") +
  labs(x = "Value", y = "Density", title = paste0(prior$class, ": ", prior$prior)) +
  theme_classic()
```

<img src="10-BayesEst_files/figure-html/plot SE intercept prior-1.png" width="100%" style="display: block; margin: auto;" />

This turns out to be quite a weak prior since the distribution extends below 0 (which is not possible for this scale) all the way to 10 which is the upper limit of this scale. It covers pretty much the entire measurement scale with the peak around 3, so it represents a lenient estimate of what we expect the reference group to be.

We can set something more informative for the sigma prior knowing what we do about standard deviations. A common prior for the standard deviation is using an exponential distribution as it cannot be lower than 0. This means the largest density is around zero and the density decreases across more positive values. Values closer to zero cover a wider range, while larger values cover a smaller range. 


```r
prior <- c(prior(exponential(1), class = sigma)) # Set prior and class

prior %>% 
  parse_dist() %>% 
  ggplot(aes(y = 0, dist = .dist, args = .args, fill = prior)) +
  stat_slab(normalize = "panels") +
  scale_fill_viridis_d(option = "plasma", end = 0.9) +
  guides(fill = "none") +
  labs(x = "Value", y = "Density", title = paste0(prior$class, ": ", prior$prior)) +
  theme_classic()
```

<img src="10-BayesEst_files/figure-html/user sigma prior-1.png" width="100%" style="display: block; margin: auto;" />

**Note on the visualisation**: Credit to the visualisation method goes to Andrew Heiss who shared some <a href="https://gist.github.com/andrewheiss/a4e0c0ab2d735625ac17ec8a081f0f32" target="_blank">code on a Github Gist</a> to visualise different priors. I adapted the code to use here to help you visualise the priors you enter. You can adapt the code to show any kind of prior used in brms models. All you need to do is specify the distribution family and parameters. Like the original code, you can even present a bunch of options to compare side by side. 

For the coefficient, the mean difference was around 1 (calculated manually by subtracting one mean from the other) and the 95% confidence interval was quite wide from 0.22 to 2.34. As we are working out what prior would best fit our knowledge, we can compare some different options side by side. We can compare a stronger prior (*SD* = 0.5) vs a weaker prior (*SD* = 1). 


```r
priors <- c(prior(normal(1, 0.5), class = b),
            prior(normal(1, 1), class = b)) # Set prior and class

priors %>% 
  parse_dist() %>% # Function from tidybayes/ggdist to turn prior into a dataframe
  ggplot(aes(y = 0, dist = .dist, args = .args, fill = prior)) + # Fill in details from prior and add fill
  stat_slab(normalize = "panels") + # ggdist layer to visualise distributions
  scale_fill_viridis_d(option = "plasma", end = 0.9) + # Add colour scheme
  guides(fill = "none") + # Remove legend for fill
  facet_wrap(~prior) + # Split into a different panel for each prior
  labs(x = "Value", y = "Density") +
  theme_classic()
```

<img src="10-BayesEst_files/figure-html/plot coefficient priors-1.png" width="100%" style="display: block; margin: auto;" />

The stronger prior on the left shows we are expecting mainly positive effects with a peak over 1 but ranges between around -0.5 (transcript to be higher than audio) and 2 (audio to be higher than transcript). The weaker prior on the right shows we are still expecting the peak over 1, but it could span from -1.5 to around 3.5. 

Lets say we think both positive and negatives effects are plausible but we expect the most likely outcome to be similar to study 1 from Schroeder and Epley. So, for this example we will go with the weaker prior. Now we have our priors, we can save them to a new object:


```r
priors <- set_prior("normal(1, 1)", class = "b") + 
  set_prior("normal(3, 3)", class = "Intercept") + 
  set_prior("exponential(1)", class = "sigma")
```

::: {.info data-latex=""}
Remember it is important to check the sensitivity of the results to the choice of prior. So, once we're finished, we will check how stable the results are to an uninformative prior, keeping the defaults. Normally it is the opposite way around and using uninformative priors first, but I did not want to put off thinking about the priors. 
:::

#### Update the prior to the posterior

This is going to be the longest section as we are going to fit the `brms` model and then explore the posterior. 

As the process relies on sampling using MCMC, it is important to set a seed within the function for reproducibility, so the semi-random numbers have a consistent starting point. This might take a while depending on your computer, then you will get a bunch of output for fitting the model and sampling from the MCMC chains. 


```r
Schroeder_fit <- brm(
  formula = Schroeder_model1, # formula we defined above 
  data = Schroeder_data, # Data frame we're using 
  family = gaussian(), # What distribution family do we want for the likelihood function? Many examples we use in psychology are Gaussian, but check the documentation for options
  prior = priors, # priors we stated above
  sample_prior = TRUE, # Setting this to true includes the prior in the object, so we can include it on plots later
  seed = 1908,
  file = "Models/Schroeder_model1" #Save the model as a .rds file
)
```

::: {.info data-latex=""}
When you have lots of data or complicated models, the fitting process can take a long time. This means its normally a good idea to save your fitted model to save time if you want to look at it again quickly. In the brm function, there is an argument called `file`. You write a character string for any further file directory and the name you want to save it as. Models are saved as a .rds file - R's own data file format you can save objects in. Behind the scenes for this book, we must run the code every time we want to update it, so all the models you see will be based on reading the models as .rds files after we first fitted the models. If you save the objects, remember to refit them if you change anything like the priors, model, or data. If the file already exists though, it will not be overwritten unless you use the `file_refit` argument. 
:::

If you save the model as a .rds file, you can load them again using the <code><span class='fu'>read_rds</span><span class='op'>(</span><span class='op'>)</span></code> function from <code class='package'>readr</code> in the tidyverse. 


```r
Schroeder_fit <- read_rds("Models/Schroeder_model1.rds")
```

There will be a lot of output here to explain the fitting and sampling process. For a longer explanation of how MCMC sampling works, see @van_ravenzwaaij_simple_2018, but for a quick overview, we want to sample from the posterior distribution based on the data and model. The default of `brms` is to sample from four chains, with each chain containing 2000 iterations (1000 of which are warm up / burn in iterations). If you get warning messages about model fit or convergence issues, you can increase the number of iterations. This becomes more important with more complex models, so all the defaults should be fine for the relatively simple models we fit in this chapter. We will return to chains and convergence when we see the trace plots later. 

Now we have fitted the model, we can also double check the priors you set are what you wanted. You will see the source for the priors you set switched from default to user. 


```r
prior_summary(Schroeder_fit)
```

<div class="kable-table">

<table>
 <thead>
  <tr>
   <th style="text-align:left;"> prior </th>
   <th style="text-align:left;"> class </th>
   <th style="text-align:left;"> coef </th>
   <th style="text-align:left;"> group </th>
   <th style="text-align:left;"> resp </th>
   <th style="text-align:left;"> dpar </th>
   <th style="text-align:left;"> nlpar </th>
   <th style="text-align:left;"> bound </th>
   <th style="text-align:left;"> source </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> normal(1, 1) </td>
   <td style="text-align:left;"> b </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> user </td>
  </tr>
  <tr>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> b </td>
   <td style="text-align:left;"> CONDITION </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> default </td>
  </tr>
  <tr>
   <td style="text-align:left;"> normal(3, 3) </td>
   <td style="text-align:left;"> Intercept </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> user </td>
  </tr>
  <tr>
   <td style="text-align:left;"> exponential(1) </td>
   <td style="text-align:left;"> sigma </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> user </td>
  </tr>
</tbody>
</table>

</div>

Now we have our model, we can get a model summary like any old linear model in R. 


```r
summary(Schroeder_fit)
```

```
##  Family: gaussian 
##   Links: mu = identity; sigma = identity 
## Formula: Hire_Rating ~ CONDITION 
##    Data: Schroeder_data (Number of observations: 39) 
##   Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
##          total post-warmup draws = 4000
## 
## Population-Level Effects: 
##           Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## Intercept     3.01      0.47     2.09     3.94 1.00     3402     2862
## CONDITION     1.57      0.57     0.46     2.66 1.00     3449     2879
## 
## Family Specific Parameters: 
##       Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## sigma     2.17      0.25     1.74     2.71 1.00     3617     2850
## 
## Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
## and Tail_ESS are effective sample size measures, and Rhat is the potential
## scale reduction factor on split chains (at convergence, Rhat = 1).
```

At the top, we have information on the model fitting process, like the family, data, and draws from the posterior summarising the chain iterations. 

Population-level effects is our main area of interest. This is where we have the posterior probability distribution summary statistics. We will look at the whole distribution soon, but for now, we can see the median point-estimate for the intercept is 3.01 with a 95% credible interval between 2.10 and 3.96. This is what we expect the mean of the reference group to be, i.e., the transcript group. 

We then have the median coefficient of 1.57 with a 95% credible interval between 0.46 and 2.66. This means our best guess for the mean difference / slope is an increase of 1.57 for the audio group. Note, you might get subtly different values to the output here since it is based on a semi-random sampling process, but the qualitative conclusions should be the same. 

For convergence issues, if Rhat is different from 1, it can suggest there are problems with the model fitting process. You can also look at the effective sample size statistics (the columns ending in ESS). These should at least be in the hundreds [@flores_beforeafter_2022] for both the bulk and tail to ensure the model has worked around the parameter space and has not missed key features of the posterior distribution. We will return to a final indicator of model fitting soon when we check the trace plots. 

For a tidier summary of the parameters, we can also use the handy <code><span class='fu'>describe_posterior</span><span class='op'>(</span><span class='op'>)</span></code> function from <code class='package'>bayestestR</code>. 


```r
describe_posterior(Schroeder_fit)
```

<div class="kable-table">

<table>
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:left;"> Parameter </th>
   <th style="text-align:right;"> Median </th>
   <th style="text-align:right;"> CI </th>
   <th style="text-align:right;"> CI_low </th>
   <th style="text-align:right;"> CI_high </th>
   <th style="text-align:right;"> pd </th>
   <th style="text-align:right;"> ROPE_CI </th>
   <th style="text-align:right;"> ROPE_low </th>
   <th style="text-align:right;"> ROPE_high </th>
   <th style="text-align:right;"> ROPE_Percentage </th>
   <th style="text-align:right;"> Rhat </th>
   <th style="text-align:right;"> ESS </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> b_Intercept </td>
   <td style="text-align:right;"> 3.006929 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> 2.0932902 </td>
   <td style="text-align:right;"> 3.942043 </td>
   <td style="text-align:right;"> 1.00000 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -0.1 </td>
   <td style="text-align:right;"> 0.1 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0.9999234 </td>
   <td style="text-align:right;"> 3382.849 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> b_CONDITION </td>
   <td style="text-align:right;"> 1.567213 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> 0.4562219 </td>
   <td style="text-align:right;"> 2.657129 </td>
   <td style="text-align:right;"> 0.99625 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -0.1 </td>
   <td style="text-align:right;"> 0.1 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0.9999724 </td>
   <td style="text-align:right;"> 3426.619 </td>
  </tr>
</tbody>
</table>

</div>

We can use this as a way to create ROPE regions for the effects and it tells us useful things like the probability of direction for the effect (how much of the posterior is above or below zero). 

This will be more useful when it comes to comparing models and building multiple regression models, but there is also a specific function to get the model $R^2$ and its 95% credible interval. This tells you the proportion of variance in your outcome that your predictor(s) explain, which is 12.6% here. 


```r
bayes_R2(Schroeder_fit)
```

```
##     Estimate  Est.Error       Q2.5     Q97.5
## R2 0.1262071 0.07090139 0.01058625 0.2746032
```

Until now, we have focused on point-estimates and intervals of the posterior, but the main strength of Bayesian statistics is summarising the parameters as a whole posterior probability distribution, so we will now turn to the various plotting options. 

The first plot is useful for seeing the posterior of each parameter and the trace plots to check on any convergence issues. 


```r
plot(Schroeder_fit)
```

<img src="10-BayesEst_files/figure-html/Schroeder parameters and trace-1.png" width="100%" style="display: block; margin: auto;" />

For this model, we have three plots: one for the intercept, one for the coefficient/slope, and one for sigma. On the left, we have the posterior probability distributions for each. On the right, we have trace plots. By default, `brms` uses four chains - or series of samples using MCMC - and this shows how each chain moves around the parameter space. Essentially, we want the trace plots to look like fuzzy caterpillars with a random series of lines. If there are spike which deviate massively from the rest, or the lines get stuck in one area, this suggests there are convergence issues. 

These plots are useful for an initial feel of the parameter posteriors, but there are a great series of functions from the <code class='package'>bayestestR</code> package [@Makowski2019] which you can use on their own, or wrap them in the <code><span class='fu'><a target='_blank' href='https://rdrr.io/r/graphics/plot.default.html'>plot</a></span><span class='op'>(</span><span class='op'>)</span></code> function after loading the <code class='package'>see</code> package [@Luedecke2021]. For example, we can see an overlay of the prior and posterior for the main parameters of interest. On its own, <code><span class='fu'>p_direction</span><span class='op'>(</span><span class='op'>)</span></code> tells you the probability of direction for each parameter, i.e., how much of the distribution is above or below 0? Wrapped in <code><span class='fu'><a target='_blank' href='https://rdrr.io/r/graphics/plot.default.html'>plot</a></span><span class='op'>(</span><span class='op'>)</span></code>, you can see the prior and posterior, with the posterior divided in areas above or below 0. 


```r
plot(p_direction(Schroeder_fit), 
     priors = TRUE) 
```

<img src="10-BayesEst_files/figure-html/Schroeder p direction-1.png" width="100%" style="display: block; margin: auto;" />

::: {.warning data-latex=""}
For this to work, you must specify priors in `brms`. It does not work with the package default options for the coefficients. 
:::

We can see the pretty wide prior in blue, then the posterior. Almost all of the posterior distribution is above zero to show we're pretty confident that audio is associated with higher hire ratings than transcript. 

The next useful plot is seeing the 95% HDI / credible interval. On its own, <code><span class='fu'>hdi</span><span class='op'>(</span><span class='op'>)</span></code> will show you the 95% HDI for your parameters. Wrapped in <code><span class='fu'><a target='_blank' href='https://rdrr.io/r/graphics/plot.default.html'>plot</a></span><span class='op'>(</span><span class='op'>)</span></code>, you can visualise the HDI compared to zero for your main parameters. If the HDI excludes zero, you can be confident in a positive or negative effect, at least conditional on these data and model. Remember, there is a difference between the small world and big world of models. This is not the absolute truth, just the most credible values conditioned on our data and model. 


```r
plot(bayestestR::hdi(Schroeder_fit)) # Specify package to avoid clash with ggdist
```

<img src="10-BayesEst_files/figure-html/Schroeder HDI-1.png" width="100%" style="display: block; margin: auto;" />

::: {.warning data-latex=""}
These plots are informative for you learning about your model and the inferences you can learn from it. However, they would not be immediately suitable to enter into a report. Fortunately, they are created using <code class='package'>ggplot</code>, so you can customise them in the same way by adding layers of additional functions. 
:::

For this example, the 95% HDI excludes 0, so we can be confident the coefficient posterior is a positive effect, with the audio group leading to higher hire ratings than the transcript group. 

Finally, we might not be interested in comparing the coefficients to a point-value of 0, we might have a stronger level of evidence in mind, where the coefficient must exclude a range of values in the ROPE process we explored in chapter 9. For example, maybe effects smaller than 1 unit difference are too small to be practically/theoretically meaningful. 

::: {.info data-latex=""}
Remember this is potentially the most difficult decision to make, maybe more so than choosing priors. Many areas of psychology do not have clear guidelines/expectations for smallest effect sizes of interest, so it is down to you to explain and justify your approach based on your understanding of the topic area.
:::


```r
plot(rope(Schroeder_fit, 
          range = c(-1, 1))) # What is the ROPE range for your smallest effects of interest? 
```

<img src="10-BayesEst_files/figure-html/Schroeder ROPE-1.png" width="100%" style="display: block; margin: auto;" />

For this example, for a sample size of 39, we have pretty strong evidence in favour of a positive effect in the audio group. The 95% HDI excludes zero, but if we set a ROPE of 1 unit, we do not quite exclude it. This means if we wanted to be more confident that the effect exceeded the ROPE, we would need more data. This is just for demonstration purposes, I'm not sure if the original study would consider an effect of 1 as practically meaningful, or whether they would just be happy with any non-zero effect.

Following from chapter 9, we saw we can also use Bayesian statistics to test hypotheses. This works in a modelling approach as `brms` has a function to test hypotheses. We must provide the fitted model object and state a hypothesis to test. This relies on a character description of the parameter and test value. For a full explanation, see the <a href="https://paul-buerkner.github.io/brms/reference/hypothesis.html" target="_blank">brms documentation online</a> for the function. Here, we will test the coefficient/slope against a point-null of 0. 


```r
hypothesis(Schroeder_fit, # brms model we fitted earlier
           hypothesis = "CONDITION = 0") 
```

```
## Hypothesis Tests for class b:
##        Hypothesis Estimate Est.Error CI.Lower CI.Upper Evid.Ratio Post.Prob
## 1 (CONDITION) = 0     1.57      0.57     0.46     2.66       0.08      0.08
##   Star
## 1    *
## ---
## 'CI': 90%-CI for one-sided and 95%-CI for two-sided hypotheses.
## '*': For one-sided hypotheses, the posterior probability exceeds 95%;
## for two-sided hypotheses, the value tested against lies outside the 95%-CI.
## Posterior probabilities of point hypotheses assume equal prior probabilities.
```

::: {.info data-latex=""}
We must state a character hypothesis which requires you to select a parameter. Here, we focus on the <code><span class='st'>"CONDITIONAudio"</span></code> parameter, i.e., our slope, which must match the name in the model. We can then state values to test against, like here against a point-null of 0 for a Bayes factor. Alternatively, you can test posterior odds where you compare masses of the posterior like CONDITIONAudio > 0.
:::

The key part of the output is the evidence ratio, but we also have the estimate and 95% credible interval. As we are testing a point-null of 0, we are testing the null hypothesis against the alternative of a non-null effect. As the value is below 1, it suggests we have evidence in favour of the alternative compared to the null. I prefer to express things above 1 as its easier to interpret. You can do this by dividing 1 by the ratio, which should provide a Bayes factor of 9.09 here. 

Alternatively, you can calculate the posterior odds by stating regions of the posterior to test. For example, if we used "CONDITIONAudio > 0", this would provide a ratio of the posterior probability of positive effects above 0 to the posterior probability of negative effects below 0. For this example, this would be a posterior odds of 209 in favour of positive effects. Note, when all the posterior is above 0, you can get a result of Inf (infinity) as all the evidence is in favour of positive effects.


```r
hypothesis(Schroeder_fit, # brms model we fitted earlier
           hypothesis = "CONDITION > 0") 
```

```
## Hypothesis Tests for class b:
##        Hypothesis Estimate Est.Error CI.Lower CI.Upper Evid.Ratio Post.Prob
## 1 (CONDITION) > 0     1.57      0.57     0.63      2.5     265.67         1
##   Star
## 1    *
## ---
## 'CI': 90%-CI for one-sided and 95%-CI for two-sided hypotheses.
## '*': For one-sided hypotheses, the posterior probability exceeds 95%;
## for two-sided hypotheses, the value tested against lies outside the 95%-CI.
## Posterior probabilities of point hypotheses assume equal prior probabilities.
```

Calculating / plotting conditional effects

- Show how to use emmeans with model 

- Show conditional effects with plot: http://paul-buerkner.github.io/brms/reference/conditional_effects.html

#### Model checking 

Finally, we have our model checking procedure. We already looked at some information for this such as Rhat and the trace plots. This suggests the model fitted OK. We also want to check the model reflects the properties of the data. This does not mean we want it exactly the same and overfit to the data, but it should follow a similar pattern to show our model captures the features of the data. 

Bayesian models are generative, which means once they are fitted, we can use them to sample values from the posterior and make predictions from it. One key process is called a posterior predictive check which takes the model and uses is to generate new samples. This shows how you have conditioned the model and what it expects. 

The plot below is a <code class='package'>brms</code> function for facilitating this. The thick blue line is your data for the outcome. The light blue lines are 100 samples from the posterior to show what the model expects about the outcome. 


```r
pp_check(Schroeder_fit, 
         ndraws = 100) # How many draws from the posterior? Higher values means more lines
```

<img src="10-BayesEst_files/figure-html/Schroeder model check-1.png" width="100%" style="display: block; margin: auto;" />

For this example, it does an OK job at capturing the pattern of data and the bulk of the observed data follows the generated curves. However, you can see the data are quite flat compared to the predicted values. As we expect a Gaussian distribution, the model will happily produce normal curves. The model also happily expects values beyond the range of data as our scale is bound to 0 and 10. This is hugely common in psychological research as we expect Gaussian distributions from ordinal bound data. So, while this model does an OK job, we could potentially improve it by focusing on an ordinal regression model so we can factor in the bounded nature of the measure. 

::: {.try data-latex=""}
If you want to challenge yourself, I recommend working through [Bürkner and Vuorre (2019)](https://doi.org/10.1177/2515245918823199) and applying your understanding to this task. This is going to be a common theme in the examples you see in the independent activities as psychology articles (myself included) often use metric models to analyse arguably ordinal data.
:::

The final thing we will check for this model is how sensitive it is to the choice of prior. A justifiable informative prior is a key strength of Bayesian statistics, but it is important to check the model under at least two sets of priors. For this example, we will compare the model output under the default package priors and our user defined priors we used all along.

In the code below, we have omitted the prior argument, so we are fitting the exact same model as before but using the default package priors. 


```r
Schroeder_fit2 <- brm(
  formula = Schroeder_model1,
  data = Schroeder_data, 
  family = gaussian(),
  seed = 1908,
  file = "Models/Schroeder_model2" #Save the model as a .rds file
)
```



If we run the <code><span class='fu'><a target='_blank' href='https://rdrr.io/r/base/summary.html'>summary</a></span><span class='op'>(</span><span class='op'>)</span></code> function again, you can check the intercept and predictor coefficients to see how they differ to the first model we fitted. Ideally, they should provide us with similar inferences, such as a similar magnitude and in the same direction. It is never going to be exactly the same under different priors, but we want our conclusions robust to the choice of prior we use. 


```r
summary(Schroeder_fit2)
```

```
##  Family: gaussian 
##   Links: mu = identity; sigma = identity 
## Formula: Hire_Rating ~ CONDITION 
##    Data: Schroeder_data (Number of observations: 39) 
##   Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
##          total post-warmup draws = 4000
## 
## Population-Level Effects: 
##                Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## Intercept          2.90      0.52     1.89     3.93 1.00     3369     2457
## CONDITIONAudio     1.82      0.73     0.33     3.24 1.00     3578     2469
## 
## Family Specific Parameters: 
##       Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## sigma     2.22      0.27     1.77     2.83 1.00     3446     2868
## 
## Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
## and Tail_ESS are effective sample size measures, and Rhat is the potential
## scale reduction factor on split chains (at convergence, Rhat = 1).
```

To make it easier to compare, we can isolate the key information from each model and present them side by side. You can see below how there is little difference in the intercept between both models. The median is similar, both probability of direction values are 100%, and the 95% HDI ranges across similar values. For our user prior, the coefficient is a little more conservative, but the difference is also small here, showing how our results are robust to the choice of prior. 

<div class="kable-table">

<table>
 <thead>
  <tr>
   <th style="text-align:left;"> Model </th>
   <th style="text-align:left;"> Parameter </th>
   <th style="text-align:right;"> Median </th>
   <th style="text-align:right;"> CI_low </th>
   <th style="text-align:right;"> CI_high </th>
   <th style="text-align:right;"> pd </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> User prior </td>
   <td style="text-align:left;"> b_Intercept </td>
   <td style="text-align:right;"> 3.006929 </td>
   <td style="text-align:right;"> 2.0932902 </td>
   <td style="text-align:right;"> 3.942043 </td>
   <td style="text-align:right;"> 1.00000 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> User prior </td>
   <td style="text-align:left;"> b_CONDITION </td>
   <td style="text-align:right;"> 1.567213 </td>
   <td style="text-align:right;"> 0.4562219 </td>
   <td style="text-align:right;"> 2.657129 </td>
   <td style="text-align:right;"> 0.99625 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Default prior </td>
   <td style="text-align:left;"> b_Intercept </td>
   <td style="text-align:right;"> 2.903251 </td>
   <td style="text-align:right;"> 1.8855461 </td>
   <td style="text-align:right;"> 3.930462 </td>
   <td style="text-align:right;"> 1.00000 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Default prior </td>
   <td style="text-align:left;"> b_CONDITIONAudio </td>
   <td style="text-align:right;"> 1.838562 </td>
   <td style="text-align:right;"> 0.3318538 </td>
   <td style="text-align:right;"> 3.237537 </td>
   <td style="text-align:right;"> 0.98900 </td>
  </tr>
</tbody>
</table>

</div>

### Independent activity (Brandt et al., 2014)

For an independent activity, we will use data from the study by [@brandt_does_2014]. The aim of Brandt et al. was to replicate a relatively famous social psychology study on the effect of recalling unethical behaviour on the perception of brightness. 

In common language, unethical behaviour is considered as "dark", so the original authors designed a priming experiment where participants were randomly allocated to recall an unethical behaviour or an ethical behaviour from their past. Participants then completed a series of measures including their perception of how bright the testing room was. Brandt et al. were sceptical and wanted to replicate this study to see if they could find similar results. 

Participants were randomly allocated (<code><span class='st'>"ExpCond"</span></code>) to recall an unethical behaviour (n = 49) or an ethical behaviour (n = 51). The key outcome is their perception of how bright the room was (<code><span class='st'>"WellLitSca"</span></code>), from 1 (not bright at all) to 7 (very bright). The research question was: Does recalling unethical behaviour lead people to perceive a room as darker than if they recall ethical behaviour? 

Use your understanding of the design to address the research question. If you follow the link to Brandt et al. above, the means and standard deviations of the original study are included in Table 2. This might be useful for thinking about your priors, but keep in mind how sensitive your conclusions are to your choice of prior. 


```r
Brandt_data <- read_csv("data/Brandt_unlit.csv")

# Recode to dummy coding 
Brandt_data <- Brandt_data %>% 
  dplyr::mutate(ExpCond = dplyr::case_when(ExpCond == 1 ~ 0,
                             ExpCond == -1 ~ 1))

# Relabel condition to be more intuitive which group is which 
# Ethical is the reference group
Brandt_data$ExpCond <- factor(Brandt_data$ExpCond, 
                                   levels = c(0, 1), 
                                   labels = c("Ethical", "Unethical"))
```

::: {.try data-latex=""}
From here, apply what you learnt in the first guided example to this new independent activity. 
:::


```r
Brandt_model1 <- NULL
```

## Multiple Linear Regression 

### Guided example (Heino et al., 2018)

#### Identify data

- Replace with Troy et al. replication from Aimi and Rhonda or Catherine's religion self-forgiveness. 

For the second guided example we covered in the lecture, we will explore the model included in @heino_bayesian_2018 for their Bayesian data analysis tutorial. They explored the feasibility and acceptability of the ”Let’s Move It” intervention to increase physical activity in 43 older adolescents. 

They randomised participants into two groups (<code><span class='st'>"intervention"</span></code>) for control (0) and intervention (1) arms (group sessions on motivation and self-regulation skills, and teacher training). Their outcome was a measure of autonomous motivation (<code><span class='st'>"value"</span></code>) on a 1-5 scale, with higher values meaning greater motivation. They measured the outcome at both baseline (0) and six weeks after (1; <code><span class='st'>"time"</span></code>).

Their research question was: To what extent does the intervention affect autonomous motivation? 


```r
Heino_data <- read_csv("data/Heino-2018.csv") %>% 
  group_by(ID, intervention, time) %>% 
  summarise(value = mean(value, na.rm = TRUE)) %>% 
  ungroup()
```

::: {.info data-latex=""}
Part of their tutorial discusses a bigger multilevel model considering different scenarios, but for this demonstration, we're just averaging over the scenarios to get the mean motivation.
:::

#### Define a descriptive model

I recommend reading the article as they explain this process in more detail. We essentially have an outcome of autonomous motivation (<code><span class='st'>"value"</span></code>) and we want to look at the interaction between <code><span class='st'>"intervention"</span></code> and <code><span class='st'>"time"</span></code>. They define a fixed intercept in the model with the `1 +` part. Its also technically a multi-level model as they define a random intercept for each participant (`(1 | ID)`) to ensure we recognise time is within-subjects. 

::: {.info data-latex=""}
By default, R includes a fixed intercept (the `1 +` part) in the model, so you would get the same results without adding it to the model. However, people often include it so it is explicit in the model formula.
:::


```r
Heino_model <- bf(value ~ 1 + time * intervention + (1 | ID))
```

#### Specify prior probability of parameters

Compared to simple linear regression, as you add predictors, the number of priors you can set also increase. In the output below, you will see how you can enter a prior for all beta coefficients or one specific for each predictors. There are also different options for setting a prior for standard deviations and sigma. 


```r
get_prior(Heino_model, data = Heino_data)
```

```
## Warning: Rows containing NAs were excluded from the model.
```

<div class="kable-table">

<table>
 <thead>
  <tr>
   <th style="text-align:left;"> prior </th>
   <th style="text-align:left;"> class </th>
   <th style="text-align:left;"> coef </th>
   <th style="text-align:left;"> group </th>
   <th style="text-align:left;"> resp </th>
   <th style="text-align:left;"> dpar </th>
   <th style="text-align:left;"> nlpar </th>
   <th style="text-align:left;"> bound </th>
   <th style="text-align:left;"> source </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> b </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> default </td>
  </tr>
  <tr>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> b </td>
   <td style="text-align:left;"> intervention </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> default </td>
  </tr>
  <tr>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> b </td>
   <td style="text-align:left;"> time </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> default </td>
  </tr>
  <tr>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> b </td>
   <td style="text-align:left;"> time:intervention </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> default </td>
  </tr>
  <tr>
   <td style="text-align:left;"> student_t(3, 3.9, 2.5) </td>
   <td style="text-align:left;"> Intercept </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> default </td>
  </tr>
  <tr>
   <td style="text-align:left;"> student_t(3, 0, 2.5) </td>
   <td style="text-align:left;"> sd </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> default </td>
  </tr>
  <tr>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> sd </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> ID </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> default </td>
  </tr>
  <tr>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> sd </td>
   <td style="text-align:left;"> Intercept </td>
   <td style="text-align:left;"> ID </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> default </td>
  </tr>
  <tr>
   <td style="text-align:left;"> student_t(3, 0, 2.5) </td>
   <td style="text-align:left;"> sigma </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> default </td>
  </tr>
</tbody>
</table>

</div>

Note, you get a warning about missing data but since its a multi-level model, we just have fewer observations in some conditions instead of the whole case being removed. 

This is another place where I recommend reading the original article for more information. They discuss their choices and essentially settle on wide weak priors for the coefficients to say small effects are more likely but they allow larger effects. The two standard deviation classes are then assigned relatively wide Cauchy priors to only allow positive values. 


```r
Heino_priors <- prior(normal(0, 5), class = "b") +
  prior(cauchy(0, 1), class = "sd") +
  prior(cauchy(0, 2), class = "sigma")
```

#### Update prior to posterior

This is going to be the longest section as we are going to fit the `brms` model and then explore the posterior. 

As the process relies on sampling using MCMC, it is important to set a seed for reproducibility, so the semi-random numbers have a consistent starting point. This might take a while depending on your computer, then you will get a bunch of output for fitting the model and sampling from the MCMC chains. 


```r
Heino_fit <- brm(
  formula = Heino_model,
  data = Heino_data,
  prior = Heino_priors,
  family = gaussian(),
  seed = 2108,
  file = "Models/Heino_model"
)
```



Now we have fitted the model, let's have a look at the summary. 


```r
summary(Heino_fit)
```

```
##  Family: gaussian 
##   Links: mu = identity; sigma = identity 
## Formula: value ~ 1 + time * intervention + (1 | ID) 
##    Data: Heino_data (Number of observations: 68) 
##   Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
##          total post-warmup draws = 4000
## 
## Group-Level Effects: 
## ~ID (Number of levels: 40) 
##               Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## sd(Intercept)     0.71      0.10     0.53     0.92 1.00      817     1710
## 
## Population-Level Effects: 
##                   Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## Intercept             3.70      0.20     3.29     4.09 1.00      810     1328
## time                  0.08      0.14    -0.19     0.37 1.00     2170     2694
## intervention         -0.08      0.26    -0.59     0.43 1.00      762     1446
## time:intervention     0.10      0.18    -0.25     0.44 1.00     2203     2552
## 
## Family Specific Parameters: 
##       Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## sigma     0.33      0.05     0.25     0.44 1.00     1107     1982
## 
## Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
## and Tail_ESS are effective sample size measures, and Rhat is the potential
## scale reduction factor on split chains (at convergence, Rhat = 1).
```

The model summary is very similar to the examples in the simple linear regression section, but we also have a new section for group-level effects since we added a random intercept for participants.

Exploring the coefficients, all the effects are pretty small, with the largest effect being 0.10 units. There is quite a bit of uncertainty here, with 95% credible intervals spanning negative and positive effects. 

This is a start, but particularly in more complicated models like this, plotting is going to be your best friend for understanding what is going on. 


```r
plot(Heino_fit)
```

<img src="10-BayesEst_files/figure-html/Heino plots-1.png" width="100%" style="display: block; margin: auto;" /><img src="10-BayesEst_files/figure-html/Heino plots-2.png" width="100%" style="display: block; margin: auto;" />

```r
plot(p_direction(Heino_fit), 
     priors = TRUE) # plot the priors
```

<img src="10-BayesEst_files/figure-html/Heino plots-3.png" width="100%" style="display: block; margin: auto;" />

```r
plot(bayestestR::hdi(Heino_fit)) # Specify to avoid clash
```

<img src="10-BayesEst_files/figure-html/Heino plots-4.png" width="100%" style="display: block; margin: auto;" />

Regardless of the output we look at here, there is not much going on across any of the predictors. The data comes from a feasibility study, so the sample size was pretty small and its mainly about how receptive participants are to the intervention. 

As a bonus extra, you can also use the <code class='package'>emmeans</code> package [@Lenth2022] to calculate marginal effects on the posterior distribution. Its not important here as there is little we can learn from breaking down the interaction further, but it might come in handy in future. 


```r
# Surround with brackets to both save and output
(Heino_means <- emmeans(Heino_fit, # add the model object  
        ~ time | intervention)) # We want to separate time by levels of intervention
```

```
## intervention = 0:
##  time emmean lower.HPD upper.HPD
##     0   3.69      3.32      4.11
##     1   3.78      3.39      4.18
## 
## intervention = 1:
##  time emmean lower.HPD upper.HPD
##     0   3.61      3.27      3.91
##     1   3.79      3.46      4.12
## 
## Point estimate displayed: median 
## HPD interval probability: 0.95
```

This provides the median value of the posterior for the combination of time and intervention. Here, we can see pretty clearly there is not much going on, with very little difference across the estimates and all the 95% credible intervals overlapping. 

Depending on how you want to express the marginal means, you can also use the <code class='package'>emmeans</code> object to calculate contrasts, expressing the effects as mean differences in the posterior for each group/condition.  


```r
contrast(Heino_means)
```

```
## intervention = 0:
##  contrast estimate lower.HPD upper.HPD
##  0 effect  -0.0418   -0.1802    0.0957
##  1 effect   0.0418   -0.0957    0.1802
## 
## intervention = 1:
##  contrast estimate lower.HPD upper.HPD
##  0 effect  -0.0909   -0.1919    0.0216
##  1 effect   0.0909   -0.0216    0.1919
## 
## Point estimate displayed: median 
## HPD interval probability: 0.95
```

#### Model check

As the final step, we can look at the posterior predictive check to make sure the model is capturing the features of the data. Compared to the first guided example, the model maps onto the data quite well, with the samples largely following the underlying data. We are still using metric models to analyse ultimately ordinal data (despite calculating the mean response), so the expected values go beyond the range of data (1-5).


```r
pp_check(Heino_fit,
         ndraws = 100) # 100 draws from the model
```

<img src="10-BayesEst_files/figure-html/Heino pp check-1.png" width="100%" style="display: block; margin: auto;" />

::: {.try data-latex=""}
If you scroll to the end of the Heino et al. article, they demonstrate how you can fit an ordinal model to the data. 
:::

### Independent activity (Coleman et al., 2014)

For an independent activity, we will use data from the study by @coleman_absorption_2019. Coleman et al. contains two studies investigating religious mystical experiences. One study focused on undergraduates and a second study focused on experienced meditators who were part of a unique religious group.  

The data set contains a range of variables used for the full model in the paper. We are going to focus on a small part of it for this exercise, but feel free to explore developing the full model as was used in study 1. The key variables are: 

1. <code><span class='st'>"Age"</span></code> - Measured in years

2. <code><span class='st'>"Gender"</span></code> - 0 = male; 1 = female

3. <code><span class='st'>"Week_med"</span></code> - Ordinal measure of how often people meditate per week, with higher values meaning more often

4. <code><span class='st'>"Time_session"</span></code> - Ordinal measure of how long people meditate per session, with higher values meaning longer

5. <code><span class='st'>"Absorption_SUM"</span></code> - Sum score of the Modified Tellegen Absorption scale, with higher values meaning greater trait levels of imaginative engagement 

6. <code><span class='st'>"EQ_SUM"</span></code> - Sum score of the Empathizing Quotient short form, with higher values meaning greater theory of mind ability 

7. <code><span class='st'>"Mscale_SUM"</span></code> - Sum score of the Hood M-scale, with higher values meaning more self-reported mystical experiences

Previous studies had explored these components separately and mainly in undergraduates, so Coleman et al. took the opportunity to explore a unique sample of a highly committed religious group. The final model included all seven variables, but for this example, we will just focus on absorption (<code><span class='st'>"Absorption_SUM"</span></code>) and mentalizing (<code><span class='st'>"EQ_SUM"</span></code>) as they were the main contributors, with the other variables as covariates.

Our research question is: How does absorption (<code><span class='st'>"Absorption_SUM"</span></code>) and mentalizing (<code><span class='st'>"EQ_SUM"</span></code>) affect mystical experiences (<code><span class='st'>"Mscale_SUM"</span></code>) as an outcome? Focus on entering the two variables as individual predictors at first, then explore an interaction. 

Use your understanding of the design to address the research question. If you follow the link to Coleman et al. above, you can see the results of study 2 which focused on undergraduate students. This study is presented second, but you can use it for this example to develop your understanding of the measures for your priors. Think about whether you have weaker or stronger priors depending on your understanding of the topic, but keep in mind how sensitive your conclusions are to your choice of prior. 


```r
Coleman_data <- read_csv("data/Coleman_2019.csv")
```

::: {.try data-latex=""}
From here, apply what you learnt in the first guided example to this new independent task. 
:::


```r
Coleman_model <- NULL
```

## Summary 

## Taking this further

## Independent activity solutions 

### Brandt et al. (2014)

### Coleman et al. (2019)
