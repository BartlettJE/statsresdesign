# Introduction to Bayesian Estimation

In this chapter...

## Learning objectives

1. 

To follow along to this chapter and try the code yourself, please download the data files we will be using in [this zip file](data/10_data.zip).

In this chapter, we need a few packages for the things we will cover. There is a separate appendix section (add) we have prepared to help install the <code class='package'>brms</code> package as it can sometimes be pretty awkward since it uses Stan and you need a C++ compiler. If you are really struggling or its very slow on your computer, brms is available on the R Studio server. See the course overview page for a link if you have never used it before. 


```r
library(brms) #fitting Bayesian models
library(bayestestR) #helper functions for plotting and understanding the models
library(tidyverse)
library(see) #helper functions for plotting objects from bayestestR
library(emmeans) #Handy function for calculating (marginal) effect sizes
```

## Simple Linear Regression 

### Guided example (Schroeder & Epley, 2015)

For this guided activity, we will use data from the study by [Schroeder and Epley (2015)](https://journals.sagepub.com/stoken/default+domain/PhtK6MPtXvkgnYRrnGbA/full). We used this in the chapter 9 for the independent activity, so we will explore the data set as the guided example in this chapter to see how we can refit it as a Bayesian regression model. 

As a reminder, the aim of the study was to investigate whether delivering a short speech to a potential employer would be more effective at landing you a job than writing the speech down and the employer reading it themselves. Thirty-nine professional recruiters were randomly assigned to receive a job application speech as either a transcript for them to read or an audio recording of them reading the speech. 

The recruiters then rated the applicants on perceived intellect, their impression of the applicant, and whether they would recommend hiring the candidate. All ratings were originally on a Likert scale ranging from 0 (low intellect, impression etc.) to 10 (high impression, recommendation etc.), with the final value representing the mean across several items. 

For this example, we will focus on the hire rating (variable <code><span><span class='st'>"Hire_Rating"</span></span></code>) to see whether the audio condition would lead to higher ratings than the transcript condition (variable <code><span><span class='st'>"CONDITION"</span></span></code>). 

Remember the key steps of Bayesian modelling from lecture 10 (Heino et al., 2018):

1. Identify data relevant to the research question 

2. Define a descriptive model, whose parameters capture the research question

3. Specify prior probability distributions on parameters in the model 

4. Update the prior to a posterior distribution using Bayesian inference 

5. Check your model against data, and identify potential problems

#### 1. Identify data

For this example, we have the data from Schroeder and Epley, and we can label the conditions to be more intuitive. 


```r
Schroeder_data <- read_csv("data/Schroeder_hiring.csv")

# Relabel condition to be more intuitive which group is which 
Schroeder_data$CONDITION <- factor(Schroeder_data$CONDITION, 
                                   levels = c(0, 1), 
                                   labels = c("Transcript", "Audio"))
```

#### 2. Define a descriptive model

The next step is to define a descriptive model. In chapter 9, we used the <code class='package'>BayesFactor</code> package to use out-of-the-box tests like a t-test, but we saw in the lecture with the [Lindeloev (2019) blog post](https://lindeloev.github.io/tests-as-linear/), common statistical models are just different expressions of linear models. So, we can express the same t-test as a linear model, using <code><span><span class='st'>"CONDITION"</span></span></code> as a single categorical predictor of <code><span><span class='st'>"Hire_Rating"</span></span></code> as our outcome. You can enter this directly in the <code><span><span class='fu'>brm</span><span class='op'>(</span><span class='op'>)</span></span></code> function below, but its normally a good idea to clearly outline each component.  


```r
Schroeder_model1 <- bf(Hire_Rating ~ CONDITION)
```

#### 3. Specify prior probability of parameters

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
   <td style="text-align:left;"> CONDITIONAudio </td>
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

The intercept and sigma are assigned student t distributions for priors. These are both pretty wide to be weak priors. 

For our example, we will use information from Schroeder and Epley. Their paper contains four studies and our data set focuses on the fourth where they apply their findings to professional recruiters. Study 1 preceded this and used students, so we can pretend we are the researchers and use this as a source of our priors for the "later" study. 

Focusing on hire rating, they found: "Evaluators who heard pitches also reported being significantly more likely to hire the candidates (*M* = 4.34, *SD* = 2.26) than did evaluators who read exactly the same pitches (*M* = 3.06, *SD* = 3.15), *t*(156) = 2.49, *p* = .01, 95% CI of the difference = [0.22, 2.34], *d* = 0.40 (see Fig. 1)". 

So, for our intercept and reference group, we can set a normally distributed prior around a mean of 3 and SD of 3 for the transcript group. Note the rounded values since these are approximations for what we expect about the measures and manipulations. 

It is normally a good idea to visualise this process to check the numbers you enter match your expectations. For the intercept, a mean and SD of 3 look like this when generating the numbers from a normal distribution:


```r
set.seed(1928) # set seed to be reproducible

hist(rnorm(100, # how many samples?
           3, # What mean? 
           3)) # What SD?
```

<img src="10-BayesEst_files/figure-html/plot prior 1-1.png" width="100%" style="display: block; margin: auto;" />

This turns out to be quite a weak prior since the distribution extends below 0 (which is not possible for this scale) all the way to 10 which is the upper limit of this scale. It covers pretty much the entire measurement scale with the peak around 3, so it represents a conservative estimate of what we expect the reference group to be. 

For the coefficient, the mean difference was around 1 (calculated manually by subtracting one mean from the other) and the 95% CI was quite wide from 0.22 to 2.34, so we can set a relatively weak prior expecting a normally distributed coefficient with a mean and SD of 1:.  


```r
set.seed(1928)

hist(rnorm(100, 
           1, 
           1))
```

<img src="10-BayesEst_files/figure-html/plot prior 2-1.png" width="100%" style="display: block; margin: auto;" />

The distribution here shows we are expecting the most likely value for the coefficient to peak around 1, but it could span from -1 (transcript to be higher than audio) to around 3 (audio to be much higher than transcript). 

Now we have our priors, we can save them to a new object:


```r
prior <- set_prior("normal(1, 1)", class = "b") + 
  set_prior("normal(3, 3)", class = "Intercept")
```

::: {.info data-latex=""}
Remember it is important to check the sensitivity of the results to the choice of prior. So, once we're finished, we will check how stable the results are to an uninformative prior, keeping the defaults.
:::

#### 4. Update the prior to the posterior

This is going to be the longest section as we are going to fit the `brms` model and then explore the posterior. 

As the process relies on sampling using MCMC, it is important to set a seed within the function for reproducibility, so the semi-random numbers have a consistent starting point. This might take a while depending on your computer, then you will get a bunch of output for fitting the model and sampling from the MCMC chains. 


```r
Schroeder_fit <- brm(
  formula = Schroeder_model1, # formula we defined above 
  data = Schroeder_data, # Data frame we're using 
  family = gaussian(), # What distribution family do we want for the likelihood function? Many examples we use in psychology are Gaussian, but check the documentation for options
  prior = prior, # priors we stated above
  sample_prior = TRUE, # Setting this to true includes the prior in the object, so we can include it on plots later
  seed = 1908,
  file = "Models/Schroeder_model1" #Save the model as a .rds file
)
```

::: {.info data-latex=""}
When you have lots of data or complicated models, the fitting process can take a long time. This means its normally a good idea to save your fitted model to save time if you want to look at it again quickly. In the code below, there is an argument called `file`. You write a character string for any further file directory and the name you want to save it as. Models are saved as a .rds file - R's own data file format you can save objects in. Behind the scenes for this book, we must run the code every time we want to update it, so all the models you see will be based on reading the models as .rds files after we first fitted the models. If you save the objectives, remember to refit them if you change anything like the priors, model, or data. 
:::

If you save the model as a .rds file, you can load them again using the <code><span><span class='fu'>read_rds</span><span class='op'>(</span><span class='op'>)</span></span></code> function from <code class='package'>readr</code> in the <code class='package'>tidyverse</code>. 


```r
Schroeder_fit <- read_rds("Models/Schroeder_model1.rds")
```

There will be a lot of output here to explain the fitting and sampling process. The lecture references includes longer explanations of how MCMC sampling works, but for a quick overview, we want to sample from the posterior distribution based on the data and model. The default of `brms` is to sample from four chains, with each chain containing 2000 iterations (1000 of which are warm up / burn in iterations). If you get warning messages about model fit or convergence issues, you can increase the number of iterations. This becomes more important with more complex models, so all the defaults should be fine for the relatively simple models we fit in this chapter. We will return to chains and convergence when we see the trace plots later. 

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
   <td style="text-align:left;"> CONDITIONAudio </td>
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

Now we have our model, we can get a model summary like any old linear model. 


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
##                Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## Intercept          3.01      0.47     2.10     3.96 1.00     3021     2438
## CONDITIONAudio     1.56      0.58     0.40     2.68 1.00     3234     2810
## 
## Family Specific Parameters: 
##       Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
## sigma     2.22      0.27     1.77     2.82 1.00     3744     3065
## 
## Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
## and Tail_ESS are effective sample size measures, and Rhat is the potential
## scale reduction factor on split chains (at convergence, Rhat = 1).
```

At the top, we have information on the model fitting process, like the family, data, and draws from the posterior summarising the chain iterations. 

Population-level effects is our main area of interest. This is where we have the posterior probability distribution summary statistics. We will look at the whole distribution soon, but for now, we can see the median point-estimate for the intercept is 3.01 with a 95% credible interval between 2.10 and 3.96. This is what we expect the mean of the reference group to be, i.e., the transcript group. 

We then have the median coefficient of 1.56 with a 95% credible interval between 0.40 and 2.68. This means our best guess for the mean difference / slope is an increase of 1.56 for the audio group. Note, you might get subtly different values to the output here since it is based on a semi-random sampling process, but the main conclusions should be the same. 

For convergence issues, if Rhat is different from 1, it can suggest there are problems with the model fitting process. You can also look at the effective sample size statistics (the columns ending in ESS), but we did not explore this in the lecture. 

For a tidier summary of the parameters, we can also use the handy <code><span><span class='fu'>describe_posterior</span><span class='op'>(</span><span class='op'>)</span></span></code> function from <code class='package'>bayestestR</code>. 


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
   <td style="text-align:right;"> 3.002274 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> 2.1383382 </td>
   <td style="text-align:right;"> 3.989010 </td>
   <td style="text-align:right;"> 1.00000 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -0.2330343 </td>
   <td style="text-align:right;"> 0.2330343 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1.0002836 </td>
   <td style="text-align:right;"> 2999.233 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> b_CONDITIONAudio </td>
   <td style="text-align:right;"> 1.563360 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> 0.4333066 </td>
   <td style="text-align:right;"> 2.692349 </td>
   <td style="text-align:right;"> 0.99525 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -0.2330343 </td>
   <td style="text-align:right;"> 0.2330343 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0.9993928 </td>
   <td style="text-align:right;"> 3173.220 </td>
  </tr>
</tbody>
</table>

</div>

We can use this as a way to create ROPE regions for the effects and it tells us useful things like the probability of direction for the effect. 

This will be more useful when it comes to comparing models and building multiple regression models, but there is also a specific function to get the model $R^2$ and its 95% credible interval. This tells you the proportion of variance in your outcome your predictor(s) explain. 


```r
bayes_R2(Schroeder_fit)
```

```
##     Estimate  Est.Error        Q2.5     Q97.5
## R2 0.1257941 0.07223518 0.007817001 0.2775261
```

Until now, we have focused on point-estimates and intervals of the posterior, but the main strength of Bayesian statistics is summarising the parameters as a whole posterior probability distribution, so we will now turn to the various plotting options. 

The first plot is useful for seeing the posterior of each parameter and the trace plots to check on any convergence issues. 


```r
plot(Schroeder_fit)
```

<img src="10-BayesEst_files/figure-html/Schroeder parameters and trace-1.png" width="100%" style="display: block; margin: auto;" />

For this model, we have three plots: one for the intercept, one for the coefficient/slope, and one for sigma. On the left, we have the posterior probability distributions for each. On the right, we have trace plots. By default, `brms` uses four chains - or series of samples using MCMC - and this shows how each chain moves around the parameter space. Essentially, we want the trace plots to look like fuzzy caterpillars with a random series of lines. If there are spike which deviate massively from the rest, or the lines get stuck in one area, this suggests there are convergence issues. 

These plots are useful for an initial feel of the parameter posteriors, but there are a great series of functions from the <code class='package'>bayestestR</code> package which you can use on their own, or wrap them in the <code><span><span class='fu'><a target='_blank' href='https://rdrr.io/r/graphics/plot.default.html'>plot</a></span><span class='op'>(</span><span class='op'>)</span></span></code> function after loading the <code class='package'>see</code> package. For example, we can see an overlay of the prior and posterior for the main parameters of interest. On its own, <code><span><span class='fu'>p_direction</span><span class='op'>(</span><span class='op'>)</span></span></code> tells you the probability of direction for each parameter, i.e., how much of the distribution is above or below 0? Wrapped in <code><span><span class='fu'><a target='_blank' href='https://rdrr.io/r/graphics/plot.default.html'>plot</a></span><span class='op'>(</span><span class='op'>)</span></span></code>, you can see the prior and posterior, with the posterior divided in areas above or below 0. 


```r
plot(p_direction(Schroeder_fit), 
     priors = TRUE) 
```

<img src="10-BayesEst_files/figure-html/Schroeder p direction-1.png" width="100%" style="display: block; margin: auto;" />

::: {.warning data-latex=""}
For this to work, you must specify priors in `brms`. It does not work with the package default options for the coefficients. 
:::

We can see the pretty wide prior in blue, then the posterior. Almost all of the posterior distribution is above zero to show we're pretty confident that audio is associated with higher hire ratings than transcript. 

The next useful plot is seeing the 95% HDI / credible interval. On its own, <code><span><span class='fu'>hdi</span><span class='op'>(</span><span class='op'>)</span></span></code> will show you the 95% HDI for your parameters. Wrapped in <code><span><span class='fu'><a target='_blank' href='https://rdrr.io/r/graphics/plot.default.html'>plot</a></span><span class='op'>(</span><span class='op'>)</span></span></code>, you can visualise the HDI compared to zero for your main parameters. If the HDI excludes zero, you can be confident in a positive or negative effect, at least conditional on these data and model. Remember, there is a difference between the small world and big world of models. This is not the absolute truth, just the most credible values conditioned on our data and model. 


```r
plot(hdi(Schroeder_fit))
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

Following from chapter 9, we saw we can also use Bayesian statistics to test hypotheses. This works in a modelling approach as `brms` has a function to test hypotheses. We must provide the fitted model object and state a hypothesis to test. This relies on a character description of the parameter and test value. For a full explanation, see the [brms documentation online](https://paul-buerkner.github.io/brms/reference/hypothesis.html) for the function. Here, we will test the coefficient/slope against a point-null of 0. 


```r
hypothesis(Schroeder_fit, # brms model we fitted earlier
           hypothesis = "CONDITIONAudio = 0") 
```

```
## Hypothesis Tests for class b:
##             Hypothesis Estimate Est.Error CI.Lower CI.Upper Evid.Ratio
## 1 (CONDITIONAudio) = 0     1.56      0.58      0.4     2.68       0.11
##   Post.Prob Star
## 1       0.1    *
## ---
## 'CI': 90%-CI for one-sided and 95%-CI for two-sided hypotheses.
## '*': For one-sided hypotheses, the posterior probability exceeds 95%;
## for two-sided hypotheses, the value tested against lies outside the 95%-CI.
## Posterior probabilities of point hypotheses assume equal prior probabilities.
```

::: {.info data-latex=""}
We must state a character hypothesis which requires you to select a parameter. Here, we focus on the <code><span><span class='st'>"CONDITIONAudio"</span></span></code> parameter, i.e., our slope, which must match the name in the model. We can then state values to test against, like here against a point-null of 0 for a Bayes factor. Alternatively, you can test posterior odds where you compare masses of the posterior like CONDITIONAudio > 0.
:::

The key part of the output is the evidence ratio, but we also have the estimate and 95% credible interval. As we are testing a point-null of 0, we are testing the null hypothesis against the alternative of a non-null effect. As the value is below 1, it suggests we have evidence in favour of the alternative compared to the null. I prefer to express things above 1 as its easier to interpret. You can do this by dividing 1 by the ratio, which should provide a Bayes factor of 9.09 here. 

Alternatively, you can calculate the posterior odds by stating regions of the posterior to test. For example, if we used "CONDITIONAudio > 0", this would provide a ratio of the posterior probability of positive effects above 0 to the posterior probability of negative effects below 0. For this example, this would be a posterior odds of 209 in favour of positive effects. Note, when all the posterior is above 0, you can get a result of Inf (infinity) as all the evidence is in favour of positive effects.


```r
hypothesis(Schroeder_fit, # brms model we fitted earlier
           hypothesis = "CONDITIONAudio > 0") 
```

```
## Hypothesis Tests for class b:
##             Hypothesis Estimate Est.Error CI.Lower CI.Upper Evid.Ratio
## 1 (CONDITIONAudio) > 0     1.56      0.58      0.6      2.5     209.53
##   Post.Prob Star
## 1         1    *
## ---
## 'CI': 90%-CI for one-sided and 95%-CI for two-sided hypotheses.
## '*': For one-sided hypotheses, the posterior probability exceeds 95%;
## for two-sided hypotheses, the value tested against lies outside the 95%-CI.
## Posterior probabilities of point hypotheses assume equal prior probabilities.
```

#### 5. Model checking 

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



If we run the <code><span><span class='fu'><a target='_blank' href='https://rdrr.io/r/base/summary.html'>summary</a></span><span class='op'>(</span><span class='op'>)</span></span></code> function again, you can check the intercept and predictor coefficients to see how they differ to the first model we fitted. Ideally, they should provide us with similar inferences, such as a similar magnitude and in the same direction. It is never going to be exactly the same under different priors, but we want our conclusions robust to the choice of prior we use. 


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
   <td style="text-align:right;"> 3.002274 </td>
   <td style="text-align:right;"> 2.1383382 </td>
   <td style="text-align:right;"> 3.989010 </td>
   <td style="text-align:right;"> 1.00000 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> User prior </td>
   <td style="text-align:left;"> b_CONDITIONAudio </td>
   <td style="text-align:right;"> 1.563360 </td>
   <td style="text-align:right;"> 0.4333066 </td>
   <td style="text-align:right;"> 2.692349 </td>
   <td style="text-align:right;"> 0.99525 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Default prior </td>
   <td style="text-align:left;"> b_Intercept </td>
   <td style="text-align:right;"> 2.903251 </td>
   <td style="text-align:right;"> 1.9052542 </td>
   <td style="text-align:right;"> 3.942468 </td>
   <td style="text-align:right;"> 1.00000 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Default prior </td>
   <td style="text-align:left;"> b_CONDITIONAudio </td>
   <td style="text-align:right;"> 1.838562 </td>
   <td style="text-align:right;"> 0.4885740 </td>
   <td style="text-align:right;"> 3.365554 </td>
   <td style="text-align:right;"> 0.98900 </td>
  </tr>
</tbody>
</table>

</div>

### Independent activity (Brandt et al., 2014)

For an independent activity, we will use data from the study by [Brandt et al. (2014)](https://econtent.hogrefe.com/doi/full/10.1027/1864-9335/a000191). The aim of Brandt et al. was to replicate a relatively famous social psychology study on the effect of recalling unethical behaviour on the perception of brightness. 

In common language, unethical behaviour is considered as "dark", so the original authors designed a priming experiment where participants were randomly allocated to recall an unethical behaviour or an ethical behaviour from their past. Participants then completed a series of measures including their perception of how bright the testing room was. Brandt et al. were sceptical and wanted to replicate this study to see if they could find similar results. 

Participants were randomly allocated (<code><span><span class='st'>"ExpCond"</span></span></code>) to recall an unethical behaviour (n = 49) or an ethical behaviour (n = 51). The key outcome is their perception of how bright the room was (<code><span><span class='st'>"WellLitSca"</span></span></code>), from 1 (not bright at all) to 7 (very bright). The research question was: Does recalling unethical behaviour lead people to perceive a room as darker than if they recall ethical behaviour? 

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

#### 1. Identify data

For the second guided example we covered in the lecture, we will explore the model included in [Heino et al. (2018)](https://doi.org/10.1080/21642850.2018.1428102) for their Bayesian data analysis tutorial. They explored the feasibility and acceptability of the ”Let’s Move It” intervention to increase physical activity in 43 older adolescents. 

They randomised participants into two groups (<code><span><span class='st'>"intervention"</span></span></code>) for control (0) and intervention (1) arms (group sessions on motivation and self-regulation skills, and teacher training). Their outcome was a measure of autonomous motivation (<code><span><span class='st'>"value"</span></span></code>) on a 1-5 scale, with higher values meaning greater motivation. They measured the outcome at both baseline (0) and six weeks after (1; <code><span><span class='st'>"time"</span></span></code>).

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

#### 2. Define a descriptive model

I recommend reading the article as they explain this process in more detail. We essentially have an outcome of autonomous motivation (<code><span><span class='st'>"value"</span></span></code>) and we want to look at the interaction between <code><span><span class='st'>"intervention"</span></span></code> and <code><span><span class='st'>"time"</span></span></code>. They define a fixed intercept in the model with the `1 +` part. Its also technically a multi-level model as they define a random intercept for each participant (`(1 | ID)`) to ensure we recognise time is within-subjects. 

::: {.info data-latex=""}
By default, R includes a fixed intercept (the `1 +` part) in the model, so you would get the same results without adding it to the model. However, people often include it so it is explicit in the model formula.
:::


```r
Heino_model <- bf(value ~ 1 + time * intervention + (1 | ID))
```

#### 3. Specify prior probability of parameters

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

#### 4. Update prior to posterior

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
plot(hdi(Heino_fit))
```

<img src="10-BayesEst_files/figure-html/Heino plots-4.png" width="100%" style="display: block; margin: auto;" />

Regardless of the output we look at here, there is not much going on across any of the predictors. The data comes from a feasibility study, so the sample size was pretty small and its mainly about how receptive participants are to the intervention. 

As a bonus extra, you can also use the <code class='package'>emmeans</code> package to calculate marginal effects on the posterior distribution. Its not important here as there is little we can learn from breaking down the interaction further, but it might come in handy in future. 


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

#### 5. Model check

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

For an independent activity, we will use data from the study by [Coleman et al. (2014)](https://psyarxiv.com/k5fp8/). Coleman et al. contains two studies investigating religious mystical experiences. One study focused on undergraduates and a second study focused on experienced meditators who were part of a unique religious group.  

The data set contains a range of variables used for the full model in the paper. We are going to focus on a small part of it for this exercise, but feel free to explore developing the full model as was used in study 1. The key variables are: 

1. <code><span><span class='st'>"Age"</span></span></code> - Measured in years

2. <code><span><span class='st'>"Gender"</span></span></code> - 0 = male; 1 = female

3. <code><span><span class='st'>"Week_med"</span></span></code> - Ordinal measure of how often people meditate per week, with higher values meaning more often

4. <code><span><span class='st'>"Time_session"</span></span></code> - Ordinal measure of how long people meditate per session, with higher values meaning longer

5. <code><span><span class='st'>"Absorption_SUM"</span></span></code> - Sum score of the Modified Tellegen Absorption scale, with higher values meaning greater trait levels of imaginative engagement 

6. <code><span><span class='st'>"EQ_SUM"</span></span></code> - Sum score of the Empathizing Quotient short form, with higher values meaning greater theory of mind ability 

7. <code><span><span class='st'>"Mscale_SUM"</span></span></code> - Sum score of the Hood M-scale, with higher values meaning more self-reported mystical experiences

Previous studies had explored these components separately and mainly in undergraduates, so Coleman et al. took the opportunity to explore a unique sample of a highly committed religious group. The final model included all seven variables, but for this example, we will just focus on absorption (<code><span><span class='st'>"Absorption_SUM"</span></span></code>) and mentalizing (<code><span><span class='st'>"EQ_SUM"</span></span></code>) as they were the main contributors, with the other variables as covariates.

Our research question is: How does absorption (<code><span><span class='st'>"Absorption_SUM"</span></span></code>) and mentalizing (<code><span><span class='st'>"EQ_SUM"</span></span></code>) affect mystical experiences (<code><span><span class='st'>"Mscale_SUM"</span></span></code>) as an outcome? Focus on entering the two variables as individual predictors at first, then explore an interaction. 

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

