# Introduction to Bayesian Hypothesis Testing

## Learning objectives

blah blah blah. 

## The Logic Behind Bayesian Inference {#bayes-logic}

To demonstrate the logic behind Bayesian inference, we will play around with this shiny app by [Wagenmakers (2015)](https://shiny.psy.lmu.de/felix/BayesLessons/BayesianLesson1.Rmd). The text walks you through the app and provides exercises to explore on your own, but we will use it to explore defining a prior distribution and seeing how the posterior updates with data. 

The example is based on estimating the the proportion of yellow candies in a bag with different coloured candies. If you see a yellow candy, it is logged as a 1. If you see a non-yellow candy, it is logged as a 0. We want to know what proportion of the candies are yellow. 

This is a handy demonstration for the logic behind Bayesian inference as it is the simplest application. Behind the scenes, we could calculate these values directly as the distributions are simple and there is only one parameter. In later examples and in lesson 10, we will focus on more complicated models which require sampling for the posterior. 

### Step 1 - Pick your prior 

First, we define what our prior expectations are for the proportion of yellow candies. For a dichotomous outcome like this (yellow or not yellow), we can model the prior as a beta distribution (define). There are only two parameters to set: *a* and *b*. 

Explore changing the parameters and their impact on the distribution, but here are a few observations to orient yourself: 

1. Setting both to 1 create a flat prior: any proportion is possible. 

2. Using the same number for both centers the distribution on 0.5, with increasing numbers showing greater certainty (higher peak). 

3. If parameter a < b, proportions less than 0.5 are more likely. 

4. If parameter b > a, proportions higher than 0.5 are more likely. 

After playing around, what proportion of yellow candies do you think are likely? Are you certain about the value or are you more accepting of the data? 

### Step 2 - Update to the posterior

Now we have a prior, its time to collect some data and update to the posterior. In the lecture, we will play around with a practical demonstration of seeing how many candies are yellow, so set your prior by entering a value for *a* and *b*, and we will see what the data tell us. There are two boxes for entering data: the number of yellows you observe, and the number of non-yellows you observe. 

If you are trying this on your own, explore changing the prior and data to see how it affects the posterior distribution. For some inspiration and key observations: 

1. Setting an uninformative (1,1) or weak prior (2, 2) over 0.5, the posterior is dominated by the data. For example, imagine we observed 5 yellows and 10 non-yellows. The posterior peaks around 0.30 but plausibly ranges between 0.2 and 0.6. Changing the data completely changes the posterior to show the prior has very little influence. As you change the number of yellows and non-yellows, the posterior updates more dramatically. 

2. Now set a strong prior (20, 20) over 0.5 with 5 yellows and 10 non-yellows. Despite the observed data showing a proportion of 0.33, the peak of the posterior distribution is slightly higher than 0.4. The posterior is a compromise between the prior and the likelihood, so a stronger prior means you need more data to change your beliefs. For example, imagine we had 10 times more data with 50 yellows and 100 non-yellows. Now, there is greater density between 0.3 and 0.4, to show the posterior is now more convinced of the proportion of yellows. 

In this demonstration, note we only have two curves for the prior and posterior, without the likelihood. When the prior and posterior come from the same distribution family, it is known as a conjugate prior (define), and the beta distribution is one of the simplest. We are simply modelling the proportion of successes and failures (here yellows vs non-yellows). 

In section 4, you can also explore Bayes factors applied to this scenario, working out how much you should shift your belief in favour of the alternative hypothesis compared to the null (here, that the proportion is exactly 0.5). At this point of the lecture, we have not explored Bayes factors yet, so we will not continue with it here. But you can always come back to it later in your own time. 

## The Logic Behind Bayes Factors

To demonstrate the logic behind Bayes factors, we will play around with this shiny app by [Magnusson](https://rpsychologist.com/d3/bayes/). Building on [section 1](#bayes-logic), the visualisation shows a Bayesian two-sample t-test, demonstrating a more complicated application compared to the beta distribution and proportions. The visualisation shows both Bayesian estimation through the posterior distribution and 95% Highest Density Interval (HDI) (define), and Bayes factors against a null hypothesis centered on 0. We will use this visualisation to reinforce what we learnt earlier and extend it to understanding the logic behind Bayes factors. 

On the visualisation, there are three settings:

1. Observed effect - Expressed as Cohen's d, this represents the standardised mean difference between two groups. You can set larger effects (positive or negative) or assume the null hypothesis is true (d = 0). 

2. Sample size - You can increase the sample size from 1 to 1000, with more steps between 1 and 100. 

3. SD of prior - The prior is always set to 0 as we are testing against the null hypothesis, but you can specify how strong this prior is. Decreasing the SD means you are more confident in an effect of 0. Increasing the SD means you are less certain about the prior. 

This visualisation is set up to test against the null hypothesis of no difference between two groups, but remember Bayes factors allow you to test any two hypotheses. The Bayes factor is represented by the difference between the two dots, where the curves represent a likelihood of 0 in the prior and posterior distributions. The Bayes factor is a ratio between these two values for the posterior odds of how much your belief should shift in favour of the experimental hypothesis compared to the null hypothesis. 

::: {.try data-latex=""}
Play around with the visualisation yourself to see how the Bayes factor changes as you alter the settings. Keep an eye on the *p*-value and 95% Confidence Interval (CI) to see where the inferences are similiar or different.
:::

To reinforce the lessons from [section 1](#bayes-logic) and the emphasis now on Bayes factors, there are some key observations:

1. With a less informative prior (higher SD), the posterior is dominated by the likelihood to show the posterior is overwhelmed by data. For example, if you set the SD to 2, the prior peaks at 0 but the distribution is so flat it accepts any reasonable effect. If you move the observed effect anywhere along the scale, the likelihood and posterior almost completely overlap until you reach d = ±2. 

2. With a stronger prior (lower SD), the posterior represents more of a compromise between the prior and likelihood. If you change the SD to 0.5 and the observed effect to 1, the posterior is closer to being an intermediary. You may have the observed data, but your prior belief in a null effect is so strong, it requires more data to be convinced otherwise.

3. With more participants / data, there is less uncertainty in the likelihood. Keeping the same parameters as point 2, with 10 participants, the likelihood peaks at d = 1, but easily spans between 0 and 2. As you increase the sample size towards 1000, the uncertainty around the likelihood is lower. More data also overwhelm the prior, so although we had a relatively strong prior that we would have a null effect, with 50 participants, the likelihood and posterior mostly overlap. 

4. Focusing on the Bayes factor supporting the experimental hypothesis, if there is an effect, evidence in favour of the experimental hypothesis increases as the observed effect increases and with increasing sample size. This is not too dissimilar to frequentist statistical power, but with Bayesian statistics, optional stopping is less of a problem (revise and add citation). So, if we do not have enough data to shift our beliefs towards either hypothesis, we can collect more data and update our beliefs. 

5. If we set the observed effect to 0, the *p*-value is 1 to suggest we cannot reject the null, but cannot shift our belief towards the null. With Bayes factors, we can focus on supporting the null and see with observed effect = 0, sample size = 50, and SD of prior = 0.5, the data are 2.60 times more likely under the null than the experimental hypothesis. So, we should shift our belief in favour of the null, but it is not very convincing. We can obtain a higher Bayes factor in support of the null by increasing the sample size or decreasing the SD of the prior. This last part might sound a little odd at first, but if your prior was very strong in favour of the null, your beliefs do not need to shift in light of the data. 

6. Finally, if you set a weak prior (SD = 2), you will see the frequentist 95% CI and the Bayesian 95% HDI are almost identical. With a weak or uninformative prior, you usually find the values from the two intervals are very similar, but you have to interpret them differently. Increasing the sample size makes both intervals smaller and changing the observed effect shifts them around. If you make a stronger prior (SD = 0.5), now the 95% HDI will change. The frequentist 95% CI will always follow the likelihood as it is only based on the observed data. The Bayesian 95% CI represents the area of the posterior, so it might be a compromise between the prior and likelihood, or it can be smaller if you have a stronger prior in favour of the null and an observed effect of 0. 

## Bayes factors for two independent samples

### Guided example (Bastian et al., 2014)

As this is the first time we've actually used R so far, we need to load some packages and the data for this task. If you do not have any of the packages, make sure you install them first. 


```r
library(BayesFactor)
library(bayestestR)
library(tidyverse)
```

For this guided example, we will reanalyse data from  [Bastian et al. (2014)](http://journals.sagepub.com/doi/10.1177/0956797614545886). This study wanted to investigate whether experiencing pain together can increase levels of bonding between participants. The study was trying to explain how people often say friendships are strengthened by adversity. 

Participants were randomly allocated into two conditions: pain or control. Participants in the pain group experienced mild pain through a cold pressor task (leaving your hand in ice cold water) and a wall squat (sitting against a wall). The control group completed a different task that did not involve pain. The participants then completed a scale to measure how bonded they felt to other participants in the group. Higher values on this scale mean greater bonding. 

The independent variable is called <code><span class='st'>"CONDITION"</span></code>. The control group has the value 0 and the pain group has the value 1. They wanted to find out whether participants in the pain group would have higher levels of bonding with their fellow participants than participants in the control group. After a little processing, the dependent variable is called <code><span class='st'>"mean_bonding"</span></code> for the mean of 7 items related to bonding. 


```r
Bastian_data <- read_csv("data/Bastian.csv")

# Relabel condition to be more intuitive which group is which 
Bastian_data$CONDITION <- factor(Bastian_data$CONDITION, 
                                   levels = c(0, 1), 
                                   labels = c("Control", "Pain"))

# First we need to get our DV from the mean of 7 items
bonding <- Bastian_data %>% 
  pivot_longer(cols = group101:group107, names_to = "item", values_to = "score") %>% 
  group_by(subid) %>% 
  summarise(mean_bonding = mean(score), ) %>% 
  ungroup()

# add back to the main dataframe and simplify 
Bastian_data <- left_join(Bastian_data, bonding, by = "subid") %>% 
  select(CONDITION, mean_bonding)
```

To use the Bayesian version of the t-test, we use similar arguments to the frequentist version by stating our design with a formula and which data frame you are referring to. For this study, we want to predict the bonding rating by the group they were allocated into: `mean_bonding ~ CONDITION`. 

The main new argument here is `rscale` which sets the width of the prior distribution around the alternative hypothesis. T-tests use a [Cauchy prior](https://en.wikipedia.org/wiki/Cauchy_distribution) which is similar to a normal distribution but with fatter tails and you only have to define one parameter: the r scale. The default prior is set to <code><span class='st'>"medium"</span></code>, but you could change this depending on your understanding of the area of research. See the function help page for different options here, but medium is equivalent to a value of 0.707 for scaling the Cauchy prior which is the default setting for most statistics software. On a two-tailed test, this means 50% of the distribution covers values between ± 0.707. You can enter a numeric value for the precise scaling or there are a few word presets like <code><span class='st'>"medium"</span></code> and <code><span class='st'>"wide"</span></code> depending on how weak you want the prior to be. 


```r
Bastian_ttest <- ttestBF(formula = mean_bonding ~ CONDITION,
                        data = Bastian_data,
                        rscale = "medium", 
                        paired = FALSE)
```

```
## Warning: data coerced from tibble to data frame
```

```r
Bastian_ttest
```

```
## Bayes factor analysis
## --------------
## [1] Alt., r=0.707 : 1.445956 ±0.01%
## 
## Against denominator:
##   Null, mu1-mu2 = 0 
## ---
## Bayes factor type: BFindepSample, JZS
```

Don't worry about the warning, there were just previous issues with using tibbles in the <code class='package'>BayesFactor</code> package. 

With the medium prior, we have a Bayes factor of 1.45 ($BF$$_1$$_0$ = 1.45), suggesting the experimental hypothesis is 1.45 times more likely than the point null hypothesis. By the guidelines from Wagenmakers et al. (2011) (add ref), this is quite weak anecdotal evidence.

The authors were pretty convinced that the pain group would score hire on the bonding rating than the control group, so lets see what happens with a one-tailed test to see how its done. We need to define the `nullInterval` argument to state we only consider negative effects. 

::: {.warning data-latex=""}
Make sure you check the order of your groups to check which direction you expect the results to go in. If you expect group A to be smaller than group B, you would code for negative effects. If you expect group A to be bigger than group B, you would code for positive effects. A common mistake is defining the wrong direction if you do not know which order the groups are coded. 
:::


```r
Bastian_onetail <- ttestBF(formula = mean_bonding ~ CONDITION,
                        data = Bastian_data,
                        rscale = "medium", 
                        paired = FALSE,
                        nullInterval = c(-Inf, 0)) # negative only as we expect control < pain
```

```
## Warning: data coerced from tibble to data frame
```

```r
Bastian_onetail
```

```
## Bayes factor analysis
## --------------
## [1] Alt., r=0.707 -Inf<d<0    : 2.790031  ±0%
## [2] Alt., r=0.707 !(-Inf<d<0) : 0.1018811 ±0.02%
## 
## Against denominator:
##   Null, mu1-mu2 = 0 
## ---
## Bayes factor type: BFindepSample, JZS
```

In a one-tailed test, we now have two tests. In row one, we have the test we want where we compare our experimental hypothesis (negative effects) against the point null. In row two, we have the opposite which is the complement of our experimental hypothesis. Even with a one-tailed test, the evidence in favour of our experimental hypothesis compared to the null is anecdotal at best ($BF$$_1$$_0$ = 2.79). 

If we wanted to test the null compared to the experimental hypothesis, we can simply take the reciprocal of the object, here demonstrated on the two-tailed object. 


```r
1 / Bastian_ttest
```

```
## Bayes factor analysis
## --------------
## [1] Null, mu1-mu2=0 : 0.6915839 ±0.01%
## 
## Against denominator:
##   Alternative, r = 0.707106781186548, mu =/= 0 
## ---
## Bayes factor type: BFindepSample, JZS
```

For this object, we already know there is anecdotal evidence in favour of the experimental hypothesis, so this is just telling us the null is less likely than the experimental hypothesis ($BF$$_0$$_1$ = 0.69). This will come in handy when you specifically want to test the null though. 

For the purposes of the rest of the demonstration, we will stick with our original object with a two-tailed test to see how we can interpret inconclusive results. In the original study, the pain group scored significantly higher than the control group, but the p-value was .048, so hardly convincing evidence. With Bayes factors, at least we can see we ideally need more data to make a decision. 

We will spend more time on this process next week, but a Bayes factor on its own is normally not enough. We also want an estimate of the effect size and the precision around it. Within the <code class='package'>BayesFactor</code> package, there is a function to sample from the posterior distribution using MCMC sampling. We need to pass the t-test object into the `posterior` function, and include the number of iterations we want. We will use 10,000 here. Depending on your computer, this may take a few seconds.

::: {.warning data-latex=""}
If you use a one-tailed test, you must index the first object (e.g., `Bastian_ttest[1]`) as a one-tailed test includes two lines: 1) the directional alternative we state against the null and 2) the complement of the alternative against the null. 
:::


```r
Bastian_samples <- posterior(Bastian_ttest,
                            iterations = 1e5) # 10,000 in math notation
```

Once we have the samples, we can use the generic `plot` function to see trace plots (more on those in chapter 10) and a density plot of the posterior distributions for several parameters. 


```r
plot(Bastian_samples)
```

<img src="09-BayesHypo_files/figure-html/Bastian plot-1.png" width="100%" style="display: block; margin: auto;" /><img src="09-BayesHypo_files/figure-html/Bastian plot-2.png" width="100%" style="display: block; margin: auto;" />

The second and fourth plots are what we are mainly interested in for a t-test. Once we know what kind of evidence we have for different hypotheses, typically we want to know what the effect size is. In the <code class='package'>BayesFactor</code> package, we get the mean difference between groups (unhelpfully named beta) and the effect size Delta, which is kind of like Cohen's d. It is calculated by dividing the t statistic by the square root of the sample size, so kind of a standardised mean difference. One of my main complaints with the <code class='package'>BayesFactor</code> package is not explaining what the outputs mean as the only explanation I could find is this [old blog post](http://bayesfactor.blogspot.com/2014/02/bayes-factor-t-tests-part-1.html), with no clear overview in the documentation.  

The plot provides the posterior distribution of different statistics based on sampling 10,000 times. For beta, we can see the peak of the distribution is around -0.5, spanning from above 0 to -1. For delta, we can see the peak of the distribution is around -0.5, and spans from above 0 to -1 again. 

For a more fine-tuned description of the posterior distribution, we can use handy functions from the <code class='package'>bayestestR</code> package. We will use this much more in chapter 10 as there are some great plotting functions, but these functions work for BayesFactor objects. To get the point estimates of each parameter, we can use the <code><span class='va'>point_estimate</span></code> function: 


```r
point_estimate(Bastian_samples)
```

<div class="kable-table">

<table>
 <thead>
  <tr>
   <th style="text-align:left;"> Parameter </th>
   <th style="text-align:right;"> Median </th>
   <th style="text-align:right;"> Mean </th>
   <th style="text-align:right;"> MAP </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> mu </td>
   <td style="text-align:right;"> 3.4268400 </td>
   <td style="text-align:right;"> 3.4269619 </td>
   <td style="text-align:right;"> 3.4263454 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> beta (Control - Pain) </td>
   <td style="text-align:right;"> -0.4876165 </td>
   <td style="text-align:right;"> -0.4921400 </td>
   <td style="text-align:right;"> -0.4616819 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> sig2 </td>
   <td style="text-align:right;"> 1.1036346 </td>
   <td style="text-align:right;"> 1.1328496 </td>
   <td style="text-align:right;"> 1.0556915 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> delta </td>
   <td style="text-align:right;"> -0.4638837 </td>
   <td style="text-align:right;"> -0.4693724 </td>
   <td style="text-align:right;"> -0.4351125 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> g </td>
   <td style="text-align:right;"> 0.5452621 </td>
   <td style="text-align:right;"> 5.8537940 </td>
   <td style="text-align:right;"> 0.0280655 </td>
  </tr>
</tbody>
</table>

</div>

Our best guess for the mean difference between groups is -0.49 and a delta of -0.46 in favour of the pain group. 

We do not just want a point estimate though, we also want the credible interval around it. For this, we have the <code><span class='va'>hdi</span></code> function. 


```r
hdi(Bastian_samples)
```

<div class="kable-table">

<table>
 <thead>
  <tr>
   <th style="text-align:left;"> Parameter </th>
   <th style="text-align:right;"> CI </th>
   <th style="text-align:right;"> CI_low </th>
   <th style="text-align:right;"> CI_high </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> mu </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> 3.1486730 </td>
   <td style="text-align:right;"> 3.7177499 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> beta (Control - Pain) </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -1.0446880 </td>
   <td style="text-align:right;"> 0.0411462 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> sig2 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> 0.7262775 </td>
   <td style="text-align:right;"> 1.5913185 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> delta </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -0.9974157 </td>
   <td style="text-align:right;"> 0.0368466 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> g </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> 0.0280655 </td>
   <td style="text-align:right;"> 7.6111096 </td>
  </tr>
</tbody>
</table>

</div>

So, we can be 95% confident that the mean difference is between -1.04 and 0.04, and that delta is between -0.99 and 0.04. As both values cross 0, we would not be confident in these findings and ideally we would need to collect more data.   

Finally, instead of separate functions, there is a handy wrapper for the median, 95% credible interval, and ROPE (more on that in section 4). 


```r
describe_posterior(Bastian_samples)
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
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> 4 </td>
   <td style="text-align:left;"> mu </td>
   <td style="text-align:right;"> 3.4268400 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> 3.1486730 </td>
   <td style="text-align:right;"> 3.7177499 </td>
   <td style="text-align:right;"> 1.00000 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -0.1 </td>
   <td style="text-align:right;"> 0.1 </td>
   <td style="text-align:right;"> 0.0000000 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> beta (Control - Pain) </td>
   <td style="text-align:right;"> -0.4876165 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -1.0446880 </td>
   <td style="text-align:right;"> 0.0411462 </td>
   <td style="text-align:right;"> 0.96521 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -0.1 </td>
   <td style="text-align:right;"> 0.1 </td>
   <td style="text-align:right;"> 0.0525468 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 5 </td>
   <td style="text-align:left;"> sig2 </td>
   <td style="text-align:right;"> 1.1036346 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> 0.7262775 </td>
   <td style="text-align:right;"> 1.5913185 </td>
   <td style="text-align:right;"> 1.00000 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -0.1 </td>
   <td style="text-align:right;"> 0.1 </td>
   <td style="text-align:right;"> 0.0000000 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> delta </td>
   <td style="text-align:right;"> -0.4638837 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -0.9974157 </td>
   <td style="text-align:right;"> 0.0368466 </td>
   <td style="text-align:right;"> 0.96521 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -0.1 </td>
   <td style="text-align:right;"> 0.1 </td>
   <td style="text-align:right;"> 0.0565152 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 3 </td>
   <td style="text-align:left;"> g </td>
   <td style="text-align:right;"> 0.5452621 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> 0.0280655 </td>
   <td style="text-align:right;"> 7.6111096 </td>
   <td style="text-align:right;"> 1.00000 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -0.1 </td>
   <td style="text-align:right;"> 0.1 </td>
   <td style="text-align:right;"> 0.0339681 </td>
  </tr>
</tbody>
</table>

</div>

There are a bunch of tests and tricks we have not covered here, so check out the [Bayesfactor package page online](https://richarddmorey.github.io/BayesFactor/) for a series of vignettes. 

### Independent activity (Schroeder & Epley, 2015)

For an independent activity, we will use data from the study by [Schroeder and Epley (2015)](https://journals.sagepub.com/stoken/default+domain/PhtK6MPtXvkgnYRrnGbA/full). The aim of the study was to investigate whether delivering a short speech to a potential employer would be more effective at landing you a job than writing the speech down and the employer reading it themselves. Thirty-nine professional recruiters were randomly assigned to receive a job application speech as either a transcript for them to read, or an audio recording of them reading the speech. 

The recruiters then rated the applicants on perceived intellect, their impression of the applicant, and whether they would recommend hiring the candidate. All ratings were originally on a Likert scale ranging from 0 (low intellect, impression etc.) to 10 (high impression, recommendation etc.), with the final value representing the mean across several items. 

For this example, we will focus on the hire rating (variable <code><span class='st'>"Hire_Rating"</span></code> to see whether the audio condition would lead to higher ratings than the transcript condition (variable <code><span class='st'>"CONDITION"</span></code>). 


```r
Schroeder_data <- read_csv("data/Schroeder_hiring.csv")

# Relabel condition to be more intuitive which group is which 
Schroeder_data$CONDITION <- factor(Schroeder_data$CONDITION, 
                                   levels = c(0, 1), 
                                   labels = c("Transcript", "Audio"))
```

From here, apply what you learnt in the first guided example to this new independent task. 


```r
Schroeder_ttest <- NULL
```

## Bayes factors for two dependent samples

### Guided example (Mehr et al., 2016)

For a paired samples t-test, the process is identical to the independent samples t-test apart from defining the variables. So, we will demonstrate a full example like before with less commentary, then you have an independent data frame to test your understanding. 

The next study we are going to look at is by [Mehr et al. (2016)](http://journals.sagepub.com/doi/10.1177/0956797615626691). They were interested in whether singing to infants conveyed important information about social affiliation. Infants become familiar with melodies that are repeated in their specific culture. The authors were interested in whether a novel person (someone they had never seen before) could signal to the child that they are a member of the same social group and attract their attention by singing a familiar song to them.

Mehr et al. (2016) invited 32 infants and their parents to participate in a repeated measures experiment. First, the parents were asked to repeatedly sing a previously unfamiliar song to the infants for two weeks. When they returned to the lab, they measured the baseline gaze (where they were looking) of the infants towards two unfamiliar people on a screen who were just silently smiling at them. This was measured as the proportion of time looking at the individual who would later sing the familiar song (0.5 would indicate half the time was spent looking at the familiar singer. Values closer to one indicate looking at them for longer). The two silent people on the screen then took it in turns to sing a lullaby. One of the people sung the song that the infant’s parents had been told to sing for the previous two weeks, and the other one sang a song with the same lyrics and rhythm, but with a different melody. Mehr et al. (2016) then repeated the gaze procedure to the two people at the start of the experiment to provide a second measure of gaze as a proportion of looking at the familiar singer. 

We are interested in whether the infants increased the proportion of time spent looking at the singer who sang the familiar song after they sang, in comparison to before they sang to the infants. We have one dependent variable (gaze proportion) and one within-subjects independent variable (baseline vs test). We want to know whether gaze proportion was higher at test than it was at baseline. 


```r
Mehr_data <- read_csv("data/Mehr_voice.csv") %>% 
  select(Baseline = Baseline_Proportion_Gaze_to_Singer, # Shorten super long names
         Test = Test_Proportion_Gaze_to_Singer)

Mehr_ttest <- ttestBF(x = Mehr_data$Baseline,
                      y = Mehr_data$Test,
                      paired = TRUE, 
                      rscale = "medium")

Mehr_ttest
```

```
## Bayes factor analysis
## --------------
## [1] Alt., r=0.707 : 2.296479 ±0%
## 
## Against denominator:
##   Null, mu = 0 
## ---
## Bayes factor type: BFoneSample, JZS
```

Like Bastian et al. (2014), we have just anecdotal evidence in favour of the experimental hypothesis over the null ($BF$$_1$$0$ = 2.30). 

::: {.try data-latex=""}
Mehr et al. (2016) expected gaze proportion to be higher at test, so try defining a one-tailed test and see what kind of evidence we have in favour of the alternative hypothesis.
:::

As before, we know there is only anecdotal evidence in favour of the experimental hypothesis, but we also want the effect size and its 95% credible interval. So, we can sample from the posterior, plot it, and get some estimates. 


```r
Mehr_samples <- posterior(Mehr_ttest,
                          iterations = 1e5)
```

Just note in the plot we have fewer panels as the paired samples approach simplifies things. Mu is our mean difference in the first panel and delta is our standardised effect in the third panel. For a dependent variable of gaze proportion, an unstandardised effect size is more informative than the standardised version.


```r
plot(Mehr_samples)
```

<img src="09-BayesHypo_files/figure-html/Mehr plots-1.png" width="100%" style="display: block; margin: auto;" />

We finally have our wrapper function for the median posterior distribution and its 95% credible interval. 


```r
describe_posterior(Mehr_samples)
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
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> 3 </td>
   <td style="text-align:left;"> mu </td>
   <td style="text-align:right;"> -0.0665363 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -0.1249159 </td>
   <td style="text-align:right;"> -0.0072690 </td>
   <td style="text-align:right;"> 0.9865 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -0.1 </td>
   <td style="text-align:right;"> 0.1 </td>
   <td style="text-align:right;"> 0.8926432 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 4 </td>
   <td style="text-align:left;"> sig2 </td>
   <td style="text-align:right;"> 0.0288608 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> 0.0167185 </td>
   <td style="text-align:right;"> 0.0462870 </td>
   <td style="text-align:right;"> 1.0000 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -0.1 </td>
   <td style="text-align:right;"> 0.1 </td>
   <td style="text-align:right;"> 1.0000000 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> delta </td>
   <td style="text-align:right;"> -0.3919915 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -0.7449702 </td>
   <td style="text-align:right;"> -0.0387591 </td>
   <td style="text-align:right;"> 0.9865 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -0.1 </td>
   <td style="text-align:right;"> 0.1 </td>
   <td style="text-align:right;"> 0.0287155 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> g </td>
   <td style="text-align:right;"> 0.4881952 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> 0.0298578 </td>
   <td style="text-align:right;"> 6.8212717 </td>
   <td style="text-align:right;"> 1.0000 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> -0.1 </td>
   <td style="text-align:right;"> 0.1 </td>
   <td style="text-align:right;"> 0.0424206 </td>
  </tr>
</tbody>
</table>

</div>

We can see here the median posterior estimate of the mean difference is -.07 with a 95% credible interval ranging from -.13 to -.01. This means we can exclude 0 from the likely effects, but we still only have anecdotal evidence in favour of the experimental hypothesis compared to the null.

### Independent activity (Zwaan et al., 2020)

For your final independent activity, we have data from [Zwaan et al. (2020)](http://link.springer.com/10.3758/s13423-017-1348-y) who wanted to replicate experiments from cognitive psychology to see how replicable they are. For this exercise, we will explore the [flanker task](https://en.wikipedia.org/wiki/Eriksen_flanker_task). 

In short, we have two conditions: congruent and incongruent. In congruent trials, five symbols like arrows are the same and participants must identify the central symbol with a keyboard response. In incongruent trials, the four outer symbols are different to the central symbol. Typically, we find participants respond faster to congruent trials than incongruent trials. The dependent variable here is the mean response time in milliseconds (ms). 

We want to know whether response times are faster to congruent trials (session1_responsecongruent) than incongruent trials (session1_incongruent). Zwaan et al. measured a few things like changing the stimuli and repeated the task in two sessions, so we will just focus on the first session for this example. In this scenario, you might want to think of a one-tailed test since there is a strong prediction to expect faster responses in the congruent condition. 


```r
Zwaan_data <- read_csv("data/Zwaan_flanker.csv")

Zwaan_ttest <- NULL
```

## Equivalence Testing vs ROPE

### Guided example for two independent samples 

### Independent activity 