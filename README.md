# multicorr

Given a dataset with null values in

```tsv
id	c1	c2	c3	v
1	A	A	C	10
2	B	A	C	20
3	A	B	C	30
4	B	B	C	
5	A	A	C	50
6	B	A	C	60
7	A	B	C	70
8	B	B	C	
```

We want to identify the _levels of the nullspace_ (meaning the values of the categorical variables
which give rise to the nulls), and specifically the _most parsimonious_ possible set of them.

In the toy dataset above (`simple_dataset.csv`) we can see that:

- There are 3 categorical variables: `c1`, `c2`, and `c3`. In R these are known as factors, and they
  can be null. In `pandas` we have categories and they
  [can't be null](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html#differences-to-r-s-factor).
  Since we're looking at nullable categorical variables, we'll adopt the R terminology and refer to the categories as "factors",
  but keep the pandas terminology of referring to the values themselves as "levels".
- The 3rd categorical variable is invariant (it's a constant). From this we can expect that it will be correlated with
  everything, including our nullspace of interest. It is redundant to any explanation
  of correlated variance in our data, and in practical terms we don't care about it (it's useless info).
- If you look at just the rows with empty `v` value (rows with `id` 4 and 8) they are both "B,B,C".
- We already know c3 is always C, and if we look closely we see these are the only rows with "B,B".
  Therefore the answer here is that when `c1` = B and `c2` = B we have a null value in `v`.

In a real world context, we might receive some data and find empty values in it, and wonder what
leads some to be empty and not others, in a structural way.

For instance, we might have data describing various modes of transport from a transport provider,
and null values for timetables may mean that a particular mode of transport is not scheduled.
Or a weather service may not provide air pressure readings for some weather stations that are
underwater (you get the idea). This 'structural' sense of the data can be eyeballed, but it's better
to establish it automatically.

For this I used simple correlation and a manually constructed 2D dummy variable procedure (which I
hope will scale well with real data).

For the toy dataset above, `corr_analysis_multicat_dummy_form.py` gives:

```py
c1_B:c2_B         1.00000
c1_B:c2_B:c3_C    1.00000
c1_B              0.57735
c2_B              0.57735
c1_B:c3_C         0.57735
c2_B:c3_C         0.57735
```

We see the "`c1` B `c2` B" value is top, and the redundant info is lower, we penalise it for its
redundant info.

## Alternatives: logistic regression

I tried a simple logreg and found the same interpretation.

- see `logreg_interaction.py`

```
         Current function value: 0.173287
         Iterations: 35
/home/louis/miniconda3/lib/python3.10/site-packages/statsmodels/base/model.py:604: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                 null_v   No. Observations:                    8
Model:                          Logit   Df Residuals:                        5
Method:                           MLE   Df Model:                            2
Date:                Sat, 01 Apr 2023   Pseudo R-squ.:                  0.6918
Time:                        21:01:31   Log-Likelihood:                -1.3863
converged:                      False   LL-Null:                       -4.4987
Covariance Type:            nonrobust   LLR p-value:                   0.04449
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
c1_B       -7.136e-05        nan        nan        nan         nan         nan
c2_B       -7.133e-05   3.11e+12  -2.29e-17      1.000    -6.1e+12     6.1e+12
c1c2_A:B     -20.8065   3.11e+12  -6.69e-12      1.000    -6.1e+12     6.1e+12
c1c2_B:A     -20.8065        nan        nan        nan         nan         nan
c1c2_B:B      20.8065   2.54e+12   8.19e-12      1.000   -4.98e+12    4.98e+12
==============================================================================

Possibly complete quasi-separation: A fraction 0.75 of observations can be
perfectly predicted. This might indicate that there is complete
quasi-separation. In this case some parameters will not be identified.
```

- It didn't converge (it's a toy dataset, might be too few data points or class imbalance)
- It correctly found strong anticorrelations in the interaction terms of c1 and c2 (A and B / vice versa), which were **never** associated with null values
- It correctly found strong correlations in the interaction term of c1_B and c2_B

In practice I expect I'd opt for this instead (but is it truly more efficient and straightforward when the categories are absolutely separable?).
