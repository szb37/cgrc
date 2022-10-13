df = read.csv("C:/gd_sync/projects/bbc/codebase/tests/fixtures//pseudodata_get_model_stats1.csv")
model_wo_guess = lm(formula = delta_score ~ condition, df)
model_w_guess  = lm(formula = delta_score ~ condition + guess + guess*condition, df)
summary(model_wo_guess)
summary(model_w_guess)


emm_fix_guess = emmeans(model_w_guess, specs = pairwise ~ condition|guess) # stratas with fixed guess
emm_fix_cond  = emmeans(model_w_guess, specs = pairwise ~ guess|condition) # stratas with fixed condition 
emm_fix_guess$contrasts # Comparison of strata with fixed guess (i.e. the two comparisons in the top row of fig5 - PL/PL vs MD/PL and PL/MD vs MD/MD)
emm_fix_cond$contrasts  # Comparison of strata with fixed condition (i.e. the bottom two comparisons on fig 5 - PL/PL vs PL/MD and MD/PL vs MD/MD)

""" Reproduce cross check values for Tukey adj test """

df=read.csv("C:/gd_sync/projects/bbc/codebase/tests/fixtures//pseudodata_get_stats_data1.csv")

df$condition = as.factor(df$condition)
df$guess     = as.factor(df$guess)
df <- within(df, condition <- relevel(condition, ref="PL"))
df <- within(df, guess <- relevel(guess, ref="PL"))

modelGuess  = lm(formula = delta_score ~ condition + guess + guess*condition, df)
em=emmeans(modelGuess, specs= c("condition", "guess"))
res <- pairs(em) 
