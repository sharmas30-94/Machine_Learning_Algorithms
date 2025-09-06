# ============================================================
# Comprehensive ML demo on Rituximab+ WES cohort (simple style)
# ============================================================

# ---- Libraries ----
  library(glmnet)
  library(rpart)
  library(survival)
  library(foreign)
  library(dplyr)
  library(caret)
  library(Boruta)
  library(mlbench)
  library(randomForest)
  library(pROC)
  library(MASS)   
  library(kernlab)



set.seed(111)

# ---- Input ----
infile <- "/Users/sharmas30/Downloads/Machine_Learning_Algorithms/data/lesions_rituximab.csv"

# ---- Load & clean (keep your style) ----
dat  <- read.csv(infile, header = TRUE, sep = ",", check.names = FALSE)
dat2 <- as.data.frame(dat)

# blanks to NA
dat2$OS.YEARS [dat2$OS.YEARS  == ""] <- NA
dat2$OS.Censor[dat2$OS.Censor == ""] <- NA

# drop NAs and zeros
dat2 <- dat2[complete.cases(dat2$OS.YEARS, dat2$OS.Censor), ]
dat2 <- dat2 %>% filter(OS.YEARS != 0)

# label: <=3 years -> 1 (positive), else 0
dat2 <- dat2 %>% mutate(OS_Category = ifelse(OS.YEARS <= 3, "Short", "Long"))

# Rituximab+ and WES only
dat3 <- dat2 %>% filter(Rituximab == 1) %>% filter(Cohort == "WES")
dat3 <- dat3[, c(9:118)]

dat3$OS_Category<-as.factor(dat3$OS_Category)



# ---- Train/Test split (stratified) ----
set.seed(222)
idx   <- createDataPartition(dat3$OS_Category, p = 0.75, list = FALSE)
train_full <- dat3[idx, ]
test_full  <- dat3[-idx, ]

# ---- Caret controls ----
ctrl_cv <- trainControl(
  method = "repeatedcv", number = 10, repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# ---- Helper: metrics (Accuracy + AUC) ----
metrics_row <- function(model, name, TR, TE) {
  lev <- levels(TR$OS_Category)                # two levels, first is the event for twoClassSummary
  pos <- lev[1]                                # event class name, e.g., "Long" (if you set it first)
  neg <- lev[2]
  
  # predicted probabilities for the positive class
  p_tr <- tryCatch(predict(model, newdata = TR, type = "prob")[, pos], error = function(e) NULL)
  p_te <- tryCatch(predict(model, newdata = TE, type = "prob")[, pos], error = function(e) NULL)
  
  # fallback: raw labels mapped to 1/0 for the pos class
  if (is.null(p_tr)) {
    raw_tr <- predict(model, newdata = TR, type = "raw")
    p_tr <- as.numeric(raw_tr == pos)
  }
  if (is.null(p_te)) {
    raw_te <- predict(model, newdata = TE, type = "raw")
    p_te <- as.numeric(raw_te == pos)
  }
  
  y_tr <- TR$OS_Category; y_te <- TE$OS_Category
  
  roc_tr <- tryCatch(pROC::roc(response = y_tr, predictor = p_tr, levels = c(neg, pos), direction = "<"),
                     error = function(e) NA)
  roc_te <- tryCatch(pROC::roc(response = y_te, predictor = p_te, levels = c(neg, pos), direction = "<"),
                     error = function(e) NA)
  
  auc_tr <- if (inherits(roc_tr, "roc")) as.numeric(pROC::auc(roc_tr)) else NA_real_
  auc_te <- if (inherits(roc_te, "roc")) as.numeric(pROC::auc(roc_te)) else NA_real_
  
  # accuracy at 0.5 cutoff
  pred_tr <- factor(ifelse(p_tr >= 0.5, pos, neg), levels = c(neg, pos))
  pred_te <- factor(ifelse(p_te >= 0.5, pos, neg), levels = c(neg, pos))
  acc_tr <- mean(pred_tr == y_tr)
  acc_te <- mean(pred_te == y_te)
  
  data.frame(model = name,
             acc_train = round(acc_tr, 4), auc_train = round(auc_tr, 4),
             acc_test  = round(acc_te, 4), auc_test  = round(auc_te, 4),
             stringsAsFactors = FALSE)
}

  

# ============================================================
# =============== Models on FULL feature set =================
# ============================================================

results_full <- data.frame()



# 2) Random Forest
set.seed(333)
fit_rf_f <- train(OS_Category ~ ., data = train_full, method = "rf",
                  trControl = ctrl_cv, metric = "ROC", importance = TRUE)
results_full <- rbind(results_full, metrics_row(fit_rf_f, "rf (full)", train_full, test_full))

# 3) XGBoost (if available)
if (requireNamespace("xgboost", quietly = TRUE)) {
  set.seed(333)
  fit_xgb_f <- train(OS_Category ~ ., data = train_full, method = "xgbTree",
                     trControl = ctrl_cv, metric = "ROC")
  results_full <- rbind(results_full, metrics_row(fit_xgb_f, "xgbTree (full)", train_full, test_full))
}

# 4) Logistic Regression (GLM)
set.seed(333)
fit_glm_f <- train(OS_Category ~ ., data = train_full, method = "glm",
                   family = binomial, trControl = trainControl(method = "none", classProbs = TRUE))
results_full <- rbind(results_full, metrics_row(fit_glm_f, "glm (full)", train_full, test_full))

# 5) Stepwise Logistic (backward)
full_glm <- glm(OS_Category ~ ., data = train_full, family = binomial)
step_glm_f <- stepAIC(full_glm, direction = "backward", trace = FALSE)
# Wrap stepwise model as a pseudo caret model for metrics
pred_tr <- predict(step_glm_f, newdata = train_full, type = "response")
pred_te <- predict(step_glm_f, newdata = test_full,  type = "response")
roc_tr  <- tryCatch(pROC::roc(train_full$OS_Category, pred_tr, levels = c("0","1"), direction = "<"), error = function(e) NA)
roc_te  <- tryCatch(pROC::roc(test_full$OS_Category,  pred_te, levels = c("0","1"), direction = "<"), error = function(e) NA)
row_step <- data.frame(
  model = "stepAIC glm (full)",
  acc_train = round(mean(ifelse(pred_tr >= 0.5, "1","0") == train_full$OS_Category), 4),
  auc_train = if (inherits(roc_tr,"roc")) round(as.numeric(pROC::auc(roc_tr)), 4) else NA,
  acc_test  = round(mean(ifelse(pred_te >= 0.5, "1","0") == test_full$OS_Category), 4),
  auc_test  = if (inherits(roc_te,"roc")) round(as.numeric(pROC::auc(roc_te)), 4) else NA,
  stringsAsFactors = FALSE
)
results_full <- rbind(results_full, row_step)

# 6) Ridge / Lasso / Elastic Net (glmnet)
lambda <- 10^seq(-3, 3, length = 100)

set.seed(333)
fit_ridge_f <- train(OS_Category ~ ., data = train_full, method = "glmnet",
                     trControl = ctrl_cv, metric = "ROC",
                     tuneGrid = expand.grid(alpha = 0, lambda = lambda))
results_full <- rbind(results_full, metrics_row(fit_ridge_f, "ridge (full)", train_full, test_full))

set.seed(333)
fit_lasso_f <- train(OS_Category ~ ., data = train_full, method = "glmnet",
                     trControl = ctrl_cv, metric = "ROC",
                     tuneGrid = expand.grid(alpha = 1, lambda = lambda))
results_full <- rbind(results_full, metrics_row(fit_lasso_f, "lasso (full)", train_full, test_full))

set.seed(333)
fit_enet_f <- train(OS_Category ~ ., data = train_full, method = "glmnet",
                    trControl = ctrl_cv, metric = "ROC", tuneLength = 10)
results_full <- rbind(results_full, metrics_row(fit_enet_f, "elasticnet (full)", train_full, test_full))

# 7) SVM (radial)
set.seed(333)
fit_svm_f <- train(OS_Category ~ ., data = train_full, method = "svmRadial",
                   trControl = ctrl_cv, metric = "ROC")
results_full <- rbind(results_full, metrics_row(fit_svm_f, "svmRadial (full)", train_full, test_full))

# 8) kNN
set.seed(333)
fit_knn_f <- train(OS_Category ~ ., data = train_full, method = "knn",
                   trControl = ctrl_cv, metric = "ROC")
results_full <- rbind(results_full, metrics_row(fit_knn_f, "knn (full)", train_full, test_full))

# 9) Naive Bayes (if klaR available)
if (requireNamespace("klaR", quietly = TRUE)) {
  set.seed(333)
  fit_nb_f <- train(OS_Category ~ ., data = train_full, method = "nb",
                    trControl = ctrl_cv, metric = "ROC")
  results_full <- rbind(results_full, metrics_row(fit_nb_f, "naiveBayes (full)", train_full, test_full))
}



# ============================================================
# ========= Run BORUTA algorithm for feature collection ======
# ============================================================
dat3$OS_Category <- factor(dat3$OS_Category)
set.seed(111)
bor <- Boruta(OS_Category ~ ., data = dat3, doTrace = 1, maxRuns = 500)
bor_fix <- TentativeRoughFix(bor)
features_confirmed <- rownames(subset(bor_stats, decision == "Confirmed"))
keep<-c(features_confirmed, "OS_Category")
train_sel <- train_full[, keep]
test_sel <- test_full[, keep]
# ============================================================
# ========= Models on BORUTA-SELECTED feature set ============
# ============================================================
results_sel <- data.frame()

# Decision Tree
set.seed(333)
fit_rpart_s <- train(OS_Category ~ ., data = train_sel, method = "rpart",
                     trControl = ctrl_cv, metric = "ROC")
results_sel <- rbind(results_sel, metrics_row(fit_rpart_s, "rpart (sel)", train_sel, test_sel))

# Random Forest
set.seed(333)
fit_rf_s <- train(OS_Category ~ ., data = train_sel, method = "rf",
                  trControl = ctrl_cv, metric = "ROC", importance = TRUE)
results_sel <- rbind(results_sel, metrics_row(fit_rf_s, "rf (sel)", train_sel, test_sel))

# XGBoost
if (requireNamespace("xgboost", quietly = TRUE)) {
  set.seed(333)
  fit_xgb_s <- train(OS_Category ~ ., data = train_sel, method = "xgbTree",
                     trControl = ctrl_cv, metric = "ROC")
  results_sel <- rbind(results_sel, metrics_row(fit_xgb_s, "xgbTree (sel)", train_sel, test_sel))
}

# Logistic (GLM)
set.seed(333)
fit_glm_s <- train(OS_Category ~ ., data = train_sel, method = "glm",
                   family = binomial, trControl = trainControl(method = "none", classProbs = TRUE))
results_sel <- rbind(results_sel, metrics_row(fit_glm_s, "glm (sel)", train_sel, test_sel))

# Stepwise Logistic
full_glm_s <- glm(OS_Category ~ ., data = train_sel, family = binomial)
step_glm_s <- stepAIC(full_glm_s, direction = "backward", trace = FALSE)
pred_tr <- predict(step_glm_s, newdata = train_sel, type = "response")
pred_te <- predict(step_glm_s, newdata = test_sel,  type = "response")
roc_tr  <- tryCatch(pROC::roc(train_sel$OS_Category, pred_tr, levels = c("0","1"), direction = "<"), error = function(e) NA)
roc_te  <- tryCatch(pROC::roc(test_sel$OS_Category,  pred_te, levels = c("0","1"), direction = "<"), error = function(e) NA)
row_step <- data.frame(
  model = "stepAIC glm (sel)",
  acc_train = round(mean(ifelse(pred_tr >= 0.5, "1","0") == train_sel$OS_Category), 4),
  auc_train = if (inherits(roc_tr,"roc")) round(as.numeric(pROC::auc(roc_tr)), 4) else NA,
  acc_test  = round(mean(ifelse(pred_te >= 0.5, "1","0") == test_sel$OS_Category), 4),
  auc_test  = if (inherits(roc_te,"roc")) round(as.numeric(pROC::auc(roc_te)), 4) else NA,
  stringsAsFactors = FALSE
)
results_sel <- rbind(results_sel, row_step)

# Ridge / Lasso / Elastic Net
set.seed(333)
fit_ridge_s <- train(OS_Category ~ ., data = train_sel, method = "glmnet",
                     trControl = ctrl_cv, metric = "ROC",
                     tuneGrid = expand.grid(alpha = 0, lambda = lambda))
results_sel <- rbind(results_sel, metrics_row(fit_ridge_s, "ridge (sel)", train_sel, test_sel))

set.seed(333)
fit_lasso_s <- train(OS_Category ~ ., data = train_sel, method = "glmnet",
                     trControl = ctrl_cv, metric = "ROC",
                     tuneGrid = expand.grid(alpha = 1, lambda = lambda))
results_sel <- rbind(results_sel, metrics_row(fit_lasso_s, "lasso (sel)", train_sel, test_sel))

set.seed(333)
fit_enet_s <- train(OS_Category ~ ., data = train_sel, method = "glmnet",
                    trControl = ctrl_cv, metric = "ROC", tuneLength = 10)
results_sel <- rbind(results_sel, metrics_row(fit_enet_s, "elasticnet (sel)", train_sel, test_sel))

# SVM
set.seed(333)
fit_svm_s <- train(OS_Category ~ ., data = train_sel, method = "svmRadial",
                   trControl = ctrl_cv, metric = "ROC")
results_sel <- rbind(results_sel, metrics_row(fit_svm_s, "svmRadial (sel)", train_sel, test_sel))

# kNN
set.seed(333)
fit_knn_s <- train(OS_Category ~ ., data = train_sel, method = "knn",
                   trControl = ctrl_cv, metric = "ROC")
results_sel <- rbind(results_sel, metrics_row(fit_knn_s, "knn (sel)", train_sel, test_sel))

# Naive Bayes
if (requireNamespace("klaR", quietly = TRUE)) {
  set.seed(333)
  fit_nb_s <- train(OS_Category ~ ., data = train_sel, method = "nb",
                    trControl = ctrl_cv, metric = "ROC")
  results_sel <- rbind(results_sel, metrics_row(fit_nb_s, "naiveBayes (sel)", train_sel, test_sel))
}

# ---- Combine & save metrics ----
results_full$set <- "full"
results_sel$set  <- "boruta_selected"

all_metrics <- rbind(results_full, results_sel)
print(all_metrics[order(-all_metrics$auc_test), ])

# write out
if (!dir.exists("results")) dir.create("results")
write.csv(all_metrics, "/Users/sharmas30/Downloads/Machine_Learning_Algorithms/results/metrics_all_models.csv", row.names = FALSE)


# ---- Confusion matrix example (best model by test AUC) ----
best_idx <- which.max(all_metrics$auc_test)
best_row <- all_metrics[best_idx, ]
cat("\nBest model by test AUC:\n"); print(best_row)


