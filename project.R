library(ggmosaic)

data = read.csv("/Users/kimhyunmin/CTGdata.csv", header=TRUE)

data$NSP <- ifelse(data$NSP==1, "Normal", (ifelse(data$NSP ==2, "Suspect", "Pathologic")))
data$Tendency <- ifelse(data$Tendency==1, "Left asymmetric", (ifelse(data$Tendency==0,"Symmetric", "Right symmetric") ))

data$NSP = as.factor(data$NSP)
data$Tendency = as.factor(data$Tendency)

#mosaic plot
ggplot(data=data) + geom_mosaic(aes(x=product(NSP), fill=Tendency))+
  geom_mosaic_text(aes(x=product(NSP), fill=Tendency)) + ggtitle("Mosaic plot of NSP and Tendency")


#data loading
data <- data[,-c(6,17,18,22)]
data$NSP = as.factor(data$NSP)

library(caret)

data <- upSample(data[,1:18], data[,19], yname="NSP")
sn <- sample(1:nrow(data), size=nrow(data)*0.7)
train <- data[sn,]

#Support Vector Classifier
ctrl <- trainControl(method="repeatedcv",
                     number=10,
                     repeats=3, summaryFunction=multiClassSummary, classProbs=FALSE)
start<- Sys.time()
svc.tune <- train(form = NSP ~ ., data = train,
                  method = "svmLinear",
test <- data[-sn,]

table(data$NSP)
table(train$NSP)

#Decision Tree

library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(caret)
library(pROC)
library(mltest)
library(officer)

set.seed(1234)
start <- Sys.time()

#fit model
rpartmod <- rpart(NSP~., data=train, method="class")
#pruning(print.cp:cv)
ptree<-prune(rpartmod, cp=rpartmod$cptable[which.min(rpartmod$cptable[,"xerror"]),"CP"])

end<-Sys.time()
time<-end-start

#prediction
rpartpred <- predict(ptree, test, type="class")
confusionMatrix(rpartpred, test$NSP)

#test plot
rparttest <- rpart(NSP~., data=test, method="class")
fancyRpartPlot(rparttest)


#roc curve
pred <- as.numeric(rpartpred)
ROC_tree <- multiclass.roc(test$NSP,pred)

rs<-ROC_tree[["rocs"]]

plot.roc(rs[[1]],  
         col="red",   
         print.auc=TRUE,  
         max.auc.polygon=FALSE,   
         auc.polygon=TRUE, auc.polygon.col="#D1F2EB",asp = NA) 
plot.roc(rs[[2]],   
         add=TRUE,  
         col="blue",
         print.auc=TRUE, print.auc.adj=c(1.11,0)  
)    
plot.roc(rs[[3]],   
         add=TRUE,   
         col="Orange",
         print.auc=TRUE, print.auc.adj=c(3,-1)  
) 
title("ROC curve for Decision Tree",line=2.5)
legend(0.2,0.31, legend=c("NSP=1", "NSP=2","NSP=3"), col=c("red", "blue","Orange"), lwd=2) 

#variance importance
library(vip)
library(ggplot2)
vip(rparttest, num_features = 5)+theme_light()

#summary
ml<-ml_test(rpartpred, test$NSP)

accuracy <- ml$accuracy
precision <- ml$precision
recall <- ml$recall
F1 <- ml$F1
time <- time[[1]]

df_tree <- data.frame("accuracy"=accuracy, "F1"=F1, "recall"=recall, "precision"=precision,"time"=time)

library(officedown)

library(palmerpenguins)
library(gtsummary)

library(flextable)


for (i in 1:dim(df_tree)[2]){
  df_tree[i]<-lapply(df_tree[i], function(y) round(y, 3) )
}
rownames(df_tree)<-c("1","2","3")


knitr::opts_chunk$set()
officedown::rdocx_document
flextable(df_tree%>%rownames_to_column("NSP"))%>%theme_vanilla()%>% set_caption(caption = "Result of Decision Tree")

#Random Forest

start <- Sys.time()

#tunning
library(randomForest)
t <- tuneRF(train[,-19], train[,19], stepFactor = 0.5, plot = TRUE, ntreeTry = 300, trace = TRUE, improve = 0.05, doBest = TRUE)

set.seed(1234)
#fitting
rf_fit <- randomForest(NSP~., data=train, ntree = 300, mtry = t$mtry, importance = TRUE, proximity = TRUE)

end<-Sys.time()
time<-end-start

print(rf_fit)

#plot
library(tidyverse)
library(randomForest)
library(dplyr)
library(ggraph)
library(igraph)

tree_func <- function(final_model, 
                      tree_num) {
  
  # get tree by index
  tree <- randomForest::getTree(final_model, 
                                k = tree_num, 
                                labelVar = TRUE) %>%
    tibble::rownames_to_column() %>%
    # make leaf split points to NA, so the 0s won't get plotted
    mutate(`split point` = ifelse(is.na(prediction), `split point`, NA))
  
  # prepare data frame for graph
  graph_frame <- data.frame(from = rep(tree$rowname, 2),
                            to = c(tree$`left daughter`, tree$`right daughter`))
  
  # convert to graph and delete the last node that we don't want to plot
  graph <- graph_from_data_frame(graph_frame) %>%
    delete_vertices("0")
  
  # set node labels
  V(graph)$node_label <- gsub("_", " ", as.character(tree$`split var`))
  V(graph)$leaf_label <- as.character(tree$prediction)
  V(graph)$split <- as.character(round(tree$`split point`, digits = 2))
  
  # plot
  plot <- ggraph(graph, 'dendrogram') + 
    theme_bw() +
    geom_edge_link() +
    geom_node_point() +
    geom_node_text(aes(label = node_label), na.rm = TRUE, repel = TRUE) +
    geom_node_label(aes(label = split), vjust = 2.5, na.rm = TRUE, fill = "white") +
    geom_node_label(aes(label = leaf_label, fill = leaf_label), na.rm = TRUE, 
                    repel = TRUE, colour = "white", fontface = "bold", show.legend = FALSE) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_blank(),
          panel.background = element_blank(),
          plot.background = element_rect(fill = "white"),
          panel.border = element_blank(),
          axis.line = element_blank(),
          axis.text.x = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks = element_blank(),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          plot.title = element_text(size = 18))
  
  print(plot)
}

#test plot
rf_fit2 <- randomForest(NSP~., data=test, ntree = 300, mtry = t$mtry, importance = TRUE, proximity = TRUE)
tree_func(final_model = rf_fit2, tree_num = 10)

#predict&confusion matrix
rf_pred <- predict(rf_fit, newdata=test)
confusionMatrix(rf_pred, test$NSP, mode="everything")

#variance importance
library(vip)
library(ggplot2)
vip(rf_fit2, num_features = 5)+theme_light()

#roc curve
pred <- as.numeric(predict(rf_fit, newdata=test))
ROC_rf <- multiclass.roc(test$NSP,pred)

rs<-ROC_rf[["rocs"]]

plot.roc(rs[[1]],  
         col="red",   
         print.auc=TRUE,  
         max.auc.polygon=FALSE,   
         auc.polygon=TRUE, auc.polygon.col="#D1F2EB",asp = NA) 
plot.roc(rs[[2]],   
         add=TRUE,  
         col="blue",
         print.auc=TRUE, print.auc.adj=c(1.11,0)  
)    
plot.roc(rs[[3]],   
         add=TRUE,   
         col="Orange",
         print.auc=TRUE, print.auc.adj=c(3,-1)  
) 
title("ROC curve for Random Forest",line=2.5)
legend(0.2,0.31, legend=c("NSP=1", "NSP=2","NSP=3"), col=c("red", "blue","Orange"), lwd=2) 

#summary

ml<-ml_test(rf_pred, test$NSP)

accuracy <- ml$accuracy
precision <- ml$precision
recall <- ml$recall
F1 <- ml$F1
time <- time[[1]]

df_rf <- data.frame("accuracy"=accuracy, "F1"=F1, "recall"=recall, "precision"=precision,"time"=time)

for (i in 1:dim(df_rf)[2]){
  df_rf[i]<-lapply(df_rf[i], function(y) round(y, 3) )
}
rownames(df_rf)<-c("1","2","3")

flextable(df_rf%>%rownames_to_column("NSP"))%>%theme_vanilla()%>% set_caption(caption = "Result of Random Forest")

#support vector classifier
ctrl <- trainControl(method="repeatedcv",
                     number=10,
                     repeats=3, summaryFunction=multiClassSummary, classProbs=FALSE)
start<- Sys.time()
svc.tune <- train(form = NSP ~ ., data = train,
                  method = "svmLinear",
                  tuneLength = 5,
                  preProc = c("center","scale"),
                  trControl=ctrl,
                  tuneGrid = expand.grid(C = seq(0.1, 2, length = 20)))
end<-Sys.time()
time_taken_svc<-end-start
svc.tune
plot(svc.tune)
df<-data.frame(svc.tune$results[which.max(svc.tune$results[,2]),c(1:5,8,9)])
for (i in 1:dim(df)[2]){
  df[i]<-lapply(df[i], function(y) round(y, 3) ) }
flextable(df)%>%theme_vanilla()%>% set_caption(caption = "Result of tuning")

y_pred_svc = predict(svc.tune, newdata = test)
caret::confusionMatrix(test$NSP, y_pred_svc)

varImp(svc.tune, data=test)
plot(varImp(svc.tune, data=test),main="Importance plot of SVC")
ml_svc<-ml_test(y_pred_svc,test$NSP)
accuracy_svc<-ml_svc$accuracy
F1_svc<-ml_svc$F1
recall_svc<-ml_svc$recall
precision_svc<-ml_svc$precision

df_svc<- data.frame("Accuracy"=accuracy_svc,"F1"=F1_svc,"Recall"=recall_svc,"precision"=precision_svc,"Time"=time _taken_svc[[1]])
for (i in 1:dim(df_svc)[2]){
  df_svc[i]<-lapply(df_svc[i], function(y) round(y, 3) )
}
rownames(df_svc)<-c("1","2","3")

flextable(df_svc%>%rownames_to_column("NSP"))%>%theme_vanilla()%>% set_caption(caption = "Result of Support vector classifier")
y_svc_new = as.numeric(predict(svc.tune, newdata = test))
ROC_svc<-multiclass.roc(test$NSP, y_svc_new)
rs<-ROC_svc[["rocs"]]

plot.roc(rs[[1]], col="red",
         print.auc=TRUE,
         max.auc.polygon=FALSE,
         auc.polygon=TRUE, auc.polygon.col="#D1F2EB",asp = NA)
plot.roc(rs[[2]], add=TRUE,
         col="blue",
         print.auc=TRUE, print.auc.adj=c(1.11,0) )
plot.roc(rs[[3]], add=TRUE,
         col="Orange",
         print.auc=TRUE, print.auc.adj=c(3,-1) )
title("ROC curve for Support vector classifier",line=2.5)
legend(x=0.2,y=0.31 ,,legend=c("NSP=1", "NSP=2","NSP=3"), col=c("red", "blue","Orange"), lwd=2)


#Support Vector Machine
start<- Sys.time()
svm.tune <- train(form = NSP ~ ., data = train,
                  method = "svmRadial", tuneLength = 5,
                  preProc = c("center","scale"), trControl=ctrl,
                  tuneGrid = expand.grid(C = seq(0.1, 2, length = 5),sigma=round(seq(0.1, 2, length = 4),2)))
end<-Sys.time()
time_taken_svm<-(end-start)*60
svm.tune

varImp(svm.tune, data=train)
plot(varImp(svm.tune, data=train),main="Importance plot of SVM")

df<-data.frame(svm.tune$results[which.max(svm.tune$results[,3]),c(1:5,8,9)])

for (i in 1:dim(df)[2]){
  df[i]<-lapply(df[i], function(y) round(y, 3) ) }
plot(svm.tune)

flextable(df)%>%theme_vanilla()%>% set_caption(caption = "Result of tuning")

y_svm_pred = predict(svm.tune, newdata = test)
caret::confusionMatrix(test$NSP, y_svm_pred)
ml_svm<-ml_test(y_svm_pred,test$NSP)
accuracy_svm<-ml_svm$accuracy
F1_svm<-ml_svm$F1
recall_svm<-ml_svm$recall
precision_svm<-ml_svm$precision

df_svm<- data.frame("Accuracy"=accuracy_svm,"F1"=F1_svm,"Recall"=recall_svm,"precision"=precision_svm,"Time"=ti me_taken_svm[[1]])

for (i in 1:dim(df_svm)[2]){
  df_svm[i]<-lapply(df_svm[i], function(y) round(y, 3) )
}
rownames(df_svm)<-c("1","2","3")

flextable(df_svm%>%rownames_to_column("NSP"))%>%theme_vanilla()%>% set_caption(caption = "Result of Support vector machine")
y_svm_new = as.numeric(predict(svm.tune, newdata = test))
ROC_svc<-multiclass.roc(test$NSP, y_svm_new)

rs<-ROC_svc[["rocs"]]

plot.roc(rs[[1]], col="red",
         print.auc=TRUE,
         auc.polygon=TRUE, auc.polygon.col="#D1F2EB",asp = NA) plot.roc(rs[[2]],
                                                                        add=TRUE,
                                                                        col="blue",
                                                                        print.auc=TRUE, print.auc.adj=c(1.11,0) )
plot.roc(rs[[3]], add=TRUE,
         col="Orange",
         print.auc=TRUE, print.auc.adj=c(3,-1) )
title("ROC curve for Support vector machine",line=2.5)
legend(x=0.2,y=0.31 , legend=c("NSP=1", "NSP=2","NSP=3"), col=c("red", "blue","Orange"), lwd=2)


#XGBoost
y_train=train[,19]-1

X_train=as.matrix(train[,1:ncol(train)-1])
X_train[,18]=X_train[,18] - 1

X_train[,18]=as.factor(X_train[,18])

y_test=test[,19]-1
X_test=as.matrix(test[,1:ncol(train)-1])
X_test[,18]=X_test[,18] - 1

X_test[,18]=as.factor(X_test[,18])

xgboost_train = xgb.DMatrix(data=X_train, label=y_train)
xgboost_test = xgb.DMatrix(data=X_test, label=y_test)

params <- list(booster = "gbtree", objective = "multi:softprob", num_class = 3, eval_metric = "mlogloss")
# Calculate # of folds for cross-validation

start_xgb<-Sys.time()
xgbcv <- xgb.cv(params = params,
                data = xgboost_train, nrounds = 1000,
                nfold = 10,
                showsd = TRUE, stratified = TRUE, print_every_n = 100, early_stop_round = 20, maximize = FALSE, prediction = TRUE)
model <- xgb.train(params = params, data = xgboost_train, nrounds = 1000)
end_xgb<-Sys.time()

time_taken_xgb<-end_xgb-start_xgb

summary(model)

xgb_test_preds = predict(model, xgboost_test)
test_labs=test$NSP-1
xgb_test_out <- matrix(xgb_test_preds, nrow = 3, ncol = length(xgb_test_preds) / 3) %>%
  t() %>%
  data.frame() %>%
  mutate(max = max.col(., ties.method = "last"), label = test_labs+1)

# Confustion Matrix
classification_error <- function(conf_mat) {
  conf_mat = as.matrix(conf_mat)
  error = 1 - sum(diag(conf_mat)) / sum(conf_mat)
  return (error) }
xgb_test_conf <- table(true = test_labs + 1, pred = xgb_test_out$max)
cat("XGB Test Classification Error Rate:", classification_error(xgb_test_conf), "\n")

xgb_test_conf2 <- caret::confusionMatrix(as.factor(xgb_test_out$label), as.factor(xgb_test_out$max),
                                         mode = "everything")

print(xgb_test_conf2)

Precision<-c(xgb_test_conf2[[4]][13],xgb_test_conf2[[4]][14],xgb_test_conf2[[4]][15])
Recall<-c(xgb_test_conf2[[4]][16],xgb_test_conf2[[4]][17],xgb_test_conf2[[4]][18])
F1<-c(xgb_test_conf2[[4]][19],xgb_test_conf2[[4]][20],xgb_test_conf2[[4]][21])
time<-rep(time_taken_xgb[[1]],3)
accuracy<-rep(xgb_test_conf2[[3]][1],3)
xgb.df1<-as.data.frame(cbind(Precision,Recall,F1,time,accuracy))
#rownames(xgb.df1)<-c("NSP=1","NSP=2","NSP=3")
xgb.df1

for (i in 1:dim(xgb.df1)[2]){ xgb.df1[i]<-lapply(xgb.df1[i], function(y) round(y, 3) ) }
rownames(xgb.df1)<-c("1","2","3")

flextable(xgb.df1%>%rownames_to_column("NSP"))%>%theme_vanilla()%>% set_caption(caption = "Result of XGBoost")

# Variable importance
library(vip)
library(ggplot2) # for theme_light() function vip(model, num_features = 5)+theme_light()

# ROC plot
library(pROC)

roc_xgb<-multiclass.roc(xgb_test_out$label,xgb_test_out$max,levels=c(1,2,3))
print(roc_xgb)
roc_xgb$auc

rs<-roc_xgb[["rocs"]]

plot.roc(rs[[1]], col="red",
         print.auc=TRUE,
         max.auc.polygon=FALSE,
         auc.polygon=TRUE, auc.polygon.col="#D1F2EB",asp = NA)
plot.roc(rs[[2]], add=TRUE,
         col="blue",
         print.auc=TRUE, print.auc.adj=c(1.11,0) )
plot.roc(rs[[3]], add=TRUE,
         col="Orange",
         print.auc=TRUE, print.auc.adj=c(3,-1) )
title("ROC curve for XGBoost classifier",line=2.5)
legend(x=0.2,y=0.31,legend=c("NSP=1", "NSP=2","NSP=3"), col=c("red", "blue","Orange"), lwd=2)

#KNN
index<-sample(1:nrow(data1),nrow(data1)*0.7)
train <- data1[index, ]
test <- data1[-index, ]

train[,19]<-as.numeric(as.character(train[,19]))
test[,19]<-as.numeric(as.character(test[,19]))

X_train=as.matrix(train[,1:ncol(train)-1])
X_train[,18]=as.factor(X_train[,18])
X_test=as.matrix(test[,1:ncol(train)-1])
X_test[,18]=as.factor(X_test[,18])
y_train=train[,19] y_test=test[,19]

trctrl <- caret::trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(1234)

start_knn<-Sys.time()
knn_fit <- caret::train(as.factor(NSP) ~., data = train, method = "knn",
                        trControl=trctrl,
                        preProcess = c("center", "scale"), tuneLength = 10)
end_knn<-Sys.time() time_taken_knn<-end_knn-start_knn
knn_fit
plot(knn_fit) # k=5

test_pred <- predict(knn_fit, newdata = test)

classification_error <- function(conf_mat) { conf_mat = as.matrix(conf_mat)
error = 1 - sum(diag(conf_mat)) / sum(conf_mat)
return (error) }

knn_test_conf <- table(true = test_labs + 1, pred = test_pred)
cat("XGB Test Classification Error Rate:", classification_error(knn_test_conf), "\n")

(knn_test_conf2<-caret::confusionMatrix(test_pred, as.factor(y_test), mode = "everything"))
Precision<-c(knn_test_conf2[[4]][13],knn_test_conf2[[4]][14],knn_test_conf2[[4]][15])
Recall<-c(knn_test_conf2[[4]][16],knn_test_conf2[[4]][17],knn_test_conf2[[4]][18])
F1<-c(knn_test_conf2[[4]][19],knn_test_conf2[[4]][20],knn_test_conf2[[4]][21])

time<-rep(time_taken_knn[[1]],3)
accuracy<-rep(knn_test_conf2[[3]][1],3)
knn.df1<-as.data.frame(cbind(Precision,Recall,F1,time,accuracy))
#rownames(knn.df1)<-c("NSP=1","NSP=2","NSP=3")
knn.df1

for (i in 1:dim(knn.df1)[2]){ knn.df1[i]<-lapply(knn.df1[i], function(y) round(y, 3) ) }
rownames(knn.df1)<-c("1","2","3")

flextable(knn.df1%>%rownames_to_column("NSP"))%>%theme_vanilla()%>% set_caption(caption = "Result of KNN")

library(pROC)

roc_knn<-multiclass.roc(as.ordered(test_pred),as.numeric(as.matrix(test['NSP'])),levels=c(1,2,3))
print(roc_knn)
roc_knn$auc

rs<-roc_knn[["rocs"]]

plot.roc(rs[[1]], col="red",
         print.auc=TRUE,
         max.auc.polygon=FALSE,
         auc.polygon=TRUE, auc.polygon.col="#D1F2EB",asp = NA)
plot.roc(rs[[2]], add=TRUE,
         col="blue",
         print.auc=TRUE, print.auc.adj=c(1.11,0) )
plot.roc(rs[[3]], add=TRUE,
         col="Orange",
         print.auc=TRUE, print.auc.adj=c(3,-1) )
title("ROC curve for KNN",line=2.5)
legend(x=0.2,y=0.31,legend=c("NSP=1", "NSP=2","NSP=3"), col=c("red", "blue","Orange"), lwd=2)