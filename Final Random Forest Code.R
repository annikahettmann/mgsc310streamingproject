library("tidyverse")


Streaming <- 
  read_csv(here::here("datasets",
                      "All_Streaming_Shows.csv"))


Streaming_clean <- Streaming %>% 
  filter(`Content Rating` != "N/A", `Content Rating` != "", `IMDB Rating` != "") %>% 
  mutate(Main_Genre = as.factor(unlist(map(strsplit(as.character(.$Genre), "\\,"), 1))),
         Content_Rating = as.numeric(unlist(map(strsplit(as.character(.$`Content Rating`),"\\+"), 1))),
         IMDB_Rating = as.numeric(`IMDB Rating`),
         RT_Rating = as.numeric(`R Rating`),
         Seasons = unlist(map(strsplit(as.character(.$`No of Seasons`), "[S]"), 1)), 
         Seasons = as.numeric(Seasons),
         Streaming_Platform = strsplit(as.character(.$`Streaming Platform`),","))


streaming_df <- Streaming_clean %>% as_tibble() %>% 
  mutate(Seasons_binary = as.factor(ifelse(Streaming_clean$Seasons>1, 1, 0))) %>% 
  mutate_if(is.character, as.factor) %>% 
  drop_na()

set.seed(1818)
streaming_split <- initial_split(streaming_df, prop = 0.75)
streaming_train <- training(streaming_split)
streaming_test <- testing(streaming_split)


# making the radom forest model 
rf_fit_streaming <- randomForest(Seasons_binary ~ 
                                   Main_Genre + Content_Rating + IMDB_Rating + 
                                   RT_Rating, 
                                 data = streaming_df,
                                 type = classification,
                                 mtry = 2,
                                 na.action = na.roughfix,
                                 ntree = 100, #number of models it is going to fit
                                 importance = TRUE)

print(rf_fit_streaming)

# This will tells us the best number of decision trees we need 
plot(rf_fit_streaming)


# to ensure the we have the right mtry we run this code to see what the graph says is the best mtry 
rf_mods <- list()
oob_err <- NULL
test_err <- NULL
for(mtry in 1:5){
  rf_fit_streaming <- randomForest(Seasons_binary ~ 
                                     Main_Genre + Content_Rating + IMDB_Rating + 
                                     RT_Rating, 
                                   data = streaming_df,
                                   type = classification,
                                   mtry = mtry,
                                   na.action = na.roughfix,
                                   ntree = 100, #number of models it is going to fit
                                   importance = TRUE)
  
  oob_err[mtry] <- rf_fit_streaming$err.rate[200]
  
  cat(mtry," ")
}

results_DF <- data.frame(mtry = 1:5, oob_err)
ggplot(results_DF, aes(x = mtry, y = oob_err)) + geom_point() + theme_minimal(base_size = 16)

# partial dependence plots only changing genre
library('pdp')
pdp::partial(rf_fit_streaming, 
             pred.var = "RT_Rating", 
             plot = TRUE,
             rug = TRUE,
             plot.engine = "ggplot2",
             smooth = TRUE)

## PDP prediction interaction
library('randomForestExplainer')
plot_predict_interaction(rf_fit_streaming, 
                         streaming_df,
                         "Content_Rating",
                         "IMDB_Rating")

#to find the most important variable
importance(rf_fit_streaming, type = 1, scale = FALSE)
#to find the most important variable graphically 
varImpPlot(rf_fit_streaming, type = 1, scale = FALSE)


# min depth distribution
plot_min_depth_distribution(rf_fit_streaming)




# Training
preds_train_streaming <- predict(rf_fit_streaming, newdata = streaming_train, type = "vote") %>% 
  as_tibble() %>% select("1")

head(preds_train_streaming)


# Testing
preds_test_streaming <- predict(rf_fit_streaming, newdata = streaming_test, type = "vote") %>% 
  as_tibble() %>% select("1")

head(preds_test_streaming)



results_train <- data.frame(
  `truth` = streaming_train %>% select(Seasons_binary) %>%
    mutate(Seasons_binary = as.numeric(Seasons_binary)-1),
  `Class1` = preds_train_streaming,
  `type` = rep("train",length(preds_train_streaming))
) %>% rename("Class1" = 2)

results_test <- data.frame(
  `truth` = streaming_test  %>% select(Seasons_binary) %>%
    mutate(Seasons_binary = as.numeric(Seasons_binary)-1),
  `Class1` =  preds_test_streaming,
  `type` = rep("test",length(preds_test_streaming))
) %>% rename("Class1" = 2)

library('plotROC')

p_train <- ggplot(results_train, 
                  aes(m = Class1, d= as.numeric(Seasons_binary))) + 
  geom_roc(labelsize = 3.5, 
           cutoffs.at = 
             c(0.99,0.9,0.7,0.5,0.3,0.1,0)) +
  theme_minimal(base_size = 16)

p_test <- ggplot(results_test, 
                 aes(m = Class1, d= as.numeric(Seasons_binary))) + 
  geom_roc(labelsize = 3.5, 
           cutoffs.at = 
             c(0.99,0.9,0.7,0.5,0.3,0.1,0)) +
  theme_minimal(base_size = 16)

plot(p_train)
plot(p_test)

results <- bind_rows(results_train,results_test)

p <- ggplot(results, 
            aes(m = Class1, d= as.numeric(Seasons_binary), color = type)) + 
  geom_roc(labelsize = 3.5, 
           cutoffs.at = 
             c(0.99,0.9,0.7,0.5,0.3,0.1,.05, 0)) +
  theme_minimal(base_size = 16)
print(p)

calc_auc(p_test)
calc_auc(p_train)
calc_auc(p)