# === Configuration Section ===
# Main directory path
MAIN_DIR <- 'PATH/TO/FOLDER'

# Input/Output file paths
INPUT_FILE <- file.path(MAIN_DIR, 'tokenized.csv')
OUTPUT_FILES <- list(
  topic_summary = file.path(MAIN_DIR, 'Topic summary figure.pdf'),
  topic_relevance = file.path(MAIN_DIR, '4.Topic relevance.pdf'),
  topic_keywords = file.path(MAIN_DIR, '4.Topic keywords.csv'),
  topic_mapping = file.path(MAIN_DIR, '4.Topic mapping.csv'),
  topic_trends = file.path(MAIN_DIR, '4.Topic trends.pdf'),
  topic_trends_fit = file.path(MAIN_DIR, '4.Topic trends fit.csv'),
  frame_event_effects = file.path(MAIN_DIR, '4.frame_event_effects.pdf'),
  frame_fan_effects = file.path(MAIN_DIR, '4.frame_fan_effects.pdf'),
  frame_event_effects_csv = file.path(MAIN_DIR, '4.frame_event_effects.csv'),
  frame_fan_effects_csv = file.path(MAIN_DIR, '4.frame_fan_effects.csv'),
  posts = file.path(MAIN_DIR, '4.Posts.csv')
)

# Model parameters
K_TOPICS <- 24


library(stm)
library(readr)
library(Cairo)
library(tm)
library(SnowballC)
library(dplyr)
library(ggplot2)
library(extrafont)
library(gridExtra)
library(huge)
library(Matrix)
library(tidyr)
library(stringr)
library(scales)

# === 1. Preparing data ===
new_data <- read_delim(INPUT_FILE, 
                       delim = ",",
                       locale = locale(encoding = "UTF-8"))

names(new_data)[names(new_data) == "链接"] <- "link"

new_data$Date <- as.Date(new_data$Date)
new_data$DateIndex <- as.numeric(new_data$Date - min(new_data$Date, na.rm = TRUE))
new_data$event <- as.numeric(new_data$event)
new_data$fan <- as.factor(new_data$fan)

# Preprocessing
processed_data <- textProcessor(
  documents = new_data$text,
  metadata = new_data[, c("text", "event", "fan", "link", "Date", "DateIndex")],  # 添加Date相关列
  removestopwords = FALSE,    
  removenumbers = FALSE,      
  removepunctuation = FALSE,  
  stem = FALSE,
  wordLengths = c(2, Inf),
  verbose = TRUE
)

# Prepare documents
out_data <- prepDocuments(
  processed_data$documents, 
  processed_data$vocab,
  processed_data$meta,
  lower.thresh = 1,
  verbose = TRUE
)

meta_data <- out_data$meta

# === 2. Run ===
stm.out_data <- stm(out_data$documents, 
                    out_data$vocab, 
                    K = K_TOPICS,  
                    prevalence = ~ event + fan,
                    data = out_data$meta, 
                    max.em.its = 275, 
                    seed = 286424,
                    verbose = TRUE,
                    reportevery = 5,
                    init.type = "Spectral")

# === 3. Calculate proportions of original topics ===
print("\n=== Calculate proportions of original topics ===")
original_proportions <- colSums(stm.out_data$theta) / nrow(stm.out_data$theta)

# Orig
original_doc_topics <- rep(NA, nrow(new_data))
for(i in 1:nrow(meta_data)) {
  original_idx <- which(new_data$link == meta_data$link[i])
  if(length(original_idx) == 1) {
    max_pos <- which.max(stm.out_data$theta[i,])
    original_doc_topics[original_idx] <- max_pos
  }
}
original_topic_counts <- table(original_doc_topics)

print("\nOriginal distribution: ")
for(i in 1:K_TOPICS) {
  print(paste("Topic", i, ":",
              "Proportion =", round(original_proportions[i], 4),
              "Number of Documents =", original_topic_counts[as.character(i)]))
}

# === 4. Reorder topics ===
sorted_indices <- order(original_proportions, decreasing = TRUE)
topic_mapping <- data.frame(
  original_topic = sorted_indices,
  new_topic = 1:K_TOPICS
)

print("\nReorder mapping:")
print("Format: New topic <- Original topic")
for(i in 1:K_TOPICS) {
  print(paste(i, "<-", topic_mapping$original_topic[i]))
}

# Reorder
stm.out_data$theta <- stm.out_data$theta[, sorted_indices]
stm.out_data$beta$logbeta[[1]] <- stm.out_data$beta$logbeta[[1]][sorted_indices,]

# === 5. Calculate proportion of new topics ===
# Calculate proportion
new_proportions <- colSums(stm.out_data$theta) / nrow(stm.out_data$theta)

new_doc_topics <- rep(NA, nrow(new_data))
for(i in 1:nrow(meta_data)) {
  original_idx <- which(new_data$link == meta_data$link[i])
  if(length(original_idx) == 1) {
    max_pos <- which.max(stm.out_data$theta[i,])
    new_doc_topics[original_idx] <- max_pos
  }
}
new_topic_counts <- table(new_doc_topics)

# Print new topics
print("\nNew topics:")
for(i in 1:K_TOPICS) {
  print(paste("Topic", i, ":",
              "Proportion =", round(new_proportions[i], 4),
              "Number of Documents =", new_topic_counts[as.character(i)]))
}

# === 6. Get top words and representative documents ===
labels <- labelTopics(stm.out_data, n = 10)
representative_docs <- list()
for (i in 1:K_TOPICS) {
  representative_docs[[i]] <- findThoughts(stm.out_data,
                                           text = meta_data$text,
                                           n = 3, 
                                           topics = i)$docs[[1]]
}

output_df <- data.frame(
  Topic_Number = character(K_TOPICS * 4),
  Proportion = character(K_TOPICS * 4),
  Label_Type = character(K_TOPICS * 4),
  Keywords = character(K_TOPICS * 4),
  Representative_Docs = character(K_TOPICS * 4),
  stringsAsFactors = FALSE
)

for (i in 1:K_TOPICS) {
  base_idx <- (i-1) * 4 + 1
  output_df$Topic_Number[base_idx] <- i
  output_df$Proportion[base_idx] <- round(new_proportions[i], 4)
  
  output_df$Label_Type[base_idx] <- "Highest Prob"
  output_df$Label_Type[base_idx + 1] <- "FREX"
  output_df$Label_Type[base_idx + 2] <- "Lift"
  output_df$Label_Type[base_idx + 3] <- "Score"
  
  output_df$Keywords[base_idx] <- paste(labels$prob[i,], collapse = ", ")
  output_df$Keywords[base_idx + 1] <- paste(labels$frex[i,], collapse = ", ")
  output_df$Keywords[base_idx + 2] <- paste(labels$lift[i,], collapse = ", ")
  output_df$Keywords[base_idx + 3] <- paste(labels$score[i,], collapse = ", ")
  
  docs <- representative_docs[[i]]
  output_df$Representative_Docs[base_idx] <- docs[1]
  output_df$Representative_Docs[base_idx + 1] <- docs[2]
  output_df$Representative_Docs[base_idx + 2] <- docs[3]
  output_df$Representative_Docs[base_idx + 3] <- ""
}


# === Save results ===
CairoPDF(OUTPUT_FILES$topic_summary,
         width = 10, height = 8)
plot.STM(stm.out_data, type = "summary", text.cex = 0.8, n = 10)
dev.off()

CairoPDF(OUTPUT_FILES$topic_relevance,
         width = 10, height = 8)
corr.out_data <- topicCorr(stm.out_data, method = "huge")
plot.topicCorr(corr.out_data, vertex.color = "grey")
dev.off()

write.csv(output_df, 
          OUTPUT_FILES$topic_keywords, 
          row.names = FALSE,
          fileEncoding = "UTF-8")

write.csv(topic_mapping,
          OUTPUT_FILES$topic_mapping,
          row.names = FALSE)

# === 7. Topic trend plots ======
theta_df <- as.data.frame(stm.out_data$theta)
theta_df$Date <- meta_data$Date

unique_dates <- sort(unique(theta_df$Date))
theta_by_date <- matrix(0, ncol=K_TOPICS, nrow=length(unique_dates))

for (i in seq_along(unique_dates)) {
  subset <- theta_df[which(theta_df$Date == unique_dates[i]), 1:K_TOPICS]
  theta_by_date[i,] <- colMeans(as.matrix(subset))
}

trend_df <- data.frame()
for (i in 1:K_TOPICS) {
  topic_i <- data.frame(
    topic.prevalence = theta_by_date[,i],
    date = unique_dates,
    topic = factor(paste("Topic", i), levels = paste("Topic", 1:K_TOPICS))
  )
  trend_df <- rbind(trend_df, topic_i)
}

year_blocks <- data.frame(
  start = as.Date(c("2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01")),
  end = as.Date(c("2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31")),
  fill = c("grey90", "white", "grey90", "white", "grey90")
)

topic_trend_plot <- ggplot(trend_df) +
  geom_rect(data = year_blocks, 
            aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf, fill = fill),
            alpha = 0.3, inherit.aes = FALSE) +
  scale_fill_identity() +
  geom_point(aes(x=date, y=topic.prevalence, group=1), 
             size=0.5,
             color="#002FA7") +
  geom_line(aes(x=date, y=topic.prevalence, group=1), 
            color="#002FA7", linewidth=0.5) +
  stat_smooth(aes(x=date, y=topic.prevalence, group=1), 
              method = "lm", se = FALSE, color="#800020", linetype="dashed") +
  facet_wrap(~topic, nrow=10, ncol=3, scales="free") +
  ylab("Topic prevalence") +
  xlab("Year") +
  scale_x_date(date_breaks = "1 year",
               date_labels = "%Y",
               expand = c(0, 0),
               limits = c(as.Date("2020-01-01"), as.Date("2024-12-31"))) +
  theme_minimal()

cairo_pdf(filename = OUTPUT_FILES$topic_trends, width = 12, height = 15, onefile = TRUE)
print(topic_trend_plot + 
        theme(text = element_text(family = "Times New Roman")) +
        scale_x_date(labels = scales::date_format("%Y")))
dev.off()

fit_results <- data.frame(
  Topic = 1:K_TOPICS,
  Slope = numeric(K_TOPICS),
  P_value = numeric(K_TOPICS),
  R_squared = numeric(K_TOPICS),
  Significance = character(K_TOPICS)
)

for (i in 1:K_TOPICS) {
  topic_data <- trend_df[trend_df$topic == paste("Topic", i),]
  fit <- lm(topic.prevalence ~ as.numeric(date), data=topic_data)
  
  fit_results$Slope[i] <- coef(fit)[2]
  fit_results$P_value[i] <- summary(fit)$coefficients[2,4]
  fit_results$R_squared[i] <- summary(fit)$r.squared
  fit_results$Significance[i] <- ifelse(fit_results$P_value[i] < 0.05, "Significant", "Not Significant")
}

write.csv(fit_results, OUTPUT_FILES$topic_trends_fit, row.names = FALSE)

# === 8. Save model ==================
stm.out_data$meta <- out_data$meta
saveRDS(stm.out_data, file = file.path(MAIN_DIR, "STM model.rds"))
save(new_data, new_doc_topics, meta_data, file = file.path(MAIN_DIR, "4.STM step3.RData"))