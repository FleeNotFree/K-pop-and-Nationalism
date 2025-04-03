# === Configuration Section ===
# Main directory path
MAIN_DIR <- 'PATH/TO/FOLDER'

# Input/Output file paths
INPUT_FILE <- file.path(MAIN_DIR, 'tokenized.csv')
OUTPUT_FILES <- list(
  topic_summary = file.path(MAIN_DIR, '4.Topic summary figure.pdf'),
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

stm.out_data <- readRDS(file.path(MAIN_DIR, "4.STM model.rds"))
load(file.path(MAIN_DIR, "4.STM step3.RData"))

meta_data <- stm.out_data$meta

topic_labels <- labelTopics(stm.out_data, n = 10)
print(topic_labels)

# === 1. Defining frame ===
frames <- list(
  "Calling for actions" = c(16, 18, 20),
  "Chinese elements, symbols & artifacts" = c(1, 2, 8, 12, 19, 22),
  "Conflicts between idolizing & nationality" = c(3, 9, 15),
  "Devaluing & attacking South Korea" = c(6, 14),
  "Historical & political events & issues" = c(4, 7, 23),
  "Relevant parties' responses" = c(5, 11, 21, 24),
  "Routine K-pop engagement" = c(10, 13, 17)
)


# === 2. Frame thetas ===

frame_order <- names(frames)
legal_names <- make.names(frame_order)
frame_thetas <- matrix(0, nrow=nrow(stm.out_data$theta), ncol=length(frames))
colnames(frame_thetas) <- legal_names

for(i in seq_along(frame_order)) {
  frame_topics <- frames[[frame_order[i]]]
  frame_thetas[,legal_names[i]] <- rowSums(stm.out_data$theta[, frame_topics, drop=FALSE])
  meta_data[[legal_names[i]]] <- frame_thetas[,legal_names[i]]
}

if (any(frame_thetas > 1)) {
  warning("Some values in frame_thetas are greater than 1!")
} else {
  print("All values in frame_thetas are within the expected range.")
}

# === 3. Use lm() to calculate effect of covariates ===
frame_effects_event <- data.frame()
frame_effects_fan <- data.frame()

meta_data$fan <- as.factor(meta_data$fan)

for (frame_name in frame_order) {
  legal_name <- make.names(frame_name)
  
  formula <- as.formula(paste0("`", legal_name, "` ~ event * fan"))
  model <- lm(formula, data = meta_data)
  
  cat("\n==== Effects for frame:", frame_name, "====\n")
  print(summary(model))
  
  # Extract event effects
  event_coef <- summary(model)$coefficients["event", ]
  frame_effects_event <- rbind(frame_effects_event, data.frame(
    Frame = factor(frame_name, levels = frame_order),
    Mean_Effect = event_coef["Estimate"],
    CI_Lower = event_coef["Estimate"] - 1.96 * event_coef["Std. Error"],
    CI_Upper = event_coef["Estimate"] + 1.96 * event_coef["Std. Error"],
    P_Value = event_coef["Pr(>|t|)"],
    Significance = factor(ifelse(event_coef["Pr(>|t|)"] < 0.05, 
                                 "Significant", "Not Significant"),
                          levels = c("Significant", "Not Significant"))
  ))
  
  # Extract event effects
  fan_levels <- grep("^fan", rownames(summary(model)$coefficients), value = TRUE)
  
  # fan1 => Low Engagement，fan2 => Medium & High Engagement
  fan_label_map <- c("fan1" = "Low Engagement", "fan2" = "Medium & High Engagement")
  
  for (fan_level in fan_levels) {
    if (fan_level %in% rownames(summary(model)$coefficients)) {
      fan_coef <- summary(model)$coefficients[fan_level, ]

      fan_label <- fan_label_map[fan_level]
      
      frame_effects_fan <- rbind(frame_effects_fan, data.frame(
        Frame = factor(frame_name, levels = frame_order),
        Fan = factor(fan_label, levels = c("Low Engagement", "Medium & High Engagement")),
        Mean_Effect = fan_coef["Estimate"],
        CI_Lower = fan_coef["Estimate"] - 1.96 * fan_coef["Std. Error"],
        CI_Upper = fan_coef["Estimate"] + 1.96 * fan_coef["Std. Error"],
        P_Value = fan_coef["Pr(>|t|)"],
        Significance = factor(ifelse(fan_coef["Pr(>|t|)"] < 0.05, 
                                     "Significant", "Not Significant"),
                              levels = c("Significant", "Not Significant"))
      ))
    } else {
      warning("Fan coefficient not found for frame: ", frame_name)
    }
  }
}

write.csv(frame_effects_event, 
          OUTPUT_FILES$frame_event_effects_csv, 
          row.names = FALSE)

write.csv(frame_effects_fan, 
          OUTPUT_FILES$frame_fan_effects_csv, 
          row.names = FALSE)

# === 5. Figures ===
plot.event <- function(data) {
  data$Frame <- factor(data$Frame, levels = rev(frame_order))

  breaks_range <- scales::extended_breaks(n = 6)(range(data$CI_Lower, data$CI_Upper))
  x_min <- min(breaks_range)
  x_max <- max(breaks_range)

  plot_xlim <- c(min(data$CI_Lower) - 0.03, max(data$CI_Upper) + 0.06)
  plot_ylim <- c(-0.3, 8.5)

  legend_df <- data.frame(
    x = rep(max(data$CI_Upper) + 0.02, 2),
    y = c(7, 6.3),
    shape = c(16, 17),
    label = c("Significant", "Not Significant")
  )

  event_labels_df <- data.frame(
    x = c(x_min - 0.025, x_max + 0.025),
    y = c(-2.3, -2.3),
    label = c("Cultural Appropriation Disputes", 
              "Suspicions of Humiliating China")
  )
  # event=0: Cultural Appropriation Disputes (reference group)
  # event=1: Suspicions of Humiliating China
  
  axis_label_df <- data.frame(
    x = 0,
    y = -3,
    label = "Coefficient"
  )
  
  p <- ggplot(data, aes(y = Frame, x = Mean_Effect)) +

    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_errorbarh(aes(xmin = CI_Lower, xmax = CI_Upper), 
                   height = 0, color = "#203A66") +
    geom_point(aes(shape = Significance), size = 3, color = "#002FA7") +  # 添加均值图标

    geom_text(data = event_labels_df, aes(x = x, y = y, label = label),
              family = "Times New Roman", size = 5, hjust = 0.5) +

    geom_text(data = axis_label_df, aes(x = x, y = y, label = label),
              family = "Times New Roman", size = 6, hjust = 0.5,
              fontface = "bold") +

    geom_point(data = legend_df, aes(x = x, y = y), 
               shape = legend_df$shape, size = 3) +
    geom_text(data = legend_df, 
              aes(x = x + 0.01, y = y, label = label),
              family = "Times New Roman", size = 4, hjust = 0) +

    scale_y_discrete(limits = levels(data$Frame)) +
    scale_shape_manual(values = c(Significant = 16, "Not Significant" = 17)) +
    scale_x_continuous(breaks = breaks_range) +
    labs(x = element_blank(), y = "") +
    theme_minimal() +
    theme(
      text = element_text(family = "Times New Roman"),
      axis.title.x = element_text(size = 14, margin = margin(t = 25, b = 0)),
      axis.text = element_text(size = 12),
      axis.text.x = element_text(margin = margin(t = 15, b = 0)),
      legend.position = "none",
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank(),
      plot.margin = margin(20, 80, 35, 15, "points")
    ) +
    coord_cartesian(clip = 'off',
                    xlim = plot_xlim,
                    ylim = plot_ylim)
  
  return(p)
}

ggsave(OUTPUT_FILES$frame_event_effects, 
       plot.event(frame_effects_event), 
       width = 10, height = 6,
       device = cairo_pdf)

plot.fan <- function(data) {
  data$Frame <- factor(data$Frame, levels = rev(frame_order))
  
  breaks_range <- scales::extended_breaks(n = 6)(range(data$CI_Lower, data$CI_Upper))
  x_min <- min(breaks_range)
  x_max <- max(breaks_range)
  
  LEGEND_FONT_SIZE <- 5
  LEGEND_DOT_SIZE <- 3
  LEGEND_SPACING <- 1.5
  LINE_SPACING <- 1.7
  
  group_levels <- c(
    "Low Engagement",      
    "Medium & High Engagement"
  )
  colors <- c("#EBC562", "#5B87DE")
  
  n_groups <- length(group_levels)
  line_positions <- seq(1, by = LINE_SPACING, length.out = length(frames) * n_groups)
  
  data$y_position <- line_positions[n_groups * (as.numeric(data$Frame) - 1) + 
                                      (n_groups - match(data$Fan, group_levels) + 1)]
  
  label_positions <- tapply(data$y_position, data$Frame, mean)
  
  plot_xlim <- c(min(data$CI_Lower) - 0.008, max(data$CI_Upper) + 0.016)
  plot_ylim <- c(-1, max(line_positions) + 1.5)
  legend_x_dot <- max(data$CI_Upper) + 0.015
  legend_x_text <- legend_x_dot + 1
  legend_y_start <- max(line_positions) - 2

  fan_legend_df <- data.frame(
    x = legend_x_dot,
    y = legend_y_start - (seq_along(group_levels)-1)*LEGEND_SPACING,
    label = group_levels,
    color = colors
  )

  sig_legend_y_start <- legend_y_start - n_groups*LEGEND_SPACING - 1
  sig_legend_df <- data.frame(
    x = legend_x_dot,
    y = c(sig_legend_y_start, sig_legend_y_start - LEGEND_SPACING),
    label = c("Significant", "Not Significant"),
    shape = c(16, 17)
  )

  axis_label_df <- data.frame(
    x = 0,
    y = -6,
    label = "Coefficient"
  )
  
  p <- ggplot(data) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_hline(yintercept = line_positions, color = "gray95", linewidth = 0.5) +
    geom_errorbarh(aes(y = y_position, xmin = CI_Lower, xmax = CI_Upper,
                       color = Fan), height = 0.2, linewidth = 0.5) +
    geom_point(aes(y = y_position, x = Mean_Effect, color = Fan,
                   shape = Significance), size = LEGEND_DOT_SIZE) +  # 添加均值图标

    geom_point(data = fan_legend_df,
               aes(x = x, y = y),
               color = fan_legend_df$color,
               size = LEGEND_DOT_SIZE) +
    geom_text(data = fan_legend_df,
              aes(x = x + 0.008, y = y, label = label),
              hjust = 0, vjust = 0.5,
              family = "Times New Roman",
              size = LEGEND_FONT_SIZE) +

    geom_point(data = sig_legend_df,
               aes(x = x, y = y),
               shape = sig_legend_df$shape,
               size = LEGEND_DOT_SIZE) +
    geom_text(data = sig_legend_df,
              aes(x = x + 0.008, y = y, label = label),
              hjust = 0, vjust = 0.5,
              family = "Times New Roman",
              size = LEGEND_FONT_SIZE) +

    geom_text(data = axis_label_df, aes(x = x, y = y, label = label),
              family = "Times New Roman", size = 6, hjust = 0.5,
              fontface = "bold") +
    scale_y_continuous(breaks = label_positions,
                       labels = levels(data$Frame)) +
    scale_color_manual(values = setNames(colors, group_levels)) +
    scale_shape_manual(values = c(Significant = 16, "Not Significant" = 17)) +
    scale_x_continuous(breaks = breaks_range) +
    labs(x = element_blank(), y = "") +
    theme_minimal() +
    theme(
      text = element_text(family = "Times New Roman"),
      axis.title.x = element_text(size = 16, margin = margin(t = 25, b = 0)),
      axis.text = element_text(size = 14),
      axis.text.x = element_text(margin = margin(t = 15, b = 0)),
      legend.position = "none",
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank(),
      plot.margin = margin(20, 200, 20, 15, "points")
    ) +
    coord_cartesian(clip = 'off',
                    xlim = plot_xlim,
                    ylim = plot_ylim)
  
  return(p)
}

ggsave(OUTPUT_FILES$frame_fan_effects, 
       plot.fan(frame_effects_fan),
       width = 12, height = 7,
       device = cairo_pdf)

# === 4. Posts with info ======
doc_frames <- rep(NA, nrow(new_data))
doc_frames[!is.na(new_doc_topics)] <- sapply(new_doc_topics[!is.na(new_doc_topics)], function(topic) {
  for(frame_name in names(frames)) {
    if(topic %in% frames[[frame_name]]) return(frame_name)
  }
  return(NA)
})

posts_df <- new_data
posts_df$Topic <- new_doc_topics
posts_df$Frame <- doc_frames

for(i in 1:K_TOPICS) {

  posts_df[[paste0("Topic", i, "_prob")]] <- NA
  
  for(j in 1:nrow(meta_data)) {
    original_idx <- which(new_data$link == meta_data$link[j])
    if(length(original_idx) == 1) {
      posts_df[[paste0("Topic", i, "_prob")]][original_idx] <- stm.out_data$theta[j, i]
    }
  }
}

for (frame_name in frame_order) {
  frame_topics <- frames[[frame_name]]
  frame_prob <- rowSums(stm.out_data$theta[, frame_topics, drop=FALSE])

  posts_df[[paste0("Frame_", make.names(frame_name), "_prob")]] <- NA

  for (j in 1:nrow(meta_data)) {
    original_idx <- which(new_data$link == meta_data$link[j])
    if (length(original_idx) == 1) {
      posts_df[[paste0("Frame_", make.names(frame_name), "_prob")]][original_idx] <- frame_prob[j]
    }
  }
}

write.csv(posts_df, 
          OUTPUT_FILES$posts, 
          row.names = FALSE,
          fileEncoding = "UTF-8")

# Save all necessary objects
save_objects <- list(
  frame_thetas = frame_thetas,
  frame_order = frame_order,
  frames = frames,
  meta_data = meta_data,
  new_data = new_data,
  new_doc_topics = new_doc_topics
)

# Save to RData file
save(list = names(save_objects), 
     file = file.path(MAIN_DIR, "4.frame_analysis.RData"), 
     envir = list2env(save_objects))
