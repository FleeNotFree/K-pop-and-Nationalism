# === Configuration Section ===
# Main directory path
MAIN_DIR <- 'PATH/TO/FOLDER'

# Load necessary libraries
library(mgcv)
library(progress)
library(ggplot2)
library(gridExtra)
library(scales)
library(extrafont)
loadfonts()

# Load saved data from STEP4
load(file.path(MAIN_DIR, "4.frame_analysis.RData"))

# Load STM model if needed for additional analysis
stm.out_data <- readRDS(file.path(MAIN_DIR, "4.STM model.rds"))

# === GAM 时间趋势 ===
library(mgcv)
library(progress)

frame_effects <- list()
fit_metrics <- data.frame()

pb <- progress_bar$new(total = length(frame_order))

for (frame_name in frame_order) {
  cat("Processing frame:", frame_name, "\n")
  pb$tick()
  
  frame_topics <- frames[[frame_name]]
  
  formula <- as.formula(paste0("frame_thetas[, '", make.names(frame_name), "'] ~ s(DateIndex)"))
  model <- gam(formula, 
               family = betar(link = "logit"),
               method = "REML",
               data = meta_data)

  frame_effects[[frame_name]] <- model

  model_summary <- summary(model)
  k_check_result <- k.check(model)

  fit_metrics <- rbind(fit_metrics, data.frame(
    Frame = frame_name,
    R_squared = if (!is.null(model_summary$r.sq)) model_summary$r.sq else NA,
    AIC = if (!is.null(AIC(model))) AIC(model) else NA,
    REML_score = if (!is.null(model_summary$sp.criterion)) model_summary$sp.criterion else NA,
    Deviance_Explained = if (!is.null(model_summary$dev.expl)) model_summary$dev.expl else NA,
    Scale = if (!is.null(model_summary$scale)) model_summary$scale else NA,
    LogLik = if (!is.null(logLik(model))) as.numeric(logLik(model)) else NA,
    DF_Residual = if (!is.null(model_summary$residual.df)) model_summary$residual.df else NA,
    EDF = if (!is.null(model_summary$edf)) model_summary$edf else NA,
    Dispersion = if (!is.null(model_summary$dispersion)) model_summary$dispersion else NA,
    Chi_Sq = if (!is.null(model_summary$chi.sq)) model_summary$chi.sq else NA,
    K_index = k_check_result[, "k-index"],
    K_p_value = k_check_result[, "p-value"]
  ))
}

write.csv(fit_metrics, file.path(MAIN_DIR, "4.gam_fit_metrics.csv"), row.names = FALSE)

# === Helper functions ===
save_coefficient_tables <- function(frame_effects) {
  all_coef_data <- data.frame()
  
  for (frame_name in names(frame_effects)) {
    model <- frame_effects[[frame_name]]
    coef_table <- summary(model)$p.table
    
    coef_df <- data.frame(
      frame = frame_name,
      coefficient = rownames(coef_table),
      estimate = coef_table[, "Estimate"],
      std_error = coef_table[, "Std. Error"],
      z_value = coef_table[, "z value"],
      p_value = coef_table[, "Pr(>|z|)"]
    )
    all_coef_data <- rbind(all_coef_data, coef_df)
  }
  
  write.csv(all_coef_data,
            file.path(MAIN_DIR, "4.frame_coefficients.csv"),
            row.names = FALSE)
}

save_coefficient_tables(frame_effects)


# === Plot functions ===
get_time_trends <- function(model, covariate = NULL, value = NULL) {
  pred_data <- data.frame(
    Date = seq(min(meta_data$Date), max(meta_data$Date), length.out = 100),
    DateIndex = as.numeric(seq(min(meta_data$Date), max(meta_data$Date), length.out = 100) - min(meta_data$Date)),  # 添加DateIndex
    event = if (!is.null(covariate) && covariate == "event") value else 0,
    fan = if (!is.null(covariate) && covariate == "fan") value else 0
  )

  pred <- predict(model, newdata = pred_data, type = "response", se.fit = TRUE)
  pred_data$mean <- pred$fit
  pred_data$ci_lower <- pred$fit - 1.96 * pred$se.fit
  pred_data$ci_upper <- pred$fit + 1.96 * pred$se.fit
  
  if (!is.null(covariate)) {
    pred_data$group <- as.character(value)
  }
  
  return(pred_data)
}

# === Plot =======
plot_frame_time <- function(frame_effects) {
  all_frames_data <- data.frame()
  plots <- list()
  
  for (frame_name in names(frame_effects)) {
    model <- frame_effects[[frame_name]]
    pred_df <- get_time_trends(model)
    pred_df$frame <- frame_name
    all_frames_data <- rbind(all_frames_data, pred_df)

    year_breaks <- seq.Date(from = as.Date(paste0(format(min(meta_data$Date), "%Y"), "-01-01")),
                            to = as.Date(paste0(format(max(meta_data$Date), "%Y"), "-01-01")),
                            by = "year")
    year_labels <- format(year_breaks, "%Y")

    year_midpoints <- year_breaks[-length(year_breaks)] + diff(year_breaks) / 2
    
    p <- ggplot(pred_df, aes(x = Date, y = mean)) +
      geom_line(color = "#002FA7", linewidth = 0.5) +
      geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.2) +

      geom_vline(xintercept = year_breaks, color = "gray60", linetype = "dashed", linewidth = 0.3) +

      scale_x_date(
        breaks = year_midpoints,
        labels = year_labels[-length(year_labels)],
        expand = expansion(mult = c(0.02, 0.02))
      ) +
      scale_y_continuous(
        limits = range(c(pred_df$mean, pred_df$ci_lower, pred_df$ci_upper)),
        expand = expansion(mult = c(0.1, 0.1))
      ) +
      labs(
        title = frame_name,
        x = "Year",
        y = "Frame Prevalence"
      ) +
      theme_minimal() +
      theme(
        text = element_text(family = "Times New Roman"),
        plot.title = element_text(size = 10, hjust = 0.5),
        axis.text = element_text(size = 8),
        axis.title.x = element_text(margin = margin(t = 10))
      )
    
    plots[[frame_name]] <- p
  }
  # Save
  all_frames_data$Date <- as.Date(all_frames_data$DateIndex, origin = min(meta_data$Date))
  write.csv(all_frames_data, 
            file.path(MAIN_DIR, "4.frame_time_all.csv"),
            row.names = FALSE)

  pdf(file.path(MAIN_DIR, "4.frame_time_trends.pdf"), 
      width = 12, height = 3 * ceiling(length(plots) / 3))
  do.call(grid.arrange, c(plots, list(ncol = 3)))
  dev.off()
}

# === 10. Extract effect ===
plot_frame_time(frame_effects)