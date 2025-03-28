MAIN_DIR <- 'PATH/TO/FOLDER'
df <- read.csv(file.path(MAIN_DIR, 'tokenized.csv'), stringsAsFactors = FALSE)

library(stm)
library(parallel)
library(ggplot2)
library(scales)

df$event <- as.numeric(df$event)
df$gender <- as.numeric(df$gender) 
df$fan <- as.factor(df$fan)

prep_and_search <- function(df) {
  print("Step 1: Starting data preparation...")
  df$Date <- as.Date(df$Date)
  df$DateIndex <- as.numeric(df$Date - min(df$Date, na.rm = TRUE))
  
  print("Step 2: Processing text...")
  processed <- textProcessor(
    documents = df$text,
    metadata = df,
    removestopwords = FALSE,    
    removenumbers = FALSE,      
    removepunctuation = FALSE,  
    stem = FALSE,
    wordLengths = c(2, Inf),
    verbose = TRUE
  )
  
  print("Step 3: Preparing documents...")
  out <- prepDocuments(processed$documents,
                       processed$vocab,
                       processed$meta,
                       lower.thresh = 1,
                       verbose = TRUE)
  
  k_values <- seq(6, 40, by = 2)
  print(paste("Step 4: Will compute", length(k_values), "K values:", paste(k_values, collapse=", ")))
  
  if (.Platform$OS.type == "windows") {
    print("Step 5: Running searchK on Windows...")
    raw_results <- searchK(documents = out$documents,
                           vocab = out$vocab,
                           K = k_values,
                           prevalence = ~ event + fan,
                           data = out$meta,
                           init.type = "Spectral")
    
    print("\nStep 6: Checking raw searchK output structure:")
    print(str(raw_results))
    
    print("\nStep 7: Saving raw results...")
    saveRDS(raw_results, file.path(MAIN_DIR, "raw_stm_results.rds"))
    
    print("\nStep 8: Checking individual metrics structures:")
    print("Semantic coherence structure:")
    print(str(raw_results$results$semcoh))
    print("Residuals structure:")
    print(str(raw_results$results$residual))
    print("Heldout structure:")
    print(str(raw_results$results$heldout))
    
    print("\nStep 9: Checking actual values:")
    print("K values:")
    print(k_values)
    print("Semantic coherence values:")
    print(raw_results$results$semcoh)
    print("Residuals values:")
    print(raw_results$results$residual)
    print("Heldout values:")
    print(raw_results$results$heldout)
    
  } else {
    print("Step 5: Running searchK on Unix-like system...")
    raw_results <- searchK(documents = out$documents,
                           vocab = out$vocab,
                           K = k_values,
                           prevalence = ~ event + fan,
                           data = out$meta,
                           cores = detectCores() - 1)
    
    # Same checking steps as Windows branch...
    print("\nStep 6: Checking raw searchK output structure:")
    print(str(raw_results))
    
    print("\nStep 7: Saving raw results...")
    saveRDS(raw_results, file.path(MAIN_DIR, "raw_stm_results.rds"))
    
    print("\nStep 8: Checking individual metrics structures:")
    print("Semantic coherence structure:")
    print(str(raw_results$results$semcoh))
    print("Residuals structure:")
    print(str(raw_results$results$residual))
    print("Heldout structure:")
    print(str(raw_results$results$heldout))
    
    print("\nStep 9: Checking actual values:")
    print("K values:")
    print(k_values)
    print("Semantic coherence values:")
    print(raw_results$results$semcoh)
    print("Residuals values:")
    print(raw_results$results$residual)
    print("Heldout values:")
    print(raw_results$results$heldout)
  }
  
  print("\nStep 10: Creating final data frame...")
  results <- data.frame(
    K = k_values,
    semantic_coherence = unlist(raw_results$results$semcoh),
    residuals = unlist(raw_results$results$residual),
    heldout = unlist(raw_results$results$heldout)
  )
  
  print("\nStep 11: Checking final data frame structure:")
  print(str(results))
  
  return(results)
}

print("Starting main execution...")
results <- prep_and_search(df)

print("\nStep 12: Writing final results to CSV...")
write.csv(results, file.path(MAIN_DIR, "2.k_metrics.csv"), row.names = FALSE)
print("Done!")