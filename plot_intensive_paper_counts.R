library(ggplot2)
library(dplyr)
library(tidyr)

load_study_concept_data <- function() {
  # Read psych_datasets and filter for valid studies
  psych_df <- read.csv("data/psych_datasets.csv")
  psych_df <- psych_df %>% filter(is_valid == TRUE)
  
  # Read brains_vs_ai for ca_concept and level mapping
  concepts_df <- read.csv("data/brains_vs_ai.csv") %>%
    filter(!is.na(ca_concept) & ca_concept != "NA") %>%
    select(ca_concept, level) %>%
    distinct()
  
  # Create a long-format dataframe: one row per study-concept pair
  study_concept_list <- list()
  
  for (i in seq_len(nrow(psych_df))) {
    study_name <- psych_df$short_name[i]
    tasks_str <- psych_df$tasks_list[i]
    
    if (!is.na(tasks_str) && tasks_str != "") {
      # Split tasks_list on ", "
      tasks <- trimws(unlist(strsplit(tasks_str, ", ")))
      
      for (task in tasks) {
        study_concept_list[[length(study_concept_list) + 1]] <- data.frame(
          study = study_name,
          ca_concept = task,
          stringsAsFactors = FALSE
        )
      }
    }
  }
  
  # Combine into a single dataframe
  if (length(study_concept_list) > 0) {
    study_concept_df <- do.call(rbind, study_concept_list)
  } else {
    stop("No study-concept pairs found")
  }
  
  # Merge with concepts_df to get level information
  study_concept_df <- study_concept_df %>%
    left_join(concepts_df, by = "ca_concept") %>%
    filter(!is.na(level))
  
  # Order concepts by level then alphabetically
  all_concepts <- concepts_df %>%
    arrange(level, ca_concept)
  
  return(list(
    study_concept_df = study_concept_df,
    concepts_df = concepts_df,
    all_concepts = all_concepts
  ))
}

# =============================================================================
# Heatmap: Studies vs ca_concepts
# =============================================================================

plot_study_concept_heatmap <- function(data = NULL) {
  if (is.null(data)) data <- load_study_concept_data()
  
  study_concept_df <- data$study_concept_df
  all_concepts <- data$all_concepts
  
  # Mark presence with 1
  study_concept_df$present <- 1
  
  # Create a complete grid of all studies x concepts
  all_studies <- unique(study_concept_df$study)
  
  complete_grid <- expand.grid(
    study = all_studies,
    ca_concept = all_concepts$ca_concept,
    stringsAsFactors = FALSE
  )
  
  # Merge to get presence/absence
  heatmap_df <- complete_grid %>%
    left_join(study_concept_df %>% select(study, ca_concept, present),
              by = c("study", "ca_concept")) %>%
    left_join(all_concepts, by = "ca_concept") %>%
    mutate(present = ifelse(is.na(present), 0, 1))
  
  # Order ca_concept by level then alphabetically
  heatmap_df <- heatmap_df %>%
    mutate(ca_concept = factor(ca_concept, levels = all_concepts$ca_concept))
  
  # Order studies alphabetically (descending for proper y-axis order)
  heatmap_df <- heatmap_df %>%
    mutate(study = factor(study, levels = sort(unique(study), decreasing = TRUE)))
  
  # Create the heatmap
  p_heatmap <- ggplot(heatmap_df, aes(x = ca_concept, y = study, fill = factor(present))) +
    geom_tile(color = "white", linewidth = 0.5) +
    scale_fill_manual(values = c("0" = "gray90", "1" = "steelblue"),
                      labels = c("No", "Yes"),
                      name = "Addressed") +
    facet_grid(. ~ level, scales = "free_x", space = "free_x") +
    labs(title = "fMRI Studies vs Liu et al. categorization",
         x = "Cognitive Atlas",
         y = "Study") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      axis.text.y = element_text(size = 8),
      strip.text = element_text(face = "bold"),
      panel.grid = element_blank(),
      legend.position = "bottom"
    )
  
  ggsave("plots/study_concept_heatmap.png", p_heatmap, width = 14, height = 10)
  print(p_heatmap)
  cat("Heatmap saved to: study_concept_heatmap.png\n")
  
  return(p_heatmap)
}

# =============================================================================
# Histogram: Count of studies per ca_concept
# =============================================================================

plot_study_concept_histogram <- function(data = NULL) {
  if (is.null(data)) data <- load_study_concept_data()
  
  study_concept_df <- data$study_concept_df
  all_concepts <- data$all_concepts
  
  # Count studies per ca_concept
  concept_counts <- study_concept_df %>%
    group_by(ca_concept, level) %>%
    summarise(n_studies = n_distinct(study), .groups = "drop")
  
  # Add concepts with zero studies
  concept_counts <- all_concepts %>%
    left_join(concept_counts, by = c("ca_concept", "level")) %>%
    mutate(n_studies = ifelse(is.na(n_studies), 0, n_studies))
  
  # Order ca_concept by level then alphabetically
  concept_counts <- concept_counts %>%
    mutate(ca_concept = factor(ca_concept, levels = all_concepts$ca_concept))
  
  # Create the histogram
  p_hist <- ggplot(concept_counts, aes(x = ca_concept, y = n_studies, fill = level)) +
    geom_col() +
    facet_grid(. ~ level, scales = "free_x", space = "free_x") +
    scale_fill_brewer(palette = "Set2") +
    labs(title = "Number of fMRI Studies Addressing Each Cognitive Concept",
         x = "Cognitive Atlas",
         y = "Number of Studies") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      strip.text = element_text(face = "bold"),
      panel.grid.major.x = element_blank(),
      legend.position = "none"
    )
  
  ggsave("plots/study_concept_histogram.png", p_hist, width = 14, height = 6)
  print(p_hist)
  cat("Histogram saved to: study_concept_histogram.png\n")
  
  return(p_hist)
}

# =============================================================================
# Histogram: Count of studies per level (any subconcept)
# =============================================================================

plot_study_level_histogram <- function(data = NULL) {
  if (is.null(data)) data <- load_study_concept_data()
  
  study_concept_df <- data$study_concept_df
  
  # Count distinct studies per level (any subconcept)
  level_counts <- study_concept_df %>%
    group_by(level) %>%
    summarise(n_studies = n_distinct(study), .groups = "drop")
  
  # Ensure all levels are present
  all_levels <- data.frame(level = c("L1", "L2", "L3"))
  level_counts <- all_levels %>%
    left_join(level_counts, by = "level") %>%
    mutate(n_studies = ifelse(is.na(n_studies), 0, n_studies))
  
  # Create the histogram
  p_level <- ggplot(level_counts, aes(x = level, y = n_studies, fill = level)) +
    geom_col() +
    scale_fill_brewer(palette = "Set2") +
    labs(title = "Intensive fMRI Studies Addressing Cognitive Levels",
         x = "Level",
         y = "Number of Studies") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(size = 12),
      panel.grid.major.x = element_blank(),
      legend.position = "none"
    )
  
  ggsave("plots/study_level_histogram.png", p_level, width = 5, height = 5)
  print(p_level)
  cat("Histogram saved to: study_level_histogram.png\n")
  
  return(p_level)
}

# =============================================================================
# Run all study-concept plots
# =============================================================================

# Load data once
study_data <- load_study_concept_data()

# Generate all plots using shared data
p_heatmap <- plot_study_concept_heatmap(study_data)
p_hist <- plot_study_concept_histogram(study_data)
p_level <- plot_study_level_histogram(study_data)
