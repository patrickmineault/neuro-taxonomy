# Plot paper counts from results/brains_vs_ai_paper_counts.csv
# Shows embedding-based paper counts by cognitive level

library(ggplot2)
library(dplyr)
library(tidyr)
library(ggrepel)

setwd("/Users/patrickmineault/LocalDocuments/taxonomy/")
df <- read.csv("results/brains_vs_ai_paper_counts.csv")
cols <- c("ca_concept_neurosynth_annotations_exact",
          "ca_concept_neurosynth_annotations_homonyms",
          "ca_concept_neuroquery_annotations_exact",
          "ca_concept_neurosynth_cogat",
          "ca_concept_neuroquery_cogat",
          "ca_concept_neurosynth_cogat_expanded",
          "ca_concept_neuroquery_cogat_expanded",
          "ca_concept_neuroquery_total_sim",
          "ca_concept_neuroquery_min_sum",
          "ca_concept_neurosynth_embedding",
          "ca_concept_neuroquery_embedding")

for ( colname in cols ) {
  # Create jitter plot for each metric
  p1 <- ggplot(df %>% filter(!is.na(colname)),
               aes(x = level, y = df[[colname]])) +
    geom_jitter(width = 0.2, height = 0, alpha = 0.7, size = 2) +
    geom_text_repel(aes(label = function.), size = 3, max.overlaps = 20) +
    labs(title = colname,
         x = "Level",
         y = "Number of Papers") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5))
  print(p1)
  ggsave(paste0("plots/", colname, ".png"), p1, width = 10, height = 8)

  # Create bar plot per level (sum of papers)
  level_counts <- df %>%
    filter(!is.na(.data[[colname]])) %>%
    group_by(level) %>%
    summarise(n_papers = sum(.data[[colname]], na.rm = TRUE), .groups = "drop")

  p2 <- ggplot(level_counts, aes(x = level, y = n_papers, fill = level)) +
    geom_col() +
    scale_fill_brewer(palette = "Set2") +
    labs(title = paste0(colname, " (by level)"),
         x = "Level",
         y = "Number of Papers") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(size = 12),
      panel.grid.major.x = element_blank(),
      legend.position = "none"
    )
  print(p2)
  ggsave(paste0("plots/", colname, "_by_level.png"), p2, width = 4.5, height = 4.5)
}

# =============================================================================
# Paper categories from Claude classification
# =============================================================================

# Load taxonomy for level mapping (from brains_vs_ai.csv)
taxonomy_df <- read.csv("data/brains_vs_ai.csv") %>%
  select(function., level) %>%
  filter(!is.na(function.) & function. != "NA") %>%
  distinct() %>%
  rename(Capacity = function., Level = level)

# Define data sources
data_sources <- list(
  list(
    name = "neuroquery",
    label = "Neuroquery",
    files = c("results/paper_categories_neuroquery.csv")
  ),
  list(
    name = "biorxiv_psyarxiv_2025",
    label = "biorxiv/psyarxiv 2025",
    files = c("results/paper_categories_biorxiv_neuroscience_abstracts_2025.csv",
              "results/paper_categories_psyarxiv_neuroscience_abstracts_2025.csv")
  )
)

for (source in data_sources) {
  cat(paste0("\nProcessing: ", source$label, "\n"))

  # Load and combine paper categories from all files for this source
  papers_list <- list()
  for (f in source$files) {
    if (file.exists(f)) {
      papers_list[[f]] <- read.csv(f)
      cat(paste0("  Loaded ", nrow(papers_list[[f]]), " papers from ", f, "\n"))
    } else {
      cat(paste0("  Warning: File not found: ", f, "\n"))
    }
  }

  if (length(papers_list) == 0) {
    cat(paste0("  Skipping ", source$label, " - no files found\n"))
    next
  }

  papers_df <- bind_rows(papers_list)
  cat(paste0("  Total papers: ", nrow(papers_df), "\n"))

  # Count papers per category (first category only)
  category1_counts <- papers_df %>%
    filter(!is.na(category_1) & category_1 != "") %>%
    group_by(category_1) %>%
    summarise(n_papers = n(), .groups = "drop") %>%
    left_join(taxonomy_df %>% select(Capacity, Level),
              by = c("category_1" = "Capacity")) %>%
    arrange(Level, category_1)

  # Order by level then alphabetically
  category1_counts <- category1_counts %>%
    mutate(category_1 = factor(category_1, levels = category1_counts$category_1))

  # Plot: Jitter plot with labels
  p_cat_jitter <- ggplot(category1_counts %>% filter(!is.na(Level)),
                         aes(x = Level, y = n_papers)) +
    geom_jitter(width = 0.2, height = 0, alpha = 0.7, size = 2) +
    geom_text_repel(aes(label = category_1), size = 3, max.overlaps = 20) +
    labs(title = paste0(source$label, " Papers per Cognitive Capacity (Primary Category)"),
         x = "Level",
         y = "Number of Papers") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

  ggsave(paste0("plots/paper_categories_jitter_", source$name, ".png"),
         p_cat_jitter, width = 10, height = 8)
  print(p_cat_jitter)

  # Plot: Papers per category (first category only) - bar chart
  p_cat1 <- ggplot(category1_counts, aes(x = category_1, y = n_papers, fill = Level)) +
    geom_col() +
    facet_grid(. ~ Level, scales = "free_x", space = "free_x") +
    scale_fill_brewer(palette = "Set2") +
    labs(title = paste0(source$label, " Papers per Cognitive Capacity (Primary Category)"),
         x = "Cognitive Capacity",
         y = "Number of Papers") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      strip.text = element_text(face = "bold"),
      panel.grid.major.x = element_blank(),
      legend.position = "none"
    )

  ggsave(paste0("plots/paper_categories_primary_", source$name, ".png"),
         p_cat1, width = 14, height = 6)
  print(p_cat1)

  # Count papers per category (both categories combined)
  # Reshape to long format
  papers_long <- papers_df %>%
    select(id, category_1, category_2) %>%
    pivot_longer(cols = c(category_1, category_2),
                 names_to = "category_rank",
                 values_to = "category") %>%
    filter(!is.na(category) & category != "")

  category_all_counts <- papers_long %>%
    group_by(category) %>%
    summarise(n_papers = n(), .groups = "drop") %>%
    left_join(taxonomy_df %>% select(Capacity, Level),
              by = c("category" = "Capacity")) %>%
    arrange(Level, category)

  # Order by level then alphabetically
  category_all_counts <- category_all_counts %>%
    mutate(category = factor(category, levels = category_all_counts$category))

  # Plot: Papers per category (both categories)
  p_cat_all <- ggplot(category_all_counts, aes(x = category, y = n_papers, fill = Level)) +
    geom_col() +
    facet_grid(. ~ Level, scales = "free_x", space = "free_x") +
    scale_fill_brewer(palette = "Set2") +
    labs(title = paste0(source$label, " Papers per Cognitive Capacity (Primary + Secondary)"),
         x = "Cognitive Capacity",
         y = "Number of Papers") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      strip.text = element_text(face = "bold"),
      panel.grid.major.x = element_blank(),
      legend.position = "none"
    )

  ggsave(paste0("plots/paper_categories_all_", source$name, ".png"),
         p_cat_all, width = 14, height = 6)
  print(p_cat_all)

  # Plot: Papers per level (first category only)
  level_counts <- category1_counts %>%
    group_by(Level) %>%
    summarise(n_papers = sum(n_papers), .groups = "drop") %>%
    filter(!is.na(Level))

  p_level <- ggplot(level_counts, aes(x = Level, y = n_papers, fill = Level)) +
    geom_col() +
    scale_fill_brewer(palette = "Set2") +
    labs(title = paste0(source$label, " Papers by Cognitive Level"),
         x = "Level",
         y = "Number of Papers") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(size = 12),
      panel.grid.major.x = element_blank(),
      legend.position = "none"
    )

  ggsave(paste0("plots/paper_categories_by_level_", source$name, ".png"),
         p_level, width = 5, height = 5)
  print(p_level)

  cat(paste0("Paper category plots saved for ", source$label, ":\n"))
  cat(paste0("  - plots/paper_categories_jitter_", source$name, ".png\n"))
  cat(paste0("  - plots/paper_categories_primary_", source$name, ".png\n"))
  cat(paste0("  - plots/paper_categories_all_", source$name, ".png\n"))
  cat(paste0("  - plots/paper_categories_by_level_", source$name, ".png\n"))
}

# =============================================================================
# Combined three-panel plot: Intensive Neuroimaging, Neuroquery, biorxiv/psyarxiv 2025
# =============================================================================

cat("\nCreating combined three-panel plot...\n")

# Panel 1: Intensive Neuroimaging (from psych_datasets.csv)
psych_df <- read.csv("data/psych_datasets.csv")
psych_df <- psych_df %>% filter(is_valid == TRUE)

concepts_df <- read.csv("data/brains_vs_ai.csv") %>%
  filter(!is.na(ca_concept) & ca_concept != "NA") %>%
  select(ca_concept, level) %>%
  distinct()

# Create study-concept pairs
study_concept_list <- list()
for (i in seq_len(nrow(psych_df))) {
  study_name <- psych_df$short_name[i]
  tasks_str <- psych_df$tasks_list[i]

  if (!is.na(tasks_str) && tasks_str != "") {
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

study_concept_df <- do.call(rbind, study_concept_list)
study_concept_df <- study_concept_df %>%
  left_join(concepts_df, by = "ca_concept") %>%
  filter(!is.na(level))

intensive_level_counts <- study_concept_df %>%
  group_by(level) %>%
  summarise(n_studies = n_distinct(study), .groups = "drop") %>%
  rename(Level = level, n_papers = n_studies)

# Ensure all levels are present
all_levels <- data.frame(Level = c("L1", "L2", "L3"))
intensive_level_counts <- all_levels %>%
  left_join(intensive_level_counts, by = "Level") %>%
  mutate(n_papers = ifelse(is.na(n_papers), 0, n_papers),
         source = "Intensive Neuroimaging")

# Panel 2: Neuroquery
neuroquery_papers <- NULL
if (file.exists("results/paper_categories_neuroquery.csv")) {
  neuroquery_papers <- read.csv("results/paper_categories_neuroquery.csv")
}

if (!is.null(neuroquery_papers)) {
  neuroquery_category1_counts <- neuroquery_papers %>%
    filter(!is.na(category_1) & category_1 != "") %>%
    group_by(category_1) %>%
    summarise(n_papers = n(), .groups = "drop") %>%
    left_join(taxonomy_df %>% select(Capacity, Level),
              by = c("category_1" = "Capacity"))

  neuroquery_level_counts <- neuroquery_category1_counts %>%
    group_by(Level) %>%
    summarise(n_papers = sum(n_papers), .groups = "drop") %>%
    filter(!is.na(Level))

  neuroquery_level_counts <- all_levels %>%
    left_join(neuroquery_level_counts, by = "Level") %>%
    mutate(n_papers = ifelse(is.na(n_papers), 0, n_papers),
           source = "Neuroquery")
} else {
  neuroquery_level_counts <- all_levels %>%
    mutate(n_papers = 0, source = "Neuroquery")
}

# Panel 3: biorxiv/psyarxiv 2025
biorxiv_psyarxiv_papers <- NULL
biorxiv_file <- "results/paper_categories_biorxiv_neuroscience_abstracts_2025.csv"
psyarxiv_file <- "results/paper_categories_psyarxiv_neuroscience_abstracts_2025.csv"

papers_list <- list()
if (file.exists(biorxiv_file)) {
  papers_list[[1]] <- read.csv(biorxiv_file)
}
if (file.exists(psyarxiv_file)) {
  papers_list[[2]] <- read.csv(psyarxiv_file)
}

if (length(papers_list) > 0) {
  biorxiv_psyarxiv_papers <- bind_rows(papers_list)

  biorxiv_psyarxiv_category1_counts <- biorxiv_psyarxiv_papers %>%
    filter(!is.na(category_1) & category_1 != "") %>%
    group_by(category_1) %>%
    summarise(n_papers = n(), .groups = "drop") %>%
    left_join(taxonomy_df %>% select(Capacity, Level),
              by = c("category_1" = "Capacity"))

  biorxiv_psyarxiv_level_counts <- biorxiv_psyarxiv_category1_counts %>%
    group_by(Level) %>%
    summarise(n_papers = sum(n_papers), .groups = "drop") %>%
    filter(!is.na(Level))

  biorxiv_psyarxiv_level_counts <- all_levels %>%
    left_join(biorxiv_psyarxiv_level_counts, by = "Level") %>%
    mutate(n_papers = ifelse(is.na(n_papers), 0, n_papers),
           source = "bioRxiv/PsyArXiv 2025")
} else {
  biorxiv_psyarxiv_level_counts <- all_levels %>%
    mutate(n_papers = 0, source = "bioRxiv/PsyArXiv 2025")
}

# Combine all three panels
combined_level_counts <- bind_rows(
  intensive_level_counts,
  neuroquery_level_counts,
  biorxiv_psyarxiv_level_counts
)

# Order source factor for left-to-right panel order
combined_level_counts <- combined_level_counts %>%
  mutate(source = factor(source, levels = c("Intensive Neuroimaging", "Neuroquery", "bioRxiv/PsyArXiv 2025")))

# Create combined three-panel plot
p_combined <- ggplot(combined_level_counts, aes(x = Level, y = n_papers, fill = Level)) +
  geom_col() +
  facet_wrap(~ source, scales = "free_y", nrow = 1) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Papers by Cognitive Level Across Data Sources",
       x = "Level",
       y = "Number of Papers") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 12),
    strip.text = element_text(face = "bold", size = 12),
    panel.grid.major.x = element_blank(),
    legend.position = "none"
  )

ggsave("plots/paper_categories_by_level_combined.png", p_combined, width = 10, height = 4)
print(p_combined)

cat("\nCombined three-panel plot saved to:\n")
cat("  - plots/paper_categories_by_level_combined.png\n")
