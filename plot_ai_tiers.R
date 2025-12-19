# Plot AI benchmark tier counts for Claude 4.5, Gemini 3 Pro, and GPT 5.2
# Three-panel plot showing L1, L2, L3 counts (divided by 25 replicates)

library(ggplot2)
library(dplyr)
library(tidyr)

setwd("/Users/patrickmineault/LocalDocuments/taxonomy/")

# Load data files
claude_df <- read.csv("data/claude_45_tiers.csv")
gemini_df <- read.csv("data/gemini_3_tiers.csv")
gpt_df <- read.csv("data/gpt_52_tiers.csv")

# Function to compute level counts (sum of counts / 25)
compute_level_counts <- function(df, source_name) {
  data.frame(
    Level = c("L1", "L2", "L3"),
    n_benchmarks = c(
      sum(df$L1_count, na.rm = TRUE) / 25,
      sum(df$L2_count, na.rm = TRUE) / 25,
      sum(df$L3_count, na.rm = TRUE) / 25
    ),
    source = source_name
  )
}

# Compute level counts for each model
claude_level_counts <- compute_level_counts(claude_df, "Claude 4.5")
gemini_level_counts <- compute_level_counts(gemini_df, "Gemini 3 Pro")
gpt_level_counts <- compute_level_counts(gpt_df, "GPT 5.2")

# Combine all three
combined_level_counts <- bind_rows(
  claude_level_counts,
  gemini_level_counts,
  gpt_level_counts
)

# Order source factor for left-to-right panel order
combined_level_counts <- combined_level_counts %>%
  mutate(source = factor(source, levels = c("Claude 4.5", "Gemini 3 Pro", "GPT 5.2")))

# Create combined three-panel plot
p_combined <- ggplot(combined_level_counts, aes(x = Level, y = n_benchmarks, fill = Level)) +
  geom_col() +
  facet_wrap(~ source, nrow = 1) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Benchmarks by Cognitive Level Across AI Models",
       x = "Level",
       y = "# benchmarks") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 12),
    strip.text = element_text(face = "bold", size = 12),
    panel.grid.major.x = element_blank(),
    legend.position = "none"
  )

ggsave("plots/ai_tiers_by_level_combined.png", p_combined, width = 10, height = 4)
print(p_combined)

cat("\nAI tier plot saved to:\n")
cat("  - plots/ai_tiers_by_level_combined.png\n")
